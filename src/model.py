from lib import *

from dataset import *
from utils import (
    set_specific_attr, 
    lmap, 
)

class PtbModel(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        embed = self.get_input_embeddings()
        model_kwargs["inputs_embeds"] = embed(inputs_tensor)
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        inputs_embeds=None, 
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        """
        ptb_input = inputs_embeds if self.config.ptb_param == 'embed' else encoder_outputs["last_hidden_state"]

        if hasattr(self, 'perturb_fn') and self.perturb_fn is not None:
            print(f'generate sentence using perturb parameter on {self.config.ptb_param}. ')
            ptb_output, self.batch_word_range = self.perturb_fn(ptb_input)

        ''' the ptb_input is same with ptb_output '''
        if self.config.ptb_param == 'embed':
            ptb_kwargs = {"inputs_embeds": ptb_input, }
        else:
            encoder_outputs["last_hidden_state"] = ptb_input
            ptb_kwargs = {"encoder_outputs": encoder_outputs, }
        """
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs, 
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            #**ptb_kwargs, 
        }

""" modify from BartForSequenceClassification """
class LatentClassfier(BartForSequenceClassification):
    pooling_options = {
        "mean": lambda x: torch.mean(x, dim=1), 
        "max": lambda x: torch.max(x, dim=1), 
        "sum": lambda x: torch.sum(x, dim=1), 
        "first": lambda x: x[:, 0, :], 
    }
    def __init__(self, config: BartConfig, pool_fn="mean", **kwargs):
        super().__init__(config, **kwargs)
        self.pool_fn = self.pooling_options[pool_fn]
    
    def set_requires_grad(self, requires_grad):
        for n, p in self.model.named_parameters():
            p.requires_grad = False 
        for n, p in self.classification_head.named_parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[1]  # encoder last hidden state

        sentence_representation = self.pool_fn(hidden_states)
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class FCN(nn.Module):
    def __init__(self, seq_len=None, d_model=1024, z_dim=128):
        assert seq_len is not None
        super(FCN, self).__init__()
        input_dim = seq_len * d_model
        hidden_dims = [int(input_dim * (0.25 ** i)) for i in range(4)] + [z_dim]
        encoder_layer_dims = [(0, 1), (1, 2), (2, 3), (3, 4)]
        decoder_layer_dims = [(4, 3), (3, 2), (2, 1), (1, 0)]
        hidden_dim_fn = lambda d: tuple(map(lambda x: hidden_dims[x], d))
        self.encoder = nn.Sequential(
            *[FCN.hidden_layer(*hidden_dim_fn(ld)) for ld in encoder_layer_dims[:-1]], 
        )
        self.enc_out_1 = nn.Linear(*hidden_dim_fn(encoder_layer_dims[-1]))
        self.enc_out_2 = nn.Linear(*hidden_dim_fn(encoder_layer_dims[-1]))
        self.decoder = nn.Sequential(
            *[FCN.hidden_layer(*hidden_dim_fn(ld)) for ld in decoder_layer_dims[:-1]], 
            FCN.hidden_layer(*hidden_dim_fn(decoder_layer_dims[-1]), activate_fn=nn.Tanh)
        )
    @staticmethod
    def hidden_layer(input_d, output_d, activate_fn=nn.ReLU):
        return nn.Sequential(
            nn.Linear(input_d, output_d), 
            activate_fn()
        )
    
    def forward(self, x, reparametrize_fn=None):
        if reparametrize_fn is None: 
            z, _ = self.encode(x)
            return self.decode(z)
        else: # vae
            mu, logvar = self.encode(x)
            z = reparametrize_fn(mu, logvar)
            return self.decode(z), mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        return self.enc_out_1(x), self.enc_out_2(x)

    def decode(self, z):
        return self.decoder(z)


class LatentVAE(nn.Module):
    def __init__(
        self, 
        input_dim=1024, 
        hidden_dim=256, 
        z_dim=128, # latent dim
        encoder_num_layers=1, 
        decoder_num_layers=1, 
        bidirectional=True, # only for encoder
        arch="lstm",
        decoder_input_type="origin", # `origin`, `constant`
    ):
        super(LatentVAE, self).__init__()
        layer_mul = 4 if bidirectional else 2
        self.hidden_dim = hidden_dim
        self.decoder_input_type = decoder_input_type
        self.seq_encoder = LatentVAE.single_cell(input_dim, hidden_dim, encoder_num_layers, bidirectional=bidirectional)
        self.enc_out_1 = nn.Linear(layer_mul * hidden_dim * encoder_num_layers, z_dim)
        self.enc_out_2 = nn.Linear(layer_mul * hidden_dim * encoder_num_layers, z_dim)
        self.decoder = nn.Linear(z_dim, 2 * hidden_dim * decoder_num_layers)
        self.seq_decoder = LatentVAE.single_cell(input_dim, hidden_dim, decoder_num_layers)
        self.decoder_head = nn.Linear(hidden_dim, input_dim)
        
        
    @staticmethod
    def single_cell(input_size, hidden_size, num_layers=1, bidirectional=False, arch="lstm"):
        return nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    
    @staticmethod 
    def fcn_layer(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim), 
            nn.ReLU(), 
        )

        # permute (1, 0 ) reshape (bz, -1)

    def forward(self, x, reparametrize_fn=None, perturb_fn=None, perturb_mode=None):
        if reparametrize_fn is None: 
            z, _ = self.encode(x)
            if perturb_mode == "latent": 
                print(f"generate predictions with perturb on {perturb_mode}")
                z = perturb_fn(z)
            x = self.decode(z)
            if perturb_mode == "enc_out": 
                print(f"generate predictions with perturb on {perturb_mode}")
                ret = perturb_fn(x)
        else: # vae
            mu, logvar = self.encode(x)
            z = reparametrize_fn(mu, logvar)
            if perturb_mode == "latent": 
                print(f"generate predictions with perturb on {perturb_mode}")
                z = perturb_fn(z)
            x = self.decode(x, z)
            if perturb_mode == "enc_out": 
                print(f"generate predictions with perturb on {perturb_mode}")
                x = perturb_fn(x)
            ret = x, mu, logvar

        return ret

    def encode(self, x):
        # input: (bz, s_len, z_dim)
        bz = x.shape[0]
        x = x.permute(1, 0, 2)
        output, (hn, cn) = self.seq_encoder(x)
        h = torch.concat((hn, cn), dim=-1)
        h = h.permute(0, 1, 2).reshape(bz, -1)
        return self.enc_out_1(h), self.enc_out_2(h)

    def decode(self, x, z):
        bz = z.shape[0]
        x = x.permute(1, 0, 2)
        h = self.decoder(z)
        h = h.reshape(bz, -1, 2 * self.hidden_dim).permute(1, 0, 2)
        h0, c0 = torch.split(h, self.hidden_dim, dim=-1)
        if self.decoder_input_type == "origin":
            input = x
        else: # constant
            input = torch.zeros(*x.shape).to(x.device)
        x, _ = self.seq_decoder(input, (h0.contiguous(), c0.contiguous()))
        rec_x = self.decoder_head(x)
        return rec_x.permute(1, 0, 2)

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def loss(self, recon_x, x, mu, logvar, kld_r=1e-5):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        mse = self.criterion(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        if hasattr(self, "counter"): # beta scheduling
            losses = (mse + self.L[self.counter] * KLD, mse, KLD, self.L[self.counter])
            self.counter += 1
        else:
            losses = (mse + kld_r * KLD, mse, KLD)
        return losses

    # beta scheduling
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
        self.counter = 0
        self.L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                self.L[int(i+c*period)] = v
                v += step
                i += 1
        return self.L 


class VAE_encoder(nn.Module):
    def __init__(self, encoder: BartEncoder, arch="latent", vae_kwargs={}):
        super().__init__()
        self.encoder = encoder
        self.arch = arch
        autoencoder_class = {
            "fcn": FCN, 
            "latent": LatentVAE, 
        }
        self.vae = autoencoder_class[arch](**vae_kwargs)
        self.vae_loss = VAELoss()

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(
        self,
        kld_r: Optional[float] = 1e-5, 
        **bart_encoder_kwargs, 
    ) -> Union[Tuple, BaseModelOutput]:

        encoder_outputs = self.encoder(**bart_encoder_kwargs)
        # encoder output transform
        bz, seq_len, d_model = encoder_outputs.last_hidden_state.shape
        
        if self.arch == "fcn":
            vae_input = encoder_outputs.last_hidden_state.reshape(bz, -1)
        else: # latent
            vae_input = encoder_outputs.last_hidden_state
        
        vae_output, mu, logvar = self.vae(
            vae_input, 
            reparametrize_fn=VAE_encoder.reparametrize, 
            perturb_fn=self.perturb_fn, 
            perturb_mode=self.perturb_mode, 
        )
        VAE_encoder.loss = self.vae_loss.loss(vae_output, vae_input, mu, logvar, kld_r=kld_r)
        
        if self.arch == "fcn":
            vae_output = vae_output.reshape(bz, seq_len, d_model)
        
        encoder_outputs.last_hidden_state = vae_output

        return encoder_outputs


MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
    "ptb": PtbModel, 
    "latent": LatentClassfier, 
}

# model that generates text conditioned on latent vector
class Seq2SeqModel(nn.Module):
    def __init__(self, args, **config_kwargs):
        super().__init__()
        args = set_specific_attr(args, get_model_specific_attr())
        self.args = args
        self.model_type = 'seq2seq'
        args.model_mode = 'ptb'
        self.config, self.tokenizer, self.model = get_model_from_args(args, **config_kwargs)
        self.set_ptb_attr()
        self.reset_encoder()

    def reset_encoder(self, arch="latent"):
        # the model should be bart-based
        if arch == "fcn":
            max_source_length = get_dataset_global("max_source_length")
            vae_kwargs = {"z_dim": self.args.z_dim, "seq_len": max_source_length}
        else: # latent
            vae_kwargs = dict(
                hidden_dim=self.args.hidden_dim, 
                z_dim=self.args.z_dim, # latent dim
                encoder_num_layers=self.args.encoder_num_layers, 
                decoder_num_layers=self.args.decoder_num_layers, 
                decoder_input_type=self.args.decoder_input_type, 
            )
        self.model.model.encoder = VAE_encoder(self.model.model.encoder, arch=self.args.use_vae, vae_kwargs=vae_kwargs)

    def set_ptb_attr(self):
        var_name = ['attr_dim', 'ptb_param']
        for var in var_name:
            v = getattr(self.args, var, None)
            setattr(self.model.config, var, v)

    def extra_settings(self, ):
        if isinstance(self.model, BartForConditionalGeneration):
            extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
            for p in extra_model_params:
                if getattr(self.args, p, None):
                    assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                    setattr(self.config, p, getattr(self.args, p))

        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        else:
            self.decoder_start_token_id = self.model.config.decoder_start_token_id

    def forward(self, **batch):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        #output_hidden_states = batch["output_hidden_states"]

        if 'labels' in batch:
            tgt_ids = batch["labels"]
            if isinstance(self.model, T5ForConditionalGeneration):
                decoder_input_ids = self.model._shift_right(tgt_ids)
            else:
                decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, \
                                        decoder_start_token_id=decoder_start_token_id)
        else:
            decoder_input_ids = None

        # outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
        #                use_prefix=True, return_dict=True, labels=tgt_ids)
        #
        # return (outputs.loss,)

        outputs = self.model(
            src_ids, 
            attention_mask=src_mask, 
            decoder_input_ids=decoder_input_ids, 
            use_cache=False,
            #output_hidden_states=output_hidden_states, 
        )
                        # use_prefix=True)

        inputs_embeds = self.model.get_input_embeddings()(src_ids)
        outputs.inputs_embeds = inputs_embeds
        outputs.vae_loss = VAE_encoder.loss

        return outputs

    def probing(self, batch: dict, 
        n_sample=10, 
        n_buffer=10, 
        l_step=1.0, 
        n_step=10, 
        eval_metric="rouge", 
        perturb_mode="latent", 
    ):
        buffer = []
        perturb_fn = lambda x: x + torch.randn(*x.shape).to(x.device) * l_step
        # todo: change to iteratively search
        new_batch = {}
        for key in ["input_ids", "attention_mask"]:
            new_batch[key] = batch[key].repeat_interleave(n_sample, dim=0).cuda()
        self.model = self.model.cuda()
        out_sents = self.generate(new_batch, return_ids=False, perturb_fn=perturb_fn, perturb_mode=perturb_mode)
        ref_sents = self.ids_to_clean_text(batch["labels"])
        
        output_list = []
        for i in range(len(ref_sents)):
            output = {"reference": ref_sents[i]}
            output.update({f"prediction_{n}": out_sents[k] for n, k in enumerate(range(i * n_sample, (i + 1) * n_sample))})
            output_list.append(output)
        
        return output_list
    
        # sample from gaussian distribution
        # evaluate them, choose top k and store them
        # use top k points as start points and do the same as above for each point

    def generate(self, batch: dict, return_ids=True, perturb_fn=None, perturb_mode="latent", visual_layer=False):
        print('for decoding, eval_max_length={}, eval_min_length={}, eval_beams={}'\
            .format(self.args.eval_max_gen_length, self.args.eval_min_gen_length, self.args.eval_beams))
        """ # todo: visual layers
        from torchvision import transforms
        def visual_layer_hook_fn(module, input, output):
            visual_dir = os.path.join(get_module_global("output_dir"), "visual")
            os.makedirs(visual_dir, exist_ok=True)
            for visual_state in [input, output]:
                if len(visual_state.shape) == 3: # visual as image
                    visual_size = (64, visual_state.shape[1])
                    for v in visual_state:
                        v = sigmoid(v.T.clone()) * 255.
                        v = v.long()
                        to_pil_fn = transforms.ToPILImage()
                        image = to_pil_fn(v)
                        resize_transform_fn = transforms.Resize(visual_size)
                        image = resize_transform_fn(image)
                        image.save() # todo

                    # resize
                    # save as image
                else: # store as numpy array
                    pass
        """
        seq2seq_model_type = (
            BartForConditionalGeneration, 
            GPT2LMHeadModel, 
        )
        if isinstance(self.model, GPT2LMHeadModel):
            # todo: set gpt parameters
            output_sequences = self.model.generate(
                input_ids=batch["input_ids"],
                emb_match=None,
                control_code=control_code,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=0.8,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )
        elif isinstance(self.model, BartForConditionalGeneration): 
            self.model.model.encoder.perturb_fn = perturb_fn
            self.model.model.encoder.perturb_mode = perturb_mode
            self.model.input_ids = batch["input_ids"]
            output_sequences = self.model.generate(
                batch["input_ids"],
                # past_key_values=None,
                attention_mask=batch["attention_mask"],
                use_cache=True,
                length_penalty=self.args.length_penalty,
                # use_prefix=True,
                decoder_start_token_id=self.config.decoder_start_token_id,
                num_beams=self.args.eval_beams,
                min_length=self.args.eval_min_gen_length,
                max_length=self.args.eval_max_gen_length,
            )
            if hasattr(self.model, 'batch_word_range'):
                self.batch_word_range = self.model.batch_word_range
        else: 
            assert isinstance(self.model, seq2seq_model_type)

        if return_ids:
            return output_sequences
        else:
            return self.ids_to_clean_text(output_sequences)

    def ids_to_clean_text(self, generated_ids: torch.Tensor):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        args = set_specific_attr(args, get_model_specific_attr())
        self.args = args
        self.model_type = 'cls'
        # args.model_mode = 'latent'
        self.config, self.tokenizer, self.model = get_model_from_args(args)

    def forward(self, **batch):
        input_batch = {k: v for k, v in batch.items() if k != 'ids'}
        outputs = self.model(**input_batch)
        return outputs

class TextGenerationWithVectorInputModel(Seq2SeqModel):
    def __init__(self, args, tgwv_args):
        config_kwargs = tgwv_args.__dict__
        self.original_embed_dim = 1024
        super().__init__(args, **config_kwargs)
        self.model_type = 'tgwv'
        
        embed_matrix = self.model.get_input_embeddings()
        self.embed_matrix = torch.nn.Embedding.from_pretrained(
            embed_matrix.weight[:, :self.original_embed_dim],
            padding_idx=self.tokenizer.pad_token_id, 
        )
        self.embed_matrix.weight.requires_grad = True

    # get batch from text_generation_with_vector_input_collate_fn
    def forward(self, **batch):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        cls_logits = batch['cls_logits']

        # create input embeddings
        embed = self.embed_matrix(src_ids)
        embed = torch.concat((embed, cls_logits), dim=-1)

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, \
                                    decoder_start_token_id=decoder_start_token_id)

        # outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
        #                use_prefix=True, return_dict=True, labels=tgt_ids)
        #
        # return (outputs.loss,)

        #outputs = self.model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)
        outputs = self.model(inputs_embeds=embed, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)
                        # use_prefix=True)

        return outputs

    def generate(self, batch: dict, return_ids=True):
        print('for decoding, eval_max_length={}, eval_min_length={}, eval_beams={}'\
            .format(self.args.eval_max_gen_length, self.args.eval_min_gen_length, self.args.eval_beams))

        seq2seq_model_type = (
            BartForConditionalGeneration, 
            GPT2LMHeadModel, 
        )
        if isinstance(self.model, GPT2LMHeadModel):
            # todo: set gpt parameters
            output_sequences = self.model.generate(
                input_ids=batch["input_ids"],
                emb_match=None,
                control_code=control_code,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=0.8,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )
        elif isinstance(self.model, BartForConditionalGeneration): 
            output_sequences = self.model.generate(
                batch["input_ids"],
                # past_key_values=None,
                attention_mask=batch["attention_mask"],
                use_cache=True,
                length_penalty=self.args.length_penalty,
                # use_prefix=True,
                decoder_start_token_id=self.config.decoder_start_token_id,
                num_beams=self.args.eval_beams,
                min_length=self.args.eval_min_gen_length,
                max_length=self.args.eval_max_gen_length,
            )
        else: 
            assert isinstance(self.model, seq2seq_model_type)

        if return_ids:
            return output_sequences
        else:
            return self.ids_to_clean_text(output_sequences)


    @staticmethod
    def add_specific_args():
        seq2seq_args_class = super(TextGenerationWithVectorInputModel, TextGenerationWithVectorInputModel).add_specific_args()
        @dataclass
        class TGWVModelArguments:
            d_model: Optional[int] = field(
                default=1024+16, 
                metadata={"help": "the embedding dim. "}
            )

        return [seq2seq_args_class, TGWVModelArguments]

def get_model_from_args(args, **config_kwargs):
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        **({"num_labels": args.num_labels} if args.num_labels is not None else {}),
        cache_dir=args.cache_dir,
        **config_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model_type = MODEL_MODES[args.model_mode]
    model = model_type.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=True, 
    )

    return config, tokenizer, model


def get_model_specific_attr():
    return ['model_name_or_path', 'config_name', 'tokenizer_name', 'cache_dir', 'use_big', 
                    'model_mode', 'num_labels']

if __name__ == '__main__':
    parser = HfArgumentParser((Seq2SeqModel.add_specific_args(), 
    ClassificationModel.add_specific_args(),
    Seq2SeqDataset.add_specific_args(), 
    ClassificationDataset.add_specific_args(), 
    DataModule.add_specific_args()))
    seq2seq_args, cls_args, seq_data_args, cls_data_args, loader_args = parser.parse_args_into_dataclasses()

    model = Seq2SeqModel(seq2seq_args)
    cls_model = ClassificationModel(cls_args)
    
    d_loader = DataModule(loader_args, seq_data_args, model.tokenizer, 'seq2seq')
    cls_loader = DataModule(loader_args, cls_data_args, cls_model.tokenizer, 'classification')
    for a, b in zip(d_loader.train_dataloader(), cls_loader.train_dataloader()):
        import pdb 
        pdb.set_trace()
        out = model(**a)
        out2 = cls_model(**b)
        out_s = model.generate(a, return_ids=False)
        
