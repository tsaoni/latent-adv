from lib import *
from perturb import *
from utils import *

import torch.nn as nn
from argparse import Namespace
from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForQuestionAnswering, 
    AutoModelForPreTraining, 
    AutoModelForTokenClassification, 
    AutoModelWithLMHead, 
    AutoModelForSeq2SeqLM, 

)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
    "paraphrase": AutoModelForSeq2SeqLM, 
}

class Seq2SeqWithNoisyAdapter(nn.Module):
    def __init__(
        self, 
        config, 
        model_name_or_path=None, 
    ):
        super().__init__()
        self.config = Namespace(
            model_name_or_path=model_name_or_path, 
            length_penalty=0.1,
            num_beams=4, 
            eval_min_gen_length=6, 
            eval_max_gen_length=50, 
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
        adp_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
        self.model.add_adapter("bottleneck_adapter", config=adp_config)
        self.model.set_active_adapters("bottleneck_adapter")
        #self.add_noisy_input_layer()

    def add_hook(self):
        def hook_fn(module, input, output):
            print(f"Module: {module} is used. ")
            #print(f"Input: {input}")
            #print(f"Output: {output}")
            self.inp, self.out = input, output
        hook = self.model.model.encoder.layers[0].attention_adapters.register_forward_hook(hook_fn)

    def add_noisy_input_layer(self):
        encoder_noisy_adapter_names = ["attention_adapters", "output_adapters"]
        decoder_noisy_adapter_names = ["attention_adapters"]
        for i, layer in enumerate(self.model.model.encoder.layers):
            for n in encoder_noisy_adapter_names:
                layer = getattr(layer, n, None)
                setattr(self.model.model.encoder.layers[i], n, PerturbLayer(Namespace(), layer))


        for i, layer in enumerate(self.model.model.decoder.layers):
            for n in decoder_noisy_adapter_names:
                layer = getattr(layer, n, None)
                setattr(self.model.model.decoder.layers[i], n, PerturbLayer(Namespace(), layer))

    def extra_settings(self, ):
        if isinstance(self.model, BartForConditionalGeneration):
            extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
            for p in extra_model_params:
                if getattr(self.args, p, None):
                    assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                    setattr(self.config, p, getattr(self.args, p))

        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        else:
            self.decoder_start_token_id = self.model.config.decoder_start_token_id

    def forward(self, **batch):
        pad_token_id = self.model.config.pad_token_id
        decoder_start_token_id = self.model.config.decoder_start_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]

        if 'labels' in batch:
            tgt_ids = batch["labels"]
            if isinstance(self.model, T5ForConditionalGeneration):
                decoder_input_ids = self.model._shift_right(tgt_ids)
            else:
                decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, \
                                        decoder_start_token_id=decoder_start_token_id)
        else:
            tgt_ids=None
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
            labels=tgt_ids, 
        )
        # use_prefix=True)

        inputs_embeds = self.model.get_input_embeddings()(src_ids)
        outputs.inputs_embeds = inputs_embeds

        return outputs

    
    def generate(self, batch: dict, return_ids=True, tokenizer=None):
        #print('for decoding, eval_max_length={}, eval_min_length={}, eval_beams={}'\
        #    .format(self.args.eval_max_gen_length, self.args.eval_min_gen_length, self.args.eval_beams))
    
        seq2seq_model_type = (
            BartForConditionalGeneration, 
            GPT2LMHeadModel, 
        )
        if isinstance(self.model, GPT2LMHeadModel):
            # todo: set gpt parameters
            output_sequences = self.model.generate(
                input_ids=batch["input_ids"],
                emb_match=None,
                #control_code=control_code,
                max_length=self.config.eval_max_gen_length,
                temperature=1.0,
                top_p=0.8,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=4,
                #repetition_penalty=args.repetition_penalty,
                #do_sample=True,
                num_return_sequences=4,
            )
        elif isinstance(self.model, BartForConditionalGeneration): 
            self.model.input_ids = batch["input_ids"]
            output_sequences = self.model.generate(
                batch["input_ids"],
                # past_key_values=None,
                attention_mask=batch["attention_mask"],
                use_cache=True,
                length_penalty=self.config.length_penalty,
                # use_prefix=True,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                #num_return_sequences=4,
                num_beams=4, #self.config.num_beams,
                #temperature=1.2, 
                #return_dict_in_generate=True,
                #output_scores=True,
                min_length=self.config.eval_min_gen_length,
                max_length=self.config.eval_max_gen_length,
            )
            if hasattr(self.model, 'batch_word_range'):
                self.batch_word_range = self.model.batch_word_range
        else: 
            assert isinstance(self.model, seq2seq_model_type)

        if return_ids:
            return output_sequences
        else:
            return self.ids_to_clean_text(output_sequences, tokenizer)

    def ids_to_clean_text(self, generated_ids: torch.Tensor, tokenizer):
        gen_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)



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

