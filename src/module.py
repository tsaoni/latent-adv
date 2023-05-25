from lib import *


from model import *
from dataset import *
from callback import *
from metric import *
from utils import (
    use_task_specific_params, 
    pickle_save, 
    lmap,
    label_smoothed_nll_loss,
    flatten_list,
    check_parameter_value,
    get_scheduler_info,
    wir_logit_to_color_range, 
)

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace = None, 
        model = None,
        callback_args_dict = None, # keys: ckpt, log
        output_dir_kwargs = None, 
        **kwargs, # attr: gpus, sortish_sampler, max_tokens_per_batch, train_batch_size, gradient_accumulation_steps, dataset_len
        # num_train_epochs, attr_dim, ptb_param
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.set_extra_attrs(kwargs)

        self.create_output_dir(output_dir_kwargs)
        self.hparams_save_path = os.path.join(self.output_dir, "hparams.pkl")
        pickle_save(self.hparams, self.hparams_save_path)
        self.check_sampler_usage()
        self.step_count = 0
        self.set_attr('model', model)
        self.set_attr('callback_args_dict', callback_args_dict)
        self.metric = Metric(
            model_mode=self.model.model_type,
            val_metric_name=self.callback_args_dict['ckpt'].val_metric, 
            metric_save_path=os.path.join(self.output_dir, "metrics_per_epoch.json"), 
        )
        self.metadata = dict() # use for logs
    
    def set_attr(self, attr, attr_value, obj=None):
        if obj is None: obj = self
        if hasattr(self.hparams, attr):
            setattr(obj, attr, getattr(self.hparams, attr))
        else:
            setattr(obj, attr, attr_value)

    def create_output_dir(self, kwargs):
        self.set_attr('output_dir_kwargs', kwargs)
        if self.hparams.output_dir is None:
            assert self.output_dir_kwargs is not None
            output_dir_name_list = []
            for key, value in self.output_dir_kwargs.items():
                output_dir_name_list.append(f"{key}={value}")

            output_dir_name = "_".join(output_dir_name_list)
            """
            'tb={}_'.format(self.hparams.train_batch_size) + \
                        'e={}_'.format(self.hparams.num_train_epochs) + 'd={}_'.format(self.hparams.dropout) + \
                        'l={}_'.format(self.hparams.label_smoothing) + 'lr={}_'.format(self.hparams.learning_rate) \
                        + 'w={}_'.format(self.hparams.weight_decay) + 's={}'.format(self.hparams.seed)
            """
            self.output_dir = os.path.join('../models', output_dir_name)
        else:
            model_dir = '../models/'
            prefix = '' if model_dir in self.hparams.output_dir else model_dir
            self.output_dir = prefix + self.hparams.output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.hparams.output_dir = self.output_dir

        print(f'the current output_dir is {self.output_dir}')
        if len(os.listdir(self.output_dir)) > 3:
            print('Output directory ({}) already exists and is not empty, may overwrite to it...'.format(self.output_dir))
        global output_dir
        output_dir = self.output_dir

    def check_sampler_usage(self):
        if self.hparams.sortish_sampler and self.hparams.gpus > 1:
            pass
            # self.hparams.replace_sampler_ddp = False
        elif self.hparams.max_tokens_per_batch is not None:
            if self.hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if self.hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

    def set_extra_attrs(self, kwargs):
        for key, value in kwargs.items():
            self.set_attr(key, value, obj=self.hparams)
     
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps * num_devices
        dataset_size = self.hparams.dataset_len
        return int(dataset_size / effective_batch_size) * self.hparams.num_train_epochs

    @property
    def is_seq2seq(self) -> bool:
        seq2seq = ["summarization", "translation", "ptb"]
        return self.model.args.model_mode in seq2seq
    
    @property
    def is_classification(self) -> bool:
        classification = ["sequence-classification", "latent"]
        return self.model.args.model_mode in classification

    def get_lr_scheduler(self):
        arg_to_scheduler = get_scheduler_info()['arg_to_scheduler']
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def set_requires_grad(self):
        if hasattr(self.model.args, "use_vae") and self.model.args.use_vae is not None:
            for n, p in self.model.named_parameters():
                p.requires_grad = True if "vae" in n else False

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        if self.model.args.model_mode == "latent": model = self.model.model.classification_head
        elif hasattr(self.model.args, "use_vae") and self.model.args.use_vae is not None:
            model = self.model.model.model.encoder.vae
        else: model = self.model
        self.set_requires_grad()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]
    
    def configure_callbacks(self):
        checkpoint_callback_list = []
        logging_callback = LoggingCallback(self.callback_args_dict['log'], self.output_dir)
        checkpoint_callback_list.append(CheckpointCallback(self.callback_args_dict['ckpt'], self.output_dir))
        if self.model.args.model_mode == "latent":
            checkpoint_callback_list.append(LatentCallback(self.output_dir))
        return [logging_callback, *checkpoint_callback_list, ]
    
    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def forward(self, batch: dict) -> Tuple:
        outputs = self.model(**batch)

        # todo: change to current state which specify what loss to use
        if self.is_seq2seq:
            lm_logits = outputs['logits']
            tgt_ids = batch['labels']
            pad_token_id = self.model.tokenizer.pad_token_id
            
            # reconstruct
            if self.hparams.label_smoothing == 0:
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
                assert lm_logits.shape[-1] == self.model.config.vocab_size
                # print(lm_logits.shape, tgt_ids.shape, lm_logits.shape[-1] )
                rec_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            else:
                lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
                rec_loss, nll_loss = label_smoothed_nll_loss(
                    lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
                )

            training_loss = self.hparams.vae_loss_ratio * VAE_encoder.loss[0]\
                                 + (1. - self.hparams.vae_loss_ratio) * rec_loss

            return (training_loss, *VAE_encoder.loss, rec_loss, )
            # attribute
            cls_logits = batch['cls_logits']
            log_cls_logits = wir_logit_to_color_range(cls_logits)
            # bart case
            def norm(t, f='z-prob'):
                if f == 'z-prob':
                    return (t - torch.mean(t)) / torch.std(t)
                elif f == 'l2':
                    return torch.nn.functional.normalize(torch.unsqueeze(t, dim=0)).squeeze(0)
            ptb_param = outputs.inputs_embeds if self.hparams.ptb_param == 'embed' else outputs.encoder_last_hidden_state
            ce_loss_fct = torch.nn.CrossEntropyLoss()
            mse_loss_fct = torch.nn.MSELoss()
            norm_ptb_param, norm_cls_logits = norm(ptb_param[:, :, self.hparams.attr_dim]), norm(log_cls_logits[:, :, 1])
            attr_loss = mse_loss_fct(norm_ptb_param, norm_cls_logits)
            
            # do weighted sum on two loss
            loss = self.hparams.loss_ratio * rec_loss + (1 - self.hparams.loss_ratio) * attr_loss

            return (loss, rec_loss, attr_loss, )

        elif self.is_classification:
            # todo: use default loss, which can be changed to customized one
            if self.hparams.label_smoothing == 0:
                loss = outputs['loss']
            else:
                # todo: implement label smoothing
                loss = outputs['loss']
        
            return (loss, )

    def get_metadata(self, batch, reset=False):
        #logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        pad = self.model.tokenizer.pad_token_id
        self.metadata["token_per_batch"] = batch["input_ids"].ne(pad).sum().item()
        if self.is_seq2seq:
            self.metadata["token_per_batch"] += batch["labels"].ne(pad).sum().item()
        self.metadata["batch_size"] = batch["input_ids"].shape[0]
        self.metadata["src_pad_tok"] = batch["input_ids"].eq(pad).sum().item()
        self.metadata["src_pad_frac"] = batch["input_ids"].eq(pad).float().mean().item()

        metadata = copy.deepcopy(self.metadata)
        if reset: self.metadata = dict()
        return metadata

    def training_step(self, batch, batch_idx) -> Dict:
        self.step_count += 1
        loss_tensors = self(batch)
        logs = self.get_metadata(batch)
        loss_name = ['loss', 'vae_Loss', 'vae_MSE', 'vae_KLD', 'CELoss'] if self.is_seq2seq else ['loss']
        loss_d = {name: loss.item() for name, loss in zip(loss_name, loss_tensors)}
        self.metric.add_batch_metric(dict(**loss_d, **logs))
        #self.logger.log_metrics({'loss': loss_tensors[0]}, step=self.step_count)

        return {"loss": loss_tensors[0], "log": logs}
 

    def validation_step(self, batch, batch_idx) -> Dict:
        return self.generate_step(batch, prefix='val')

    def test_step(self, batch, batch_idx):
        return self.generate_step(batch, prefix='test')

    """
    def on_train_epoch_end(self):
        self.metric.calc_metric_per_period(prefix='train')

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.metric.calc_metric_per_period(prefix=prefix)

        # metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        # preds = flatten_list([x["preds"] for x in outputs])
        # log after epoch end. 
        # self.log('val_{}'.format(self.val_metric_name), float(eval_metric[self.val_metric_name]))
        # return all_metrics


    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")
    """

    def generate_step(self, batch: dict, prefix='val', visual_layer=False) -> dict:
        # todo: add batch eval metric
        bsz = batch["input_ids"].size(0)
        if self.model.model_type in ['seq2seq', 'tgwv']:
            t0 = time.time()
            generated_ids = self.model.generate(batch, visual_layer=visual_layer)
            preds: List[str] = self.model.ids_to_clean_text(generated_ids)
            target: List[str] = self.model.ids_to_clean_text(batch["labels"])
            gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
            loss_tensors = self(batch)
            loss = loss_tensors[0].item()
            # print('INPUT:', self.ids_to_clean_text(batch["input_ids"]))
            # print(preds, target)
            #rouge: Dict = self.calc_generative_metrics(preds, target)
            summ_len = np.mean(lmap(len, generated_ids))
            base_metrics = dict(loss=loss, gen_time=gen_time, gen_len=summ_len)# , preds=preds, target=target)#, **rouge)
            seq_kwargs = dict(
                predictions=preds, 
                references=target, 
            )
            metric_kwargs = {'seq2seq_kwargs': seq_kwargs}
        
        elif self.model.model_type == 'cls':
            outputs = self.model(**batch)
            preds: torch.Tensor(List[int]) = outputs.logits.argmax(dim=-1)
            target: torch.Tensor(List[int]) = batch["labels"]
            loss_tensors = self(batch)
            loss = loss_tensors[0].item()
            #base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
            base_metrics = dict()
            batch_acc = sum([1 if p == t else 0 for p, t in zip(preds, target)]) / bsz
            base_metrics.update(acc=batch_acc, loss=loss) #, preds=preds.tolist(), target=target.tolist())
            seq_kwargs = dict(
                predictions=preds.tolist(), 
                references=target.tolist(), 
            )
            metric_kwargs = {'result_kwargs': seq_kwargs}

        self.metric.add_batch_metric(base_metrics, type_path=prefix, **metric_kwargs)
        # log metric to checkpoint callback. 
        if len(self.metric.metric_dicts['val']) > 0:
            metric_name = self.callback_args_dict['ckpt'].val_metric
            self.log(metric_name, self.metric.metric_dicts['val'][-1][metric_name])

        return base_metrics

    

    """ save and load """

    @pl.utilities.rank_zero_only
    def save_checkpoint(self, checkpoint) -> None:
        print('Saving the the checkpoint.')
        return
    
    @pl.utilities.rank_zero_only
    def save_specific_model_checkpoint(self, output_dir, model, tokenizer=None):
        
        pass

    """
    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any], filepath='checkpoint') -> None:
        # if filepath is not None:
        #     save_path = filepath[:-5]
        # else:
        #     save_path = self.output_dir.joinpath("checkpoint-hello")
        save_path = Path(self.output_dir).joinpath("checkpoint-curr_best")
        print('the suggested save_path is {}, saving to {}'.format(filepath, save_path))

        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print('SAVING TO checkpoint {}'.format(save_path))
    """
    
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