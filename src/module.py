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
)

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace = None, 
        model = None,
        callback_args_dict = None, # keys: ckpt, log
        output_dir_kwargs = None, 
        **kwargs, # attr: gpus, sortish_sampler, max_tokens_per_batch, train_batch_size, gradient_accumulation_steps, dataset_len
        # num_train_epochs, 
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
            self.output_dir = os.path.join('../models', self.hparams.output_dir)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.hparams.output_dir = self.output_dir

        print(f'the current output_dir is {self.output_dir}')
        if len(os.listdir(self.output_dir)) > 3 and self.hparams.do_train:
            print('Output directory ({}) already exists and is not empty, overwrite to it...'.format(self.output_dir))


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
        seq2seq = ["summarization", "translation"]
        return self.model.args.model_mode in seq2seq

    def get_lr_scheduler(self):
        arg_to_scheduler = get_scheduler_info()['arg_to_scheduler']
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
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
        logging_callback = LoggingCallback(self.callback_args_dict['log'], self.output_dir)
        checkpoint_callback = CheckpointCallback(self.callback_args_dict['ckpt'], self.output_dir)
        return [logging_callback, checkpoint_callback]
    
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
            
            if self.hparams.label_smoothing == 0:
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

                assert lm_logits.shape[-1] == self.model.config.vocab_size
                # print(lm_logits.shape, tgt_ids.shape, lm_logits.shape[-1] )
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            else:
                lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(
                    lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
                )

        elif self.model.args.model_mode == 'sequence-classification':
            # todo: use default loss, which can be changed to customized one
            if self.hparams.label_smoothing == 0:
                loss = outputs['loss']
            else:
                # todo: implement label smoothing
                loss = outputs['loss']
            
        return (loss,)

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
        self.metric.add_batch_metric(dict(loss=loss_tensors[0].item(), **logs))
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

    def generate_step(self, batch: dict, prefix='val') -> dict:
        # todo: add batch eval metric
        bsz = batch["input_ids"].size(0)
        if self.model.model_type in ['seq2seq', 'tgwv']:
            t0 = time.time()
            generated_ids = self.model.generate(batch)
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
    
    @staticmethod
    def add_specific_args():
        scheduler_dict = get_scheduler_info()
        @dataclass
        class TrainingArguments:
            adafactor: bool = field(
                default=False, 
                metadata={"help": ""}
            )
            adam_epsilon: Optional[float] = field(
                default=1e-8, 
                metadata={"help": "Epsilon for Adam optimizer. "}
            )
            seed: Optional[int] = field(
                default=112, 
                metadata={"help": "the magic seed. "}
            )
            warmup_steps: Optional[int] = field(
                default=100, 
                metadata={"help": ""}
            )
            learning_rate: Optional[float] = field(
                default=5e-05, 
                metadata={"help": "the learning rate"}
            )
            weight_decay: Optional[float] = field(
                default=0.0, 
                metadata={"help": ""}
            )
            dropout: Optional[float] = field(
                default=0.0, 
                metadata={"help": "dropout rate. "}
            )
            label_smoothing: Optional[float] = field(
                default=0.0, 
                metadata={"help": "label smoothing rate. "}
            )
            max_grad_norm: Optional[float] = field(
                default=1.0, 
                metadata={"help": ""}
            )
            lr_scheduler: Optional[str] = field(
                default='linear', 
                metadata={
                    "help": "Learning rate scheduler", 
                    "choices": scheduler_dict['arg_to_scheduler_choices'], 
                    "metavar": scheduler_dict['arg_to_scheduler_metavar'],
                }
            )
            output_dir: Optional[str] = field(
                default=None, 
                metadata={"help": ""}
            )

          
        return TrainingArguments
    
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