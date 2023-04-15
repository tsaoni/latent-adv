from lib import *

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

''' log metrics and generated result '''
class LoggingCallback(pl.Callback):
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        args, 
        output_dir, 
    ):
        super().__init__()
        self.args = args
        self.logging_dir = os.path.join(output_dir, 'log')
        os.makedirs(self.logging_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, *unused_args):
        metric = dict()
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        metric.update(lrs)
        if len(pl_module.metric.batch_metric_dicts['train']) > 0: 
            metric.update(pl_module.metric.batch_metric_dicts['train'][-1])
        trainer.logger.log_metrics(metric, step=trainer.global_step) # wandb

    @rank_zero_only
    def write_pred_logs(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        type_path: str, 
        save_generations=True, 
    ) -> None:
        self.logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        # Log results
        od = os.path.join(self.logging_dir, f"{trainer.global_step:05d}")
        generations_file = os.path.join(od, f"{type_path}_generations.json")
        os.makedirs(od, exist_ok=True)
    
        if save_generations:
            pred_ref_list = []
            if len(pl_module.metric.result_dicts) == 0: return
            predictions = pl_module.metric.result_dicts[-1]['predictions']
            references = pl_module.metric.result_dicts[-1]['references']
            for p, r in zip(predictions, references):
                pred_ref_list.append({'pred': p, 'ref': r})
            with open(generations_file, 'w') as f:
                json.dump(pred_ref_list, f, indent=4)

        pl_module.metric.result_dicts = [] # clean up preds for save memory (?)
    
    @rank_zero_only
    def write_metric_logs(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        type_path: str, 
    ) -> None:
        od = os.path.join(self.logging_dir, f"{trainer.global_step:05d}")
        results_file = os.path.join(od, f"{type_path}_results.json")
        os.makedirs(od, exist_ok=True)
        if len(pl_module.metric.metric_dicts[type_path]) > 0:
            metric = pl_module.metric.metric_dicts[type_path][-1] 
            with open(results_file, 'w') as f:
                json.dump(metric, f, indent=4)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    def update_metric(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prefix='train'):
        pl_module.metric.calc_metric_per_period(prefix=prefix)
        self.write_metric_logs(trainer, pl_module, prefix)
        if prefix in ['val', 'test']:
            self.write_pred_logs(trainer, pl_module, prefix)
        if len(pl_module.metric.metric_dicts[prefix]) > 0:
            trainer.logger.log_metrics(pl_module.metric.metric_dicts[prefix][-1], step=trainer.current_epoch)

    @rank_zero_only
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.update_metric(trainer, pl_module, prefix='train')

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.update_metric(trainer, pl_module, prefix='test')

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        self.update_metric(trainer, pl_module, prefix='val')

    @staticmethod
    def get_logger(logger_name, output_dir):
        if logger_name == "default":
            logger = None  # don't pollute wandb logs unnecessarily (?)
            # logger = logging.getLogger(__name__)
        elif logger_name == "wandb":
            project = os.environ.get("WANDB_PROJECT")
            logger = WandbLogger(name=output_dir, project=project)

        elif logger_name == "wandb_shared":
            logger = WandbLogger(name=output_dir, project=f"hf_")

        return logger

    @staticmethod
    def add_specific_args():
        @dataclass
        class LoggingArguments:
            logger_name: Optional[str] = field(
                default=None, 
                metadata={"help": "choices: wandb, default, "}
            )

        return LoggingArguments

class CheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, args, output_dir):
        self.args = args
        checkpoint_path = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)
        
        super().__init__(
            dirpath=checkpoint_path,
            filename='model-{epoch:02d}-{' + args.val_metric + ':.2f}',
            save_top_k=args.save_top_k,
            monitor=args.val_metric,
            mode='min' if 'loss' in args.val_metric else 'max', 
            save_last=True, 
        )

    @staticmethod
    def add_specific_args():
        @dataclass
        class CheckpointArguments:
            val_metric: Optional[str] = field(
                default=None, 
                metadata={"help": "choices: bleu, rouge2, loss, accuracy, "}
            )
            save_top_k: Optional[int] = field(
                default=1, 
                metadata={"help": "how many checkpoints to save. "}
            )

        return CheckpointArguments



class OthersSeq2SeqLoggingCallback(pl.Callback):

    logger = logging.getLogger(__name__)

    def on_train_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        self.logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        # Uncommenting this will save val generations
        # return self._write_logs(trainer, pl_module, "valid")

def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        # period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )