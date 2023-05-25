from lib import *
from utils import get_scheduler_info

@dataclass
class ScriptArguments:
    do_train_seq2seq: bool = field(
        default=False, 
        metadata={"help": "do training if set to true. "}
    )
    do_eval_seq2seq: bool = field(
        default=False, 
        metadata={"help": "do evaluation if set to true. "}
    )
    do_train_cls: bool = field(
        default=False, 
        metadata={"help": "do training if set to true. "}
    )
    do_eval_cls: bool = field(
        default=False, 
        metadata={"help": "do evaluation if set to true. "}
    )
    fp16: bool = field(
        default=False, 
        metadata={"help": ""}
    )
    gpus: Optional[int] = field(
        default=1, 
        metadata={"help": ""}
    )
    n_tpu_cores: Optional[int] = field(
        default=1, 
        metadata={"help": ""}
    )
    fp16_opt_level: Optional[str] = field(
        default="O2", 
        metadata={"help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "}
    )
    num_train_epochs: Optional[int] = field(
        default=5, 
        metadata={"help": ""}
    )
    max_train_steps: Optional[int] = field(
        default=400, 
        metadata={"help": ""}
    )
    max_eval_steps: Optional[int] = field(
        default=400, 
        metadata={"help": ""}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, 
        metadata={"help": ""}
    )
    gradient_clip_val: Optional[float] = field(
        default=1.0, 
        metadata={"help": ""}
    )
    early_stopping_patience: Optional[int] = field(
        default=-1, 
        metadata={"help": "-1 means never early stop. "
        "early_stopping_patience is measured in validation checks, not epochs. "
        "So val_check_interval will effect it."}
    )
    seq_local_ckpt: Optional[str] = field(
        default=None, 
        metadata={"help": "use seq2seq model from checkpoints. "}
    )
    cls_local_ckpt: Optional[str] = field(
        default=None, 
        metadata={"help": "use classification model from checkpoints. "}
    )
    tgwv_local_ckpt: Optional[str] = field(
        default=None, 
        metadata={"help": "use model from checkpoints. "}
    )
    tgwv_v_mode: Optional[str] = field(
        default=None, 
        metadata={"help": "options: `softmax`, `raw`, `drop`. "}
    )
    attr_dim: Optional[int] = field(
        default=0, 
        metadata={"help": "the dim used for attribute. "}
    )
    ptb_param: Optional[str] = field(
        default='embed', 
        metadata={"help": "options: `embed`, `enc_out`. "}
    )
    loss_ratio: Optional[float] = field(
        default=0.6, 
        metadata={"help": "the ratio of the reconstruction loss. "}
    )
    latent_ckpt: Optional[str] = field(
        default=None, 
        metadata={"help": "checkpoint of the latent classifier. "}
    )
    pool: Optional[str] = field(
        default="mean", 
        metadata={"help": "the pooling function used in latent classifier. "}
    )
    check_val_every_n_epoch: Optional[int] = field(
        default=1, 
        metadata={"help": "do eval for each n epoch. "}
    )
    limit_val_batches: Optional[float] = field(
        default=1.0, 
        metadata={"help": "do eval only for the partial of the batch. "}
    )

@dataclass
class LoggingArguments:
    logger_name: Optional[str] = field(
        default=None, 
        metadata={"help": "choices: wandb, default, "}
    )

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

@dataclass
class SeqDataArguments:
    seq2seq_max_source_length: Optional[int] = field(
        default=1024, 
        metadata={"help": ""}
    )
    seq2seq_sortish_sampler: bool = field(
        default=False, 
        metadata={"help": ""}
    )
    seq2seq_n_train: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    seq2seq_n_val: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    seq2seq_n_test: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    seq2seq_data_dir: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    max_target_length: Optional[int] = field(
        default=1024, 
        metadata={"help": ""}
    )
    val_max_target_length: Optional[int] = field(
        default=1024, 
        metadata={"help": ""}
    )
    test_max_target_length: Optional[int] = field(
        default=1024, 
        metadata={"help": ""}
    )
    src_lang: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    tgt_lang: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )

@dataclass
class ClsDataArguments:
    cls_max_source_length: Optional[int] = field(
        default=1024, 
        metadata={"help": ""}
    )
    cls_sortish_sampler: bool = field(
        default=False, 
        metadata={"help": ""}
    )
    cls_n_train: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    cls_n_val: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    cls_n_test: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    cls_data_dir: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )

@dataclass
class DataModuleArguments:
    train_batch_size: Optional[int] = field(
        default=10, 
        metadata={"help": ""}
    )
    eval_batch_size: Optional[int] = field(
        default=10, 
        metadata={"help": ""}
    )
    max_tokens_per_batch: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
    num_workers: Optional[int] = field(
        default=4, 
        metadata={"help": ""}
    )

@dataclass
class SeqModelArguments:
    seq2seq_model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    seq2seq_config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    seq2seq_tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    seq2seq_cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    seq2seq_use_big: bool = field(
        default=False, 
        metadata={"help": "whether to use large tokenizer. "}
    )
    seq2seq_num_labels: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
    seq2seq_model_mode: Optional[str] = field(
        default="summarization", 
        metadata={"help": ""}
    )
    # generation
    eval_beams: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
    eval_min_gen_length: Optional[int] = field(
        default=6, 
        metadata={"help": ""}
    )
    eval_max_gen_length: Optional[int] = field(
        default=10, 
        metadata={"help": ""}
    )
    length_penalty: Optional[float] = field(
        default=1.0, 
        metadata={"help": "never generate more than n tokens. "}
    )
    use_vae: Optional[str] = field(
        default="latent", 
        metadata={"help": "specify which vae architecture to use, "
                    "if wish not to train with vae, don't set the value. "}
    )
    z_dim: Optional[int] = field(
        default=128, 
        metadata={"help": "the latent dimension. "}
    )
    hidden_dim: Optional[int] = field(
        default=256, 
        metadata={"help": "for lstm config. "}
    )
    encoder_num_layers: Optional[int] = field(
        default=1, 
        metadata={"help": "for lstm config. "}
    )
    decoder_num_layers: Optional[int] = field(
        default=1, 
        metadata={"help": "for lstm config. "}
    )
    decoder_input_type: Optional[str] = field(
        default="origin", 
        metadata={"help": "for lstm config. "}
    )

@dataclass
class ClsModelArguments:
    cls_model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    cls_config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cls_tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cls_cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    cls_use_big: bool = field(
        default=False, 
        metadata={"help": "whether to use large tokenizer. "}
    )
    cls_num_labels: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
    cls_model_mode: Optional[str] = field(
        default="sequence-classification", 
        metadata={"help": ""}
    )

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
    vae_loss_ratio: Optional[float] = field(
        default=1.0, 
        metadata={"help": "loss ratio of VAE. "}
    )
