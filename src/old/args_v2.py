from lib import *
from utils import get_scheduler_info


@dataclass
class DataTrainingArguments:
    n_train: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    n_val: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    n_test: Optional[int] = field(
        default=-1, 
        metadata={"help": ""}
    )
    data_dir: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    max_source_length: Optional[int] = field(
        default=128, 
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
    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": ""}
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": ""}
    )
    sampler_type: Optional[str] = field(
        default="random", 
        metadata={"help": ""}
    )



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    use_big: bool = field(
        default=False, 
        metadata={"help": "whether to use large tokenizer. "}
    )
    num_labels: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
    model_mode: Optional[str] = field(
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

scheduler_dict = get_scheduler_info()
@dataclass
class TrainingArguments:
    do_train: bool = field(
        default=False, 
        metadata={"help": "do training if set to true. "}
    )
    do_eval: bool = field(
        default=False, 
        metadata={"help": "do evaluation if set to true. "}
    )
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
    fp16: bool = field(
        default=False, 
        metadata={"help": ""}
    )
    gpus: Optional[int] = field(
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

