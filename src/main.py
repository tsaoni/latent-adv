from lib import *
from args import *
#from model import *
from dataset import *
from trainer import *

args = parse_args()
accelerator = Accelerator(log_with=args.report_to, project_dir=args.output_dir) \
    if args.with_tracking else Accelerator()
if args.seed is not None:
    set_seed(args.seed)
if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)
accelerator.wait_for_everyone()

with accelerator.main_process_first():
    dataset = CustomDataset(
        args.model_name_or_path, 
        task=args.task, 
        train_file=args.train_file, 
        validation_file=args.validation_file, 
        task_name=args.task_name, 
        pad_to_max_length=args.pad_to_max_length, 
        max_length=args.max_length, 
        use_fp16=accelerator.use_fp16
    )

#num_labels = dataset.config.num_labels
#config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
attacker = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-base", 
    #args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    ignore_mismatched_sizes=args.ignore_mismatched_sizes, 
)

eval_model = AutoModelWithLMHead.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    ignore_mismatched_sizes=args.ignore_mismatched_sizes, 
)

target_model = AutoModelForSequenceClassification.from_pretrained(
    args.target_model_name_or_path,
    from_tf=bool(".ckpt" in args.target_model_name_or_path),
    #config=config,
    ignore_mismatched_sizes=args.ignore_mismatched_sizes,
)

train_dataset, eval_dataset = (dataset.processed_datasets[k] for k in ["train", "val"])
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=dataset.collate_fn, batch_size=args.per_device_train_batch_size, num_workers=1)
eval_dataloader = DataLoader(eval_dataset, collate_fn=dataset.collate_fn, batch_size=args.per_device_eval_batch_size, num_workers=1)

output_dir = f"../model/{args.task}"
trainer = Trainer(
    attacker, eval_model, target_model, 
    train_dataloader, eval_dataloader, 
    weight_decay=1e-4, 
    learning_rate=1e-3, 
    gradient_accumulation_steps=1, 
    lr_scheduler_type="linear", 
    gpus=1, 
    train_batch_size=args.per_device_train_batch_size, 
    val_batch_size=args.per_device_eval_batch_size, 
    num_train_epochs=1, 
    seed=112, 
    tokenizer_name=args.model_name_or_path, 
    max_source_length=128, 
    num_warmup_steps=10, 
    output_dir=output_dir, 
    use_wandb=True, 
)
trainer.train()

import pdb 
pdb.set_trace()


