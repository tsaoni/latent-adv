from lib import *
import linecache
import numpy as np
from argparse import Namespace
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler

def get_sampler(mode, dataset):
    sampler_dict = {
        "random": RandomSampler, 
    }
    return sampler_dict[mode](dataset)

def get_dataset(
    data_dir, 
    type_path="train", 
    ds_class="glue", 
    glue_type="mrpc", 
    **dataset_kwargs
):
    ds_class_dict = {
        "glue": lambda type: load_dataset("glue", type), 
        #"mlm": mlm_dataset, 
        #"seq2seq": Seq2SeqDataset, 
    }
    dataset_class = ds_class_dict[ds_class]
    if ds_class == "mlm":
        file_path = os.path.join(data_dir, f"{type_path}.txt")
        dataset = dataset_class(file_path, **dataset_kwargs)
    if ds_class == "seq2seq":
        dataset_kwargs = {k: dataset_kwargs[k] for k in ["tokenizer_name"]}
        dataset_kwargs.update({"type_path": type_path})
        dataset = dataset_class(data_dir, **dataset_kwargs)
    else: # glue
        dataset = ds_class_dict["glue"](glue_type)
    return dataset

def get_dataloader(
    dataset, 
    batch_size=10, 
    sampler_type="random", 
    dataloader_drop_last=False, 
    num_workers=1, 
    seed=112, 
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=get_sampler(sampler_type, dataset),
        collate_fn=dataset.collate_fn,
        drop_last=dataloader_drop_last,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed)
    )

class CustomDataset(Dataset):
    def __init__(
        self, 
        model_name_or_path, 
        task=None, 
        train_file=None, 
        validation_file=None, 
        task_name="mrpc", 
        use_slow_tokenizer=False, 
        pad_to_max_length=True, 
        max_length=128, 
        use_fp16=False, 
    ):
        self.config = Namespace(
            task=task, 
            task_name=task_name, 
            tokenizer_name=model_name_or_path, 
            num_labels=1, 
            pad_to_max_length=pad_to_max_length, 
            max_length=max_length, 
            use_fp16=use_fp16, 
        )
        if task is not None:
            args = [task, task_name] if task_name is not None else [task]
            self.raw_datasets = load_dataset(*args)
        else: 
            data_files = {}
            if train_file is not None:
                data_files["train"] = train_file
            if validation_file is not None:
                data_files["validation"] = validation_file
            extension = (train_file if train_file is not None else validation_file).split(".")[-1]
            self.raw_datasets = load_dataset(extension, data_files=data_files)

        # Labels
        if task_name is not None:
            is_regression = task_name == "stsb"
            if not is_regression:
                label_list = self.raw_datasets["train"].features["label"].names
                self.config.num_labels = len(label_list)
            else:
                self.config.num_labels = 1
        else:
            is_regression = self.raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            if is_regression:
                self.config.num_labels = 1
            else:
                label_list = self.raw_datasets["train"].unique("label")
                label_list.sort() 
                self.config.num_labels = len(label_list)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=not use_slow_tokenizer)

        # preprocess
        if task_name is not None:
            task_name = task if task_name is None else task_name
            sentence1_key, sentence2_key = task_to_keys[task_name]
        else:
            if task == "glue":
                non_label_column_names = [name for name in self.raw_datasets["train"].column_names if name != "label"]
                if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                    sentence1_key, sentence2_key = "sentence1", "sentence2"
                else:
                    if len(non_label_column_names) >= 2:
                        sentence1_key, sentence2_key = non_label_column_names[:2]
                    else:
                        sentence1_key, sentence2_key = non_label_column_names[0], None
            elif task == "multi_nli":
                sentence1_key, sentence2_key = "premise", "hypothesis"

        padding = "max_length" if pad_to_max_length else False

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
            relabel = {"0": 2, "1": 1, "2": 0}
            if "label" in examples:
                result["labels"] = list(map(lambda x: relabel[str(x)], examples["label"]))
            #if "label" in examples:
            #    result["labels"] = relabel[str(examples["label"])]
            return result

        processed_datasets = self.raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=[name for name in self.raw_datasets["train"].column_names if name not in [sentence1_key, sentence2_key]],
            desc="Running tokenizer on dataset",
        )
        valid_key = "validation_matched" if task_name == "mnli" or task == "multi_nli" else "validation"
        self.processed_datasets = {
            "train": processed_datasets["train"], 
            "val": processed_datasets[valid_key]
        }
        self.text_column_names = [sentence1_key, sentence2_key]

        #train_dataset = processed_datasets["train"]
        #eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

        #with open(file_path, encoding="utf-8") as f:
        #    self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
    def __len__(self):
        return len(self.processed_datasets[self.type_path])

    def __getitem__(self, i):
        #line = linecache.getline(self.config.file_path, i + 1)
        return self.processed_datasets[self.type_path][i]

    def collate_fn(self, x):
        text_dict = {k: [data[k] for data in x] for k in self.text_column_names}
        x = [{k: v for k, v in data.items() if k not in self.text_column_names} for data in x]
        if self.config.pad_to_max_length:
            batch = default_data_collator(x)
        else:
            batch = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.config.use_fp16 else None))(x)
        batch.update(text_dict)
        return batch

if __name__ == "__main__":
    pass
    #from transformers import (
    #    GlueDataTrainingArguments, 
    #    PreTrainedTokenizerBase, 
    #)
    #args = GlueDataTrainingArguments(task_name="mrpc", data_dir="../data/glue", max_seq_length=128)
    #tokenizer = PreTrainedTokenizerBase()
    #ds = GlueDataset(args, tokenizer, limit_length=10, mode="train")