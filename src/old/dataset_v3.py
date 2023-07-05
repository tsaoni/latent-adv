from lib import *
from old.dataset_v2 import Seq2SeqDataset

def get_sampler(mode, dataset):
    sampler_dict = {
        "random": RandomSampler, 
    }
    return sampler_dict[mode](dataset)

def get_dataloader(
    data_dir, 
    type_path="train", 
    batch_size=10, 
    sampler_type="random", 
    dataloader_drop_last=False, 
    num_workers=1, 
    seed=112, 
    ds_class="seq2seq", 
    **dataset_kwargs
):
    ds_class_dict = {
        "mlm": mlm_dataset, 
        "seq2seq": Seq2SeqDataset, 
    }
    dataset_class = ds_class_dict[ds_class]
    if ds_class == "mlm":
        file_path = os.path.join(data_dir, f"{type_path}.txt")
        dataset = dataset_class(file_path, **dataset_kwargs)
    else: # seq2seq
        dataset_kwargs = {k: dataset_kwargs[k] for k in ["tokenizer_name"]}
        dataset_kwargs.update({"type_path": type_path})
        dataset = dataset_class(data_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=get_sampler(sampler_type, dataset),
        collate_fn=dataset.collate_fn,
        drop_last=dataloader_drop_last,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed)
    )

class mlm_dataset(Dataset):
    def __init__(
        self, 
        file_path, 
        tokenizer_name=None, 
        mlm=False, 
        mlm_probability=0.15, 
        block_size=128, 
    ):
        self.config = Namespace(
            file_path=file_path, 
            tokenizer_name=tokenizer_name, 
            mlm=mlm, 
            mlm_probability=mlm_probability, 
            block_size=block_size, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._dataset = CustomLineByLineTextDataset(self.tokenizer, file_path, block_size)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        return self._dataset[i]

    @property
    def collate_fn(self):
        return CustomDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=self.config.mlm, mlm_probability=self.config.mlm_probability
        )

class CustomLineByLineTextDataset(LineByLineTextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        super().__init__(tokenizer, file_path, block_size)
        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        self.examples[i].update({"src_text": self.lines[i]})
        return self.examples[i]

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        extra_keys = ["src_text"]
        collate_input = [{k: v for k, v in ex.items() if k not in extra_keys} for ex in examples]
        batch = super().torch_call(collate_input)
        extra_data = {k: [ex[k] for ex in examples] for k in extra_keys}
        batch.update(extra_data)
        return batch