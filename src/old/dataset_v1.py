from lib_v1 import *

from utils_v1 import *

class BaseDataset(Dataset):

    DEFAULT_MAX_SOURCE_LENGTH = 150
    DEFAULT_MAX_TARGET_LENGTH = 150

    def __init__(
        self,
        tokenizer,
        data_dir=None,
        max_source_length=DEFAULT_MAX_SOURCE_LENGTH,
        max_target_length=None,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        **kwargs, # here are unused kwargs...
    ):
        super().__init__()
        # read file
        #check_variable_status(data_dir, name='data_dir', status='None')
        self.data_dir = data_dir
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        # if os.path.exists(self.len_file):
        if False: # haven't use sampler ...
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.set_max_source_length(max_source_length)
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def set_max_source_length(self, max_source_length):
        if os.path.exists(self.len_file):
            f = open(self.len_file)
            self.max_source_length = np.asarray(f.read().split('\n'), dtype=int).max()
            print(f'the max source length is {self.max_source_length}, calculate from len file. ')
        else:
            self.max_source_length = max_source_length
            print(f'the max source length is {self.max_source_length}, use argument value. ')

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class Seq2SeqDataset(BaseDataset):
    def __init__(self, args, tokenizer, type_path='train'):
        self.args = set_specific_attr(args, get_dataset_specific_attr())
        n_observations_per_split = {
            "train": self.args.n_train,
            "val": self.args.n_val,
            "test": self.args.n_test,
        }
        n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        target_lens = {
            "train": self.args.max_target_length,
            "val": self.args.val_max_target_length,
            "test": self.args.test_max_target_length,
        }
        assert target_lens["train"] <= target_lens["val"], f"target_lens: {target_lens}"
        assert target_lens["train"] <= target_lens["test"], f"target_lens: {target_lens}"
        self.args.max_target_length = target_lens[type_path]
        super().__init__(
            tokenizer,
            type_path=type_path,
            n_obs=n_obs[type_path],
            **self.args.__dict__, 
        )

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            # src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            # tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
        ).data
        batch_encoding["ids"] = [x["id"] for x in batch] #torch.tensor([x["id"] for x in batch])
        return batch_encoding

 
class ClassificationDataset(BaseDataset):
    name_with_no_token_type_ids = [
        'textattack/roberta-base-rotten_tomatoes', 
        'textattack/distilbert-base-uncased-rotten-tomatoes', 
        'facebook/bart-large', 
    ]
    def __init__(self, args, tokenizer, type_path='train'):
        self.args = set_specific_attr(args, get_dataset_specific_attr())
        n_observations_per_split = {
            "train": self.args.n_train,
            "val": self.args.n_val,
            "test": self.args.n_test,
        }
        n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        super().__init__(
            tokenizer,
            type_path=type_path,
            n_obs=n_obs[type_path],
            **self.args.__dict__, 
        )

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        source_inputs = [encode_line(self.tokenizer, x["src_texts"], \
                        self.max_source_length, dataset_name=None) for x in batch]
        target_inputs = [label_to_tensor(x['tgt_texts']) for x in batch]

        source_ids = [x["input_ids"].squeeze() for x in source_inputs]
        src_mask = [x["attention_mask"].squeeze() for x in source_inputs]
        if self.tokenizer.name_or_path in self.name_with_no_token_type_ids:
            src_token_type_ids = None
        else:
            src_token_type_ids = [x["token_type_ids"].squeeze() for x in source_inputs]
        target_ids = target_inputs
                
        input_ids = torch.stack(source_ids)
        masks = torch.stack(src_mask)
        token_type_ids = None if src_token_type_ids is None else torch.stack(src_token_type_ids)
        target_ids = torch.stack(target_ids).squeeze().to(torch.long)
        pad_token_id = self.pad_token_id
        trim_batch_tuple = trim_batch(input_ids, token_type_ids, pad_token_id, attention_mask=masks)
        if len(trim_batch_tuple) == 3:
            source_ids, source_mask, source_token_type_ids = trim_batch_tuple
        else:
            source_ids, source_mask = trim_batch_tuple
        token_type_ids_kwargs = {} if token_type_ids is None else {"token_type_ids": source_token_type_ids, }
        batch_encoding = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            **token_type_ids_kwargs, 
        }
        batch_encoding.update({"ids": [x["id"] for x in batch]})

        # batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding

class TGWVDataset(ClassificationDataset):
    def __init__(
            self, 
            args, seq_tokenizer, cls_tokenizer, cls_model, 
            padding_num: int, 
            type_path='train', 
            v_mode='softmax', 
        ):
        super().__init__(args, cls_tokenizer, type_path=type_path)
        self.seq_tokenizer = seq_tokenizer
        self.cls_model = cls_model
        self.padding_num = padding_num
        self.v_mode = v_mode
        global max_source_length
        max_source_length = self.max_source_length

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding: Dict[str, torch.Tensor] = self.seq_tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            # src_lang=self.src_lang,
            tgt_texts=[x["src_texts"] for x in batch],
            # tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
        ).data
        seq_len = batch_encoding['input_ids'].shape[1]
        cls_batch = super().collate_fn(batch)
        mask_id = self.tokenizer.mask_token_id

        assert hasattr(self, 'batch_size') and hasattr(self, 'type_path'), \
            'should have attribute `batch_size` and `type_path` before loading wir. '
        batch_idx = int(batch[0]['id'] / self.batch_size)
        cls_logits = text_generation_with_vector_input_collate_fn(
            seq_len, cls_batch, self.cls_model, self.padding_num, 
            mask_id=mask_id, 
            mode=self.v_mode, 
            wir_pt_path=os.path.join(self.data_dir, "wir", self.type_path), 
            batch_idx=batch_idx, 
            id_list=[x["id"] for x in batch], 
            load=True, 
        )
        cls_labels = cls_batch['labels']
        batch_encoding.update({"cls_logits": cls_logits, "cls_labels": cls_labels})
        batch_encoding.update({"src_texts": [x["src_texts"] for x in batch]})
        batch_encoding.update({"ids": [x["id"] for x in batch]})

        # batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


def get_dataset_specific_attr():
    return ['max_source_length', 'sortish_sampler', 'n_train', 'n_val', 
            'n_test', 'data_dir', ]

def get_specific_key(args, key_name):
    for key in args.__dict__.keys():
        if key_name in key:
            return getattr(args, key)

class DataModule(pl.LightningDataModule):
    DATASET_CLS = {
        'seq2seq': Seq2SeqDataset, 
        'classification': ClassificationDataset, 
        'tgwv': TGWVDataset, 
    }

    def __init__(
        self, 
        args, 
        dataset_args,
        dataset_mode, 
        gpus = 1, 
        mode='fit', 
        tokenizer=None,
        **tgwv_kwargs, 
    ):
        super().__init__() 
        assert dataset_mode in self.DATASET_CLS.keys()
        self.dataset_mode = dataset_mode
        self.dataset_class = self.DATASET_CLS[dataset_mode]
        self.tokenizer = tokenizer
        self.args, self.dataset_args = args, set_specific_attr(dataset_args, get_dataset_specific_attr())
        self.b_size = {
            'train': args.train_batch_size, 
            'val': args.eval_batch_size, 
            'test': args.eval_batch_size, 
        }
        self.gpus = gpus
        global max_source_length
        max_source_length = get_specific_key(dataset_args, "max_source_length")
        if not dataset_mode == 'tgwv':
            self.dataset_arguments = [dataset_args, self.tokenizer]
            self.dataset_kwargs = {}
        else: # args, seq_tokenizer, cls_tokenizer, cls_model, padding_num: int, 
            self.dataset_arguments = [dataset_args]
            self.dataset_arguments += [tgwv_kwargs[x] for x in ['seq_tokenizer', 'cls_tokenizer', 'cls_model', 'padding_num']]
            self.dataset_kwargs = {'v_mode': tgwv_kwargs['v_mode']}
           

        if mode == 'fit':
            self.train_loader = self.get_dataloader("train", shuffle=True)

    def save_wir_at_start(self, dataset, batch_size):
        wir_pt_path = os.path.join(dataset.data_dir, "wir", dataset.type_path)

        if self.dataset_class == TGWVDataset and not os.path.exists(wir_pt_path):
            print('iterate all the data to save wir. ')
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.args.num_workers,
                sampler=None,
            )
            try:
                loader = iter(loader)
                while True:
                    next(loader)
            except StopIteration:
                print('the wir should be saved during iteration. ')
        else:
            print('the wir path already exists. ')

    def get_dataset(self, type_path):
        dataset = self.dataset_class(
            *self.dataset_arguments, 
            type_path=type_path,
            **self.dataset_kwargs, 
        )
        return dataset

    def get_dataloader(
        self, 
        type_path: str, 
        shuffle: bool = False, 
        batch_size: int = None, 
    ) -> DataLoader:
        dataset = self.get_dataset(type_path)
        batch_size = self.b_size[type_path] if batch_size is None else batch_size
        dataset.batch_size, dataset.type_path = batch_size, type_path
        if self.dataset_mode == "seq2seq": self.save_wir_at_start(dataset, batch_size)

        if dataset.args.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.args.num_workers,
                sampler=sampler,
                drop_last=True, 
            )

        elif self.args.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.args.max_tokens_per_batch, distributed=self.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.args.num_workers,
                # batch_size=None,
                drop_last=True, 
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.args.num_workers,
                sampler=None,
                drop_last=True, # just used for fixing the problem that load data separately in last batch
            )
            return loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("val", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", shuffle=False)


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__ == '__main__':
    parser = HfArgumentParser((Seq2SeqDataset.add_specific_args(), 
                    ClassificationDataset.add_specific_args(), 
                    DataModule.add_specific_args()))
    
    seq2seq_args, cls_args, loader_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    d = Seq2SeqDataset(seq2seq_args, tokenizer)
    d_cls = ClassificationDataset(cls_args, tokenizer)
    d_loader = DataModule(loader_args, seq2seq_args, 'seq2seq', tokenizer=tokenizer)
    cls_loader = DataModule(loader_args, cls_args, 'classification', tokenizer=tokenizer)
    for a, b in zip(d_loader.train_dataloader(), cls_loader.train_dataloader()):
        import pdb 
        pdb.set_trace()