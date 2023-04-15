from lib import *

from dataset import *
from utils import (
    shift_tokens_right, 
    set_specific_attr, 
    lmap, 
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
}

# model that generates text conditioned on latent vector

class Seq2SeqModel(nn.Module):
    def __init__(self, args, **config_kwargs):
        super().__init__()
        args = set_specific_attr(args, get_model_specific_attr())
        self.args = args
        self.model_type = 'seq2seq'
        self.config, self.tokenizer, self.model = get_model_from_args(args, **config_kwargs)

    def extra_settings(self, ):
        if isinstance(self.model, BartForConditionalGeneration):
            extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
            for p in extra_model_params:
                if getattr(self.args, p, None):
                    assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                    setattr(self.config, p, getattr(self.args, p))

        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        else:
            self.decoder_start_token_id = self.model.config.decoder_start_token_id

    def forward(self, **batch):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, \
                                    decoder_start_token_id=decoder_start_token_id)

        # outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
        #                use_prefix=True, return_dict=True, labels=tgt_ids)
        #
        # return (outputs.loss,)

        outputs = self.model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)
                        # use_prefix=True)

        return outputs

    def generate(self, batch: dict, return_ids=True):
        print('for deocoding, eval_max_length={}, eval_min_length={}, eval_beams={}'\
            .format(self.args.eval_max_gen_length, self.args.eval_min_gen_length, self.args.eval_beams))

        seq2seq_model_type = (
            BartForConditionalGeneration, 
            GPT2LMHeadModel, 
        )
        if isinstance(self.model, GPT2LMHeadModel):
            # todo: set gpt parameters
            output_sequences = self.model.generate(
                input_ids=batch["input_ids"],
                emb_match=None,
                control_code=control_code,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=0.8,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )
        elif isinstance(self.model, BartForConditionalGeneration): 
            output_sequences = self.model.generate(
                batch["input_ids"],
                # past_key_values=None,
                attention_mask=batch["attention_mask"],
                use_cache=True,
                length_penalty=self.args.length_penalty,
                # use_prefix=True,
                decoder_start_token_id=self.config.decoder_start_token_id,
                num_beams=self.args.eval_beams,
                min_length=self.args.eval_min_gen_length,
                max_length=self.args.eval_max_gen_length,
            )
        else: 
            assert isinstance(self.model, seq2seq_model_type)

        if return_ids:
            return output_sequences
        else:
            return self.ids_to_clean_text(output_sequences)

    def ids_to_clean_text(self, generated_ids: torch.Tensor):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    @staticmethod
    def add_specific_args():
        @dataclass
        class ModelArguments:
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

        return ModelArguments

class ClassificationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        args = set_specific_attr(args, get_model_specific_attr())
        self.args = args
        self.model_type = 'cls'
        self.config, self.tokenizer, self.model = get_model_from_args(args)

    def forward(self, **batch):
        outputs = self.model(**batch)
        return outputs

    @staticmethod
    def add_specific_args():
        @dataclass
        class ModelArguments:
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
            
        return ModelArguments

class TextGenerationWithVectorInputModel(Seq2SeqModel):
    def __init__(self, args, tgwv_args):
        config_kwargs = tgwv_args.__dict__
        self.original_embed_dim = 1024
        super().__init__(args, **config_kwargs)
        self.model_type = 'tgwv'
        embed_matrix = self.model.get_input_embeddings()
        self.embed_matrix = torch.nn.Embedding.from_pretrained(
            embed_matrix.weight[:, :self.original_embed_dim],
            padding_idx=self.tokenizer.pad_token_id, 
        )
        self.embed_matrix.weight.requires_grad = True

    # get batch from text_generation_with_vector_input_collate_fn
    def forward(self, **batch):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        cls_logits = batch['cls_logits']

        # create input embeddings
        embed = self.embed_matrix(src_ids)
        embed = torch.concat((embed, cls_logits), dim=-1)

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, \
                                    decoder_start_token_id=decoder_start_token_id)

        # outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
        #                use_prefix=True, return_dict=True, labels=tgt_ids)
        #
        # return (outputs.loss,)

        #outputs = self.model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)
        outputs = self.model(inputs_embeds=embed, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)
                        # use_prefix=True)

        return outputs

    def generate(self, batch: dict, return_ids=True):
        print('for decoding, eval_max_length={}, eval_min_length={}, eval_beams={}'\
            .format(self.args.eval_max_gen_length, self.args.eval_min_gen_length, self.args.eval_beams))

        seq2seq_model_type = (
            BartForConditionalGeneration, 
            GPT2LMHeadModel, 
        )
        if isinstance(self.model, GPT2LMHeadModel):
            # todo: set gpt parameters
            output_sequences = self.model.generate(
                input_ids=batch["input_ids"],
                emb_match=None,
                control_code=control_code,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=0.8,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )
        elif isinstance(self.model, BartForConditionalGeneration): 
            output_sequences = self.model.generate(
                batch["input_ids"],
                # past_key_values=None,
                attention_mask=batch["attention_mask"],
                use_cache=True,
                length_penalty=self.args.length_penalty,
                # use_prefix=True,
                decoder_start_token_id=self.config.decoder_start_token_id,
                num_beams=self.args.eval_beams,
                min_length=self.args.eval_min_gen_length,
                max_length=self.args.eval_max_gen_length,
            )
        else: 
            assert isinstance(self.model, seq2seq_model_type)

        if return_ids:
            return output_sequences
        else:
            return self.ids_to_clean_text(output_sequences)


    @staticmethod
    def add_specific_args():
        seq2seq_args_class = super(TextGenerationWithVectorInputModel, TextGenerationWithVectorInputModel).add_specific_args()
        @dataclass
        class TGWVModelArguments:
            d_model: Optional[int] = field(
                default=1024+16, 
                metadata={"help": "the embedding dim. "}
            )

        return [seq2seq_args_class, TGWVModelArguments]

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

def get_model_specific_attr():
    return ['model_name_or_path', 'config_name', 'tokenizer_name', 'cache_dir', 'use_big', 
                    'model_mode', 'num_labels']

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
        
