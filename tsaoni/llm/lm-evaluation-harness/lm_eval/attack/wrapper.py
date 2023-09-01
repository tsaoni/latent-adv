"""
HuggingFace Model Wrapper
--------------------------
"""

import torch
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast, 
)

from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer
from textattack.models.wrappers import PyTorchModelWrapper

#from textattack.pytorch_model_wrapper import PyTorchModelWrapper
#from lm_eval.utils import chunks
torch.cuda.empty_cache()

class LLMWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, query_fn=None, encode_pair_fn=None):
        assert isinstance(
            model, (PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                PreTrainedTokenizer,
                PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.query_fn = query_fn
        self.encode_pair_fn = encode_pair_fn

    def __call__(self, text_input_list):
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        # text_input_list size: (batch_size, n_choices, 2)
        batch_logits = []
        for text_input in text_input_list:
            res = []
            n_choices = len(text_input)
            for i, chunk in enumerate(text_input):
                context_enc, continuation_enc = self.encode_pair_fn(*chunk)
                res += self.query_fn(
                    [(i, context_enc, continuation_enc)]
                )
            logits = torch.tensor([x[0] for x in res]).to(self.model.device)
            batch_logits.append(logits)
        
        return torch.stack(batch_logits, dim=0)
        """
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits
        """

    """ unimplemented """
    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
