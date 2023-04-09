# all
from dataclasses import dataclass, field

# main
import os, sys
import json
import argparse
import pytorch_lightning as pl
from typing import List, Optional
from argparse import Namespace
from pathlib import Path
# from transformers import (
# )

# callback
from pytorch_lightning.loggers import WandbLogger

# trainer
import abc
import os, sys
import time
import copy
import argparse
import torch
import evaluate
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser, 
    BartForConditionalGeneration,
    MBartTokenizer, 
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    T5ForConditionalGeneration, 
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.models.bart.modeling_bart import shift_tokens_right # for seq2seq model

# model
import torch.nn as nn

# utils
import argparse
import pickle
import itertools
import linecache
import logging
import os
import numpy as np
import pytorch_lightning as pl
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Union
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from rouge_score import rouge_scorer, scoring
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from transformers import (
    BartTokenizer,
)

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


try:
    from fairseq.data.data_utils import batch_by_size
    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False
