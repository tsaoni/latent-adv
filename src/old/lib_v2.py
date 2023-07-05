from dataclasses import dataclass, field
import os, sys, json
import random, math
import argparse
import linecache
from typing import Callable, Dict, Iterable, List, Union, Optional
from argparse import Namespace
from pathlib import Path
import wandb
import os, sys
import time
import copy
import argparse
import torch
import evaluate
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler
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
    BartTokenizer,
    BartForSequenceClassification, 
    BartConfig, 
    BartModel, 
    DataCollatorForLanguageModeling, 
    DataCollatorForSeq2Seq, 
    LineByLineTextDataset, 
    SchedulerType,
    get_scheduler,
    GlueDataset, 
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,  # for seq2seq model
    Seq2SeqSequenceClassifierOutput, 
    BartClassificationHead, 
    Seq2SeqModelOutput, 
    BaseModelOutput, 
    BartEncoder, 
)
import torch.nn as nn
from torch.autograd import Variable
#import torchvision.transforms as transforms
#from PIL import Image
from torch.nn import (
    MSELoss, 
    CrossEntropyLoss, 
    BCEWithLogitsLoss, 
)
from functools import cached_property
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from rouge_score import rouge_scorer, scoring
from transformers.utils import ModelOutput

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from transformers.adapters import (
    AdapterConfig
)
from tqdm.auto import tqdm


try:
    from fairseq.data.data_utils import batch_by_size
    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False
