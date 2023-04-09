from lib import *


def get_scheduler_info():
    # update this and the import above to support new schedulers from transformers.optimization
    arg_to_scheduler = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
        "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
        "polynomial": get_polynomial_decay_schedule_with_warmup,
        # '': get_constant_schedule,             # not supported for now
        # '': get_constant_schedule_with_warmup, # not supported for now
    }
    arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
    arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

    scheduler_dict = dict(
        arg_to_scheduler=arg_to_scheduler,
        arg_to_scheduler_choices=arg_to_scheduler_choices,
        arg_to_scheduler_metavar=arg_to_scheduler_metavar,
    )

    return scheduler_dict

def set_specific_attr(args, attrs):
    args = argparse.Namespace(**args.__dict__)
    args_dict = copy.deepcopy(args.__dict__)
    for key, value in args_dict.items():
        for a in attrs:
            if a in key:
                setattr(args, a, value)
                delattr(args, key)
                print(f'the original attribute is {key}, new attribute is {a}. the value is set to {value}')
    return args

def label_to_tensor(label: str):
    return torch.Tensor([int(label)])

def encode_line(tokenizer, line, max_length, dataset_name=None, pad_to_max_length=True, return_tensors="pt"):
    #check_variable_status(dataset_name, name='dataset_name')
    #is_nli = check_nli_dataset(dataset_name)
    is_nli = False
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    separater = '@separater@'
    texts = (sent.strip() for sent in line.split(separater)) if is_nli else (line, )
    return tokenizer(
        [*texts],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def trim_batch(
    input_ids,
    token_type_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return (input_ids[:, keep_column_mask], token_type_ids[:, keep_column_mask])
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], token_type_ids[:, keep_column_mask])

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def use_task_specific_params(model, task=None):
    """Update config with summarization specific params."""
    if task is not None:
        task_specific_params = model.config.task_specific_params

        if task_specific_params is not None:
            pars = task_specific_params.get(task, {})
            check_task_specific_params_type(pars)
            print(f"using task specific params for {task}: {pars}")
            model.config.update(pars)

def check_task_specific_params_type(pars):
    int_params = ['num_labels']
    float_params = []
    for param in int_params:
        if param in pars.keys():
            pars[param] = int(pars[param])
    for param in float_params:
        if param in pars.keys():
            pars[param] = float(pars[param])

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)

def load_json(path):
    with open(path) as f:
        return json.load(f)

''' check functions '''

def check_argument_setting(args, arg_name):
    if arg_name == 'task':
        assert args.task in ['agnews', 'mrpc', 'news-summary']
    elif arg_name == 'model_mode':
        assert args.model_mode in ['base', 'sequence-classification', 'question-answering', \
            'pretraining', 'token-classification', 'language-modeling', \
            'summarization', 'translation']

def check_variable_status(variable, name="", status='None'):
    if status == 'None':
        if variable is None:
            raise ValueError('{} parameter should not be none. '.format(name))

def check_parameter_value(hparams, param_name_list, check_all=False):
    if isinstance(hparams, argparse.Namespace):
        hparams = vars(hparams)
    # not_none_param = ['model_mode', 'model_name_or_path', 'config_name', 'tokenizer_name']
    check_exist = False
    for param_name in param_name_list:
        if hparams[param_name] is None:
            if check_all:
                raise ValueError('{} parameter should not be none. '.format(param_name))
        else:
            check_exist = True
    if not check_exist:
        raise ValueError('paramters in list should have at least one not none value. ')

def check_nli_dataset(dataset_name):
    nli = []
    if dataset_name in nli:
        return True
    return False
