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
        if token_type_ids is None: 
            return (input_ids[:, keep_column_mask], )
        else: 
            return (input_ids[:, keep_column_mask], token_type_ids[:, keep_column_mask])
    else:
        if token_type_ids is None: 
            return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], )
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

def get_diff_substr(src: str, tgt: str) -> Dict:
    n = len(src)
    m = len(tgt)
    dist_m = np.zeros([n + 1, m + 1], dtype=int).tolist()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0: dist_m[0][j] = j
            elif j == 0: dist_m[i][0] = i
            elif src[i - 1] == tgt[j - 1]:
                dist_m[i][j] = dist_m[i - 1][j - 1]
            else:
                dist_m[i][j] = 1 + min(dist_m[i - 1][j], dist_m[i][j - 1])

    """ debug
    tmp = copy.deepcopy(dist_m)
    for i in range(n + 1):
        for j in range(m + 1):
            tmp[i][j] = str(tmp[i][j])
    tgt_list, src_list = ['#', '#'], ['#']
    tgt_list += [tgt[i] for i in range(m)]
    print(tgt_list)
    src_list += [src[i] for i in range(n)]
    for i in range(n+1):
        r = [src_list[i]] + tmp[i]
        print(r)

    tgt_list = ['#']
    tgt_list += [tgt[i] for i in range(m)]
    """
    # find ins/del substrings
    substr = {'ins': [''], 'del': ['']}
    i, j = n, m
    while i + j > 0:
        # print(f'i={i}, src={src_list[i]}, j={j}, tgt={tgt_list[j]}')
        if i == 0:
            substr['ins'][0] = tgt[j - 1] + substr['ins'][0]
            j -= 1
        elif j == 0:
            substr['del'][0] = src[i - 1] + substr['del'][0]
            i -= 1
        elif src[i - 1] == tgt[j - 1]:
            i -= 1
            j -= 1
            for k in substr.keys():
                if not substr[k][0] == '':
                    substr[k].insert(0, '')
        elif dist_m[i][j - 1] + 1 == dist_m[i][j]:
            substr['ins'][0] = tgt[j - 1] + substr['ins'][0]
            j -= 1
        else:
            substr['del'][0] = src[i - 1] + substr['del'][0]
            i -= 1

    # remove strings with empty spaces
    for k in substr.keys():
        striped = np.array(list(map(lambda x: x.strip(), substr[k])))
        substr[k] = striped[striped != ''].tolist()

    return substr

def show_texts_in_html(df=None, colored_text_list=None, mode_list=['del', 'ins'], show_html=False):
    #assert df.shape == np(colored_texts).shape[:-1]
    import pandas as pd
    # create a sample dataframe
    df = pd.DataFrame({'original': ['i am a very cute tsaoni. ', 'she was doing laundries. '],
                    'perturb1': ['there is a cute horse.', 'he is doing laundry. '],})
                    #'perturb2': ['it was eating tsai tsai.', 'doing laundry is his favorite. '],
                    #'perturb3': ['so curious the tsaoni is.', 'he dont like doing hw. '],})

    n, m = df.shape
    colored_text_list = []
    for i in range(n):
        single_colored_text_list = []
        for j in range(1, m):
            substr_dict = get_diff_substr(df.iloc[i][0], df.iloc[i][j])
            single_colored_text_list.append([substr_dict['del'], substr_dict['ins']])
        colored_text_list.append(single_colored_text_list)
    
    """
    # create a function to apply color styles to text
    def color_text(text):
        if text == 'Alice':
            return 'color: red'
        elif text == 'Bob':
            return 'color: green'
        elif text == 'Charlie':
            return 'color: blue'
        else:
            return 'color: black'

    # apply the color_text function to the entire dataframe
    styled_df = df.style.applymap(color_text)
    """

    class TextColorInfo:
        mode_color = {
            'ins': (255, 0, 0), 
            'del': (0, 0, 255), 
            'ptb': [], 
            'bg': (0, 0, 0), 
        }

        def __init__(self, multi_color_texts: List =None):
            if multi_color_texts is not None:
                self.multi_text_color_info = []
                for text in multi_color_texts:
                    self.multi_text_color_info.append([{'text': x, 'color': -1, 'replace_num': 0} for x in text])
                max_n = 100
                if m > 2: color_list = [x * int(max_n / (m - 2)) for x in range(m - 1)]
                else: color_list = [0, 100]
                for color in color_list: self.set_color('ptb', color)

        def get_color(self, mode=None, ptb_idx=None, pos: Tuple =None):
            if pos is not None:
                i, j = pos
                color = self.multi_text_color_info[i][j]['color']
                if color == -1: 
                    color = (0, 0, 0)
                else: # colored
                    color = get_color_mapping(color)
            elif mode == 'ptb':
                color = get_color_mapping(self.mode_color[mode][ptb_idx])
            else:
                color = self.mode_color[mode]
            return f'rgb({color[0]}, {color[1]}, {color[2]})'

        def set_color(self, mode, color: Union[Tuple, int]):
            if mode == 'ptb':
                self.mode_color[mode].append(color) # store int
            else:
                self.mode_color[mode] = color
        
        def single_colored_text(self, s, colored_texts, ptb_idx=None, mode=None):
            text_color_info = []
            for text in colored_texts:
                split_list = s.split(text)
                text_color_info += [
                    {'text': split_list[0], 'color': self.get_color('bg'), }, 
                    {'text': text, 'color': self.get_color(ptb_idx=ptb_idx, mode=mode), }, 
                ]
                del_text = split_list[0] + text
                s = s[len(del_text):] if len(s) > len(del_text) else ''

            text_color_info.append({'text': s, 'color': self.get_color('bg'), })
            return text_color_info

        def multi_colored_text(self, idx, s=None, colored_texts=None, ptb_idx=None, action=None):
            if action == 'set':
                def get_colored_idxs(s, colored_text):
                    split_list = s.split(colored_text)
                    if len(split_list) == 1: return []
                    start, end = len(split_list[0]), len(split_list[0] + colored_text)
                    return [x for x in range(start, end)]
                colored_idxs = []
                for text in colored_texts:
                    colored_idxs += get_colored_idxs(s, text) 
                # reset color
                interpolate_fn = lambda l1, l2: sum([l1[i] * l2[i] for i in range(2)]) / sum(l2)
                color = self.mode_color['ptb'][ptb_idx]
                for i in colored_idxs:
                    l1 = (self.multi_text_color_info[idx][i]['color'], color)
                    l2 = (self.multi_text_color_info[idx][i]['replace_num'], 1)
                    self.multi_text_color_info[idx][i]['color'] = interpolate_fn(l1, l2)
                    self.multi_text_color_info[idx][i]['replace_num'] += 1

            elif action == 'get':
                text_color_info = []
                for i, x in enumerate(self.multi_text_color_info[idx]):
                    info = copy.deepcopy(x)
                    info['color'] = self.get_color(pos=(idx, i))
                    text_color_info.append(info)
                return text_color_info
            else:
                raise ValueError('action should be one of the values, `set` and `get`. ')

    multi_colored_texts = df.iloc[:, 0].tolist()
    color_info_opt = TextColorInfo(multi_colored_texts)
    def highlight_text(s, s_origin=None, idx=None, ptb_idx=None, colored_texts=None, mode='ptb'):
        #if mode in ['ins', 'del']:
        #    text_color_info = color_info_opt.single_colored_text(s, colored_texts, mode)
        if mode == 'ptb':
            if ptb_idx == -1:
                text_color_info = color_info_opt.multi_colored_text(idx, action='get')
            else:
                text_color_info = color_info_opt.single_colored_text(s, colored_texts[1], ptb_idx=ptb_idx, mode=mode)
                color_info_opt.multi_colored_text(idx, s=s_origin, 
                                            colored_texts=colored_texts[0], ptb_idx=ptb_idx, action='set')

        #words = s.split()
        highlighted_words = []
        for info in text_color_info:
            highlighted_words.append(f'<span style="color:{info["color"]};">{info["text"]}</span>')
        return ''.join(highlighted_words)

    # colored_text_list dim:: 0: sent num, 1: ptb num, 2: 2, 3: colored num
    for i in range(n):
        for j in range(1, m):
            df.iloc[i][j] = highlight_text(df.iloc[i][j], s_origin=df.iloc[i][0], 
                                           idx=i, ptb_idx=j-1, colored_texts=colored_text_list[i][j-1])
        df.iloc[i][0] = highlight_text(df.iloc[i][0], idx=i, ptb_idx=-1, )
    html_table = df.to_html(escape=False)
    #styled_df = highlight_text(df)

    # render the styled dataframe as an HTML table
    #html_table = styled_df.render()

    # print the HTML table
    #print(html_table)
    with open('text.html', 'w') as f:
        f.write(html_table)
    if show_html:
        os.system('python -m http.server --directory .')

def get_color_mapping(num, min_n=0, max_n=100, ub="ba08ba", lb="08ba08"):
    assert num >= min_n and num <= max_n
    def hex_to_TupleInt(hex_s: str):
        assert len(hex_s) == 6
        return tuple([int(hex_s[idx:idx+2], 16) for idx in range(0, 6, 2)])
    
    ub = hex_to_TupleInt(ub) if isinstance(ub, str) else ub
    lb = hex_to_TupleInt(lb) if isinstance(lb, str) else lb
    
    dist = int(sum([abs(x - y) for x, y in zip(ub, lb)]) / (max_n - min_n) * (max_n - num))
    ret_color = list(copy.deepcopy(ub))

    def reduce(dist, idx, ub, lb):
        direct = 1 if ub[idx] - lb[idx] > 0 else -1
        return direct * min(dist, abs(ub[idx] - lb[idx]))

    for i in range(3):
        if dist > 0:
            ret_color[i] = ret_color[i] - reduce(dist, i, ub, lb)
            dist -= abs(ub[i] - lb[i])
        
    return tuple(ret_color)