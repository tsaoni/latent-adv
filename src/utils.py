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

def check_same_model_weight(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(param1, param2):
            print(f"Weight mismatch found: {name1}")

def text_generation_with_vector_input_collate_fn(
        seq_len, 
        cls_batch, 
        cls_model, 
        padding_num: int, 
        mask_id=None, 
        mode='softmax', # options: raw, softmax, drop
        wir_pt_path=None, 
        batch_idx=None, 
        id_list=None, 
        load=True, 
        #dataset_path=None, 
        #cls_model_name=None, 
    ):
    assert mask_id is not None or not mode == 'drop'
    outs = cls_model(**cls_batch)
    softmax_fn = torch.nn.Softmax(dim=-1)
    logits = softmax_fn(outs.logits.detach()) if mode in ['softmax', 'drop'] else outs.logits.detach()
    stack_logits = torch.stack(tuple([logits] * seq_len), dim=1)
    if mode in ['softmax', 'raw']:
        pad_logits = torch.nn.functional.pad(
            stack_logits, 
            pad=(0, padding_num), 
            mode='constant', 
            value=0.0, 
        )
        return pad_logits
    else: # drop
        return word_importance_logits(cls_batch, cls_model, stack_logits, padding_num, mask_id, \
                                      wir_pt_path=wir_pt_path, batch_idx=batch_idx, id_list=id_list, load=load)

def load_wir_from_file(wir_pt_path, id_list, seq_len):
    # print(f'load wir tensor from {batch_idx * bz} to {upper_bd}. ')
    wir_file_name = [os.path.join(wir_pt_path, f"{idx}.pt") for idx in id_list]
    wir_list = []
    max_wir_len = 0
    for pth in wir_file_name:
        wir = torch.load(pth)
        wir_list.append(wir)
        if len(wir) > max_wir_len: max_wir_len = len(wir)
    
    # pad to same length first, then pad/truncate to seq_len
    for i, wir in enumerate(wir_list):
        wir_list[i] = torch.nn.functional.pad(
            wir, 
            pad=(0, 0, 0, max_wir_len - len(wir)), 
            mode='constant', 
            value=0.0, 
        )
    if max_wir_len > seq_len:
        batch_wir = torch.stack(wir_list, dim=0).narrow(1, 0, seq_len)
    else:
        batch_wir = torch.nn.functional.pad(
            torch.stack(wir_list, dim=0), 
            pad=(0, 0, 0, seq_len - max_wir_len), 
            mode='constant', 
            value=0.0, 
        )

    return batch_wir

def save_wir_to_file(wir_batch_tensor, wir_pt_path, id_list):
    wir_file_name = [os.path.join(wir_pt_path, f"{idx}.pt") for idx in id_list ]
    for pth, wir_tensor in zip(wir_file_name, wir_batch_tensor):
        torch.save(wir_tensor, pth)

def word_importance_logits(
        cls_batch, cls_model, unmasked_outs, padding_num: int, mask_id, 
        wir_pt_path=None, 
        batch_idx=None, 
        id_list=None, 
        load=True, 
    ):
    """
     "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            **token_type_ids_kwargs, 
    """
    bz, seq_len = cls_batch['input_ids'].shape
    seq2seq_batch_len = unmasked_outs.shape[1]

    first_load_f = None
    if wir_pt_path is not None and load:
        first_load_f = os.path.join(wir_pt_path, f"{batch_idx * bz}.pt")
        if os.path.exists(first_load_f):
            return load_wir_from_file(wir_pt_path, id_list, seq2seq_batch_len)
    else:
        stacked_cls_batch = dict()
        for key in cls_batch.keys():
            reshape_size = (-1, ) if key == 'labels' else (-1, seq_len)
            stacked_cls_batch[key] = torch.stack([cls_batch[key]] * seq_len, dim=1)
            if key == 'input_ids':
                # set mask id
                mask_tensor = torch.diag(torch.tensor([True] * seq_len))
                mask_tensor = torch.stack([mask_tensor] * bz, dim=0)
                stacked_cls_batch[key][mask_tensor] = mask_id
            stacked_cls_batch[key] = torch.reshape(stacked_cls_batch[key], reshape_size)
        
        stacked_out = cls_model(**stacked_cls_batch)
        softmax_fn = torch.nn.Softmax(dim=-1)
        stacked_out_logits = softmax_fn(stacked_out.logits.detach())
        stacked_out_logits = torch.reshape(stacked_out_logits, (bz, seq_len, -1))
        # calc word important logits
        if seq_len < seq2seq_batch_len: # pad
            padding = [stacked_out_logits[:, -1:, :]] * (seq2seq_batch_len - seq_len)
            stacked_out_logits = torch.cat([stacked_out_logits, *padding], dim=1)
        else:
            stacked_out_logits = stacked_out_logits[:, :seq2seq_batch_len, :]
        word_important_logits = unmasked_outs - stacked_out_logits
        # word_important_logits = softmax_fn(word_important_logits) # the values are similar

        stacked_out_pad_logits = torch.nn.functional.pad(
            word_important_logits, 
            pad=(0, padding_num), 
            mode='constant', 
            value=0.0, 
        )
        if first_load_f is not None and not os.path.exists(first_load_f):
            os.makedirs(wir_pt_path, exist_ok=True)
            save_wir_to_file(stacked_out_pad_logits, wir_pt_path, id_list)

        return stacked_out_pad_logits

def is_special_token(token, tokenizer):
    special_tokens = np.unique([v for k, v in tokenizer.init_kwargs.items() if k[-5:] == 'token']).tolist()
    return token in special_tokens

def wir_logit_to_color_range(logits):
    logit_min_mask = logits < 0
    logits_pos = torch.abs(logits)
    logits_pos_biased = logits_pos / logits_pos[logits_pos > 0].min()
    log_logits = torch.log(logits_pos_biased)
    log_logits[log_logits == -torch.inf] = 0.0
    w, b = torch.max(log_logits) - torch.min(log_logits), torch.min(log_logits)
    log_logits = (log_logits - b) / w
    log_logits[logit_min_mask] = 0.0
    return log_logits

def show_wir_in_html(
        opts=None,
        seq_len=None, 
        ptb_len=None, 
        cls_batch=None, 
        cls_model=None, 
        cls_tokenizer=None, 
        display_filename=None,  
    ):
    assert set(opts).issubset(['insert', 'write', 'show'])
    global display_texts
    if 'display_texts' not in globals().keys(): 
        display_texts = defaultdict(list)

    if 'insert' in opts:
        cls_logits = text_generation_with_vector_input_collate_fn(
            seq_len, cls_batch, cls_model, 
            padding_num=0, 
            mask_id=cls_tokenizer.mask_token_id, 
            mode='drop',
            #id_list=cls_batch['ids'], 
            load=False, 
        )

        cls_logits = wir_logit_to_color_range(cls_logits)
        for ids, logits in zip(cls_batch['input_ids'], cls_logits):
            disp_text_info_list = []
            for id, logit in zip(ids, logits):
                def get_color_info(token, logit):
                    if is_special_token(token, cls_tokenizer): return []

                    def range_mapping(num, src_range=(0, 1), tgt_range=(0, 255)):
                        diff_fn = lambda x: x[1] - x[0]
                        return int(tgt_range[0] + (num - src_range[0]) * diff_fn(tgt_range) / diff_fn(src_range))
                    
                    if np.argmax(logit) == 0: color = (0, 0, range_mapping(logit[0]))
                    else: color = (range_mapping(logit[1]), 0, 0)
                    return [{"text": token, "color": color}]
                
                disp_text_info_list += get_color_info(cls_tokenizer.decode([id]), logit)

            # turn display text info list into colored text
            final_disp_info_list = []
            for info in disp_text_info_list:
                if "##" == info['text'][0:2]:
                    if len(final_disp_info_list) > 0:
                        elementwise_add_fn = lambda x, y: tuple([m+n for m, n in zip(x, y)])
                        final_disp_info_list[-1]['color'] = elementwise_add_fn(final_disp_info_list[-1]['color'], info['color'])
                        final_disp_info_list[-1]['text'] += info['text'][2:]
                        final_disp_info_list[-1]['subword_num'] += 1
                else:
                    final_disp_info_list.append(info)
                    final_disp_info_list[-1].update({'subword_num': 1})
            colored_text = lambda x: '<span style="color:rgb{};">{}</span>'.format(
                tuple([int(i / x['subword_num']) for i in x['color']]), x['text'], 
            )
            text_list = [colored_text(info) for info in final_disp_info_list]
            disp_text = " ".join(text_list)
            display_texts['ptb={:.2f}'.format(ptb_len)].append(disp_text)


    if 'write' in opts:
        pd.set_option('display.max_colwidth', 60)
        df = pd.DataFrame(display_texts)
        html_table = df.to_html(escape=False)
        with open(display_filename, 'w') as f:
            f.write(html_table)
    if 'show' in opts:
        os.system('python -m http.server --directory .')


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

def colored_in_html(label, tuple_num):
    if not label == np.argmax(tuple_num):
        print(label, tuple_num)
    if tuple_num[0] > tuple_num[1]:
        color = "blue" #if label == np.argmax(tuple_num) else "green"
        return '(<span style="color:{};">{:.4f}</span>, {:.4f})'.format(color, *tuple_num)
    else:
        color = "red" #if label == np.argmax(tuple_num) else "green"
        return '({:.4f}, <span style="color:{};">{:.4f}</span>)'.format(tuple_num[0], color, tuple_num[1])
    
def show_texts_in_html(df, ptb_len=None, df_out=None, word_range=None, show_html=False, html_name='text.html'): # colored_text_list=None, mode_list=['del', 'ins'], 
    # special bug: the dataframe can be modified when create in local scope
    df = copy.deepcopy(df)
    """
    df = pd.DataFrame({'original': ['i am a very cute tsaoni. ', 'she was doing laundries. '],
                    'perturb1': ['there is a cute horse.', 'he is doing laundry. '],})
                    #'perturb2': ['it was eating tsai tsai.', 'doing laundry is his favorite. '],
                    #'perturb3': ['so curious the tsaoni is.', 'he dont like doing hw. '],})
    """
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
                color_info_opt.multi_colored_text(idx, s=s_origin, \
                                            colored_texts=colored_texts[0], ptb_idx=ptb_idx, action='set')

        #words = s.split()
        highlighted_words = []
        for info in text_color_info:
            highlighted_words.append(f'<span style="color:{info["color"]};">{info["text"]}</span>')
        return ''.join(highlighted_words)

    def ptb_word_range_highlight(s, ptb_len, word_range):
        color = (255, int(ptb_len * 255), 0)
        s_len = len(s.split())
        prv, nxt = (0, word_range[0]), (word_range[1], s_len)
        prv_str = '' if word_range[0] == 0 else ' '.join([*s.split()[slice(*prv)], ''])
        nxt_str = '' if word_range[1] == s_len else ' '.join(['', *s.split()[slice(*nxt)]])
        highlight_s = ' '.join(s.split()[slice(*word_range)])
        color_fn = lambda c: f'rgb{c}'
        html_s = f'<span style="background-color:{color_fn(color)};">{highlight_s}</span>'
        return ''.join([prv_str, html_s, nxt_str])

    # colored_text_list dim:: 0: sent num, 1: ptb num, 2: 2, 3: colored num

    for i in range(n):
        for j in range(1, m):
            ptb_range_text = ptb_word_range_highlight(df.iloc[i][0], ptb_len, word_range[j-1][i])
            diff_text = highlight_text(df.iloc[i][j], s_origin=df.iloc[i][0], \
                                           idx=i, ptb_idx=j-1, colored_texts=colored_text_list[i][j-1])
            df.iloc[i][j] = '<br><br>'.join([ptb_range_text, diff_text])
            
        df.iloc[i][0] = highlight_text(df.iloc[i][0], idx=i, ptb_idx=-1, )

    pd.set_option('display.max_colwidth', 60)
    if df_out is not None:
        final_df = pd.concat([pd.concat([df.iloc[[i]], df_out.iloc[[i]]]) for i in range(len(df))])
        html_table = final_df.to_html(escape=False)
    else:
        html_table = df.to_html(escape=False)
    #styled_df = highlight_text(df)

    # render the styled dataframe as an HTML table
    #html_table = styled_df.render()

    # print the HTML table
    #print(html_table)
    with open(html_name, 'w') as f:
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

def generate_len_file(src_file_name, len_file_name):
    with open(src_file_name, 'r') as src_f, open(len_file_name, 'w') as len_f:
        lens = []
        for line in src_f:
            words = line.split()
            lens.append(f"{len(words)}")
        len_f.write('\n'.join(lens))

def get_dataset_global(global_name):
    import dataset
    return getattr(dataset, global_name)

def get_module_global(global_name):
    import module
    return getattr(module, global_name)

def get_model_global(global_name):
    import model
    return getattr(model, global_name)

def sigmoid(x: torch.FloatTensor, a=1.0):
    return 1. / (1. + torch.exp(-a*x))