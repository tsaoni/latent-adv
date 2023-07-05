from lib import *
from model import LatentClassfier
from utils import *

'''
latent-adv-exp1
../models/t=tgwv_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls_cls_m=Hazqeel.electra-small-finetuned-sst2-rotten_tomatoes-distilled
'''

def max_common_substring(seq1, seq2, ret_pos='first'):
    n, m = len(seq1), len(seq2)
    len_m = np.zeros((n + 1, m + 1), dtype=int).tolist()
    max_pos = (1, 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                len_m[i][j] = len_m[i - 1][j - 1] + 1
                if len_m[i][j] > len_m[max_pos[0]][max_pos[1]]:
                    max_pos = (i, j)
            else:
                len_m[i][j] = 0

    max_len = len_m[max_pos[0]][max_pos[1]]
    ret_pos_dict = {
        'first': (max_pos[0] - max_len, max_pos[0]), 
        'second': (max_pos[1] - max_len, max_pos[1]), 
    }
    return ret_pos_dict[ret_pos]


class PerturbEval():
    def __init__(
        self, generator, eval_tokenizer, eval_model, data_loader, 
        embed_type='probing', 
        perturb_range=(0, 1), 
        use_wandb=False, 
        output_dir=None, 
        max_source_length=10, 
        ptb_mode='softmax', 
        ptb_param="embed", 
        latent_ckpt=None, 
        pool=None, 
    ):
        self.config = argparse.Namespace(
            similarity_fn=None, 
            perturb_inc_rate=1.0, 
            max_perturb_step=5, 
            embed_type=embed_type, # unify, drop, 
            perturb_range=perturb_range, 
            use_wandb=use_wandb, #use_wandb, 
            output_dir=None, # for wandb proj name, beware not to exceed 128 chars
            max_source_length=max_source_length, 
            ptb_mode=ptb_mode, 
            ptb_param=ptb_param, 
        )
        self.generator = generator
        self.eval_tokenizer = eval_tokenizer
        self.eval_model = eval_model
        self.data_loader = data_loader
        self.calc_corr = False
        self.max_corr_attr = [329, 145, 111, 199, 34, 44, 173, 386, 203, 67, 10, 273, 319, 206, 340, 385, 451, 307, 408, 236]

        if latent_ckpt is not None:
            state_dict = torch.load(latent_ckpt)
            self.ptb_model = BartClassificationHead(1024, 1024, 2, 0.1)
            BartClassificationHead.load_state_dict(self.ptb_model, state_dict)
            self.pool = pool
        else:
            self.ptb_model = None

    def perturb(
        self, data, ptb_param, ptb_len, 
        ptb_length: Union[int, float]= 0.5, 
        probing_l_step=1.0, 
    ):
        ptb_bz, *_ = ptb_param.shape
        bz, seq_len = data['input_ids'].shape
        eval_beams = int(ptb_bz / bz)

    
        def get_word_range(text, ptb_length, method='random'):
            text_len = len(text.split())
            if text.split()[-1] == '.': text_len -= 1
            if isinstance(ptb_length, float):
                ptb_length = int(text_len * ptb_length)

            if method == 'random':
                if ptb_length > text_len: return (0, text_len)
                else: 
                    rand_range = (0, text_len - ptb_length)
                    import random
                    start = random.randint(*rand_range)
                    return (start, start + ptb_length)

        def frac_to_seq_range(
            seq_len, ptb_range, 
            text=None, 
            tokenizer=None, 
            frac='text', 
        ):
            assert frac in ['encode', 'text']
            ptb_seq_range = []
            if frac == 'encode':
                map_to_seq_range = lambda r: tuple([int(seq_len * i) for i in r])
                for r in ptb_range:
                    ptb_seq_range.append(map_to_seq_range(r))
            elif frac == 'text':
                text_list = text.split()
                for r in ptb_range:
                    span_text = ' '.join(text_list[slice(*r)])
                    span_text = span_text if r[0] == 0 else f' {span_text}'
                    span_text_enc = tokenizer.encode(span_text)[1:-1] # remove start and end token
                    text_enc = tokenizer.encode(text)
                    ptb_seq_range.append(max_common_substring(text_enc, span_text_enc))

            return ptb_seq_range

        word_range_list = []
        if self.config.embed_type == 'unify':
            for b_idx in range(bz):
                text = data['src_texts'][b_idx]
                word_range = get_word_range(text, ptb_length)
                word_range_list.append(word_range)
                ptb_seq_range = frac_to_seq_range(-1, [word_range], text=text, tokenizer=self.eval_tokenizer)

                l = data['cls_labels'][b_idx].item()
                # binary cases
                if l == 0: 
                    ptb, bds, bd_check_fn = (ptb_len, -ptb_len), (torch.zeros, torch.ones), (torch.max, torch.min)
                else:
                    ptb, bds, bd_check_fn = (-ptb_len, ptb_len), (torch.ones, torch.zeros), (torch.min, torch.max)
                if self.config.ptb_mode in ['drop', 'raw']: # perturb without bound checking
                    for r in ptb_seq_range:
                        # ptb_param[eval_beams * b_idx:eval_beams * (b_idx+1), slice(*r), slice(*self.config.perturb_range)] -= ptb[1]
                        ptb_param[eval_beams * b_idx:eval_beams * (b_idx+1), slice(*r), self.max_corr_attr] -= ptb[1]
                        # cls_logits[b_idx, :, idx] = cls_logits[b_idx, :, idx] - ptb[i]
                #else: # softmax
                #    for i, idx in enumerate(range(*self.config.perturb_range)):
                #        for r in ptb_seq_range:
                #            cls_logits[b_idx, slice(*r), idx] = cls_logits[b_idx, slice(*r), idx] - ptb[i]
                #        tmp = torch.stack((cls_logits[b_idx, :, idx], bds[i](seq_len)), dim=-1)
                #        cls_logits[b_idx, :, idx] = bd_check_fn[i](tmp, dim=-1).values

        elif self.config.embed_type == 'probing':
            param_shape = ptb_param.shape
            g_noise = torch.randn(*param_shape).to(ptb_param.device) * probing_l_step
            ptb_param += g_noise

        return ptb_param, word_range_list

        
    def wandb_log(self, log_type: str, data: dict):
        if self.config.use_wandb: 
            if log_type == 'scatter':
                # init
                proj = 'perturb_eval' if self.config.output_dir is None else self.config.output_dir
                for l in range(0, 2):
                    run_name = '_'.join([f"{k[0]}={data[k]}" for k in ['ptb-size', 'cls-acc']] + [f"l={l}m={self.config.ptb_mode}"])
                    wandb.init(project=proj, name=run_name)
                    # log
                    log_data = data[f'label_{l}']
                    table = wandb.Table(data=log_data, columns=["label_0", "label_1"])
                    wandb.log({"label_feature_plot" : wandb.plot.scatter(
                        table=table, 
                        x="label_0", y="label_1", 
                        title="label feature"
                    )})
                    # finish
                    wandb.finish()

    def generate(self, data, ptb_len, mode="normal"):
        
        if mode == "normal":
            ptb_fn = lambda ptb_param: self.perturb(copy.deepcopy(data), ptb_param, ptb_len)
            out_sents = self.generator.generate(data, return_ids=False, perturb_fn=ptb_fn)
            batch_word_range = self.generator.batch_word_range
        else: # probing
            out_sents = probing()
            # todo: deal with batch_word_range
        return out_sents, batch_word_range

    def calc_max_corr_attr(self, ptb_corr, top_k=5):
        self.max_corr_attr = []
        for corr in ptb_corr:
            replace = -1
            min_idx = -1
            for i, attr in enumerate(self.max_corr_attr):
                if min_idx == -1 or ptb_corr[attr][1] < ptb_corr[self.max_corr_attr[min_idx]][1]:
                    min_idx = i
            if not min_idx == -1 and corr[1] > ptb_corr[self.max_corr_attr[min_idx]][1]:
                replace = min_idx
            if len(self.max_corr_attr) < top_k: 
                self.max_corr_attr.append(corr[0])
            elif not replace == -1:
                self.max_corr_attr[replace] = corr[0]

    def calc_corr_for_each_dim(self):
        proj = 'perturb_eval'
        #wandb.init(project=proj, name="ptb_corr")
        dim = self.generator.config.d_model
        #run_name = "ptb_dim={}"
        ptb_corr_avg, ptb_corr = [], []
        #table_corr = wandb.Table(columns=["dim", "corr"])
        #table_corr_avg = wandb.Table(columns=["corr_avg"])
        for d in range(dim):
            print(f"dim={d}")
            ptb_corr_for_dim = []
            #wandb.init(project=proj, name=run_name.format(d))
            self.generator.model.to("cuda")
            for i, data in enumerate(self.data_loader):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to("cuda")
                out = self.generator(**data, output_hidden_states=True)
                param = out.inputs_embeds if self.config.ptb_param == "embed" else out.encoder_last_hidden_state
                # calc correlation coef 
                sqz_fn = lambda x: x.squeeze()
                param_attr_list = list(map(sqz_fn, torch.split(param.detach()[:,:,d], 1)))
                cls_logit_attr_list = list(map(sqz_fn, torch.split(data["cls_logits"][:,:,1], 1)))
                for param_attr, cls_attr in zip(param_attr_list, cls_logit_attr_list):
                    attr = torch.stack((param_attr, cls_attr))
                    corr = torch.corrcoef(attr)[0, 1].item()
                    ptb_corr_for_dim.append([d, corr])

            #ptb_corr += ptb_corr_for_dim
            ptb_corr_avg.append([d, np.mean(np.array(ptb_corr_for_dim)[:, 1])])
            #for corr in ptb_corr_for_dim:
            #    table_corr.add_data(*corr)
            self.calc_max_corr_attr(ptb_corr_avg, top_k=20)
            corr_avg = np.mean(np.array(ptb_corr_for_dim)[:, 1])
            #table_corr_avg.add_data(corr_avg)
            with open("corr_avg_log_enc_out_cuda.txt", "w") as f:
                print(f"corr={corr_avg}, max_corr_attr={self.max_corr_attr}\n", f)
            """
            wandb.log({
                
                "corr/corr_per_data" : wandb.plot.scatter(
                    table=table_corr, 
                    x="dim", y="corr", 
                    title="corr per data"
                ), 
                "corr/corr_avg" : wandb.plot.histogram(
                    table_corr_avg, 
                    "corr_avg", 
                    title="average corr"
                ), 
            })
            """
            #wandb.finish()

        #wandb.finish()

    def set_device(self, data, device="cuda"):
        self.generator.model.to(device)
        self.eval_model.to(device)
        if self.ptb_model is not None: self.ptb_model.to(device)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

    def eval_loop(self, show_html=False):
        if self.calc_corr:
            self.calc_corr_for_each_dim()

        step = 0
        eval_batch_num = 10 # the control-loop counter
        while step < self.config.max_perturb_step:
            scatter_plot_info = {'label_0': [], 'label_1': []}
            total_num = 0
            text_list, out_list, word_range_list = [], [], []
            for i, data in enumerate(self.data_loader):
                if i >= eval_batch_num: break # loop controled by counter, may be removed. 
                total_num += data['input_ids'].shape[0]
                ptb_len = self.config.perturb_inc_rate * step
                self.set_device(data)
                out_sents, batch_word_range = self.generate(data, ptb_len)
                batch = self.generate_batch_data(out_sents, data['cls_labels'])
                self.set_device(batch)
                out = torch.nn.Softmax(dim=-1)(self.eval_model(**batch).logits)
                # src
                src_batch = self.generate_batch_data(data['src_texts'], data['cls_labels'])
                self.set_device(src_batch)
                src_out = torch.nn.Softmax(dim=-1)(self.eval_model(**src_batch).logits)
                #out = self.eval_model(**batch).logits # not to apply softmax
                # scatter info
                for l, dist in zip(data['cls_labels'].tolist(), out.tolist()):
                    scatter_plot_info[f'label_{l}'].append(dist)
                # html
                ## show wir
                seq_len = batch['input_ids'].shape[-1]
                self.set_device(batch, device="cpu")
                show_wir_in_html(
                    opts=['insert'], 
                    seq_len=seq_len, 
                    ptb_len=ptb_len, 
                    cls_batch=batch, 
                    cls_model=self.eval_model, 
                    cls_tokenizer=self.eval_tokenizer,   
                )
                ## show word diff
                if step == 0: 
                    num_to_one_hot_fn = lambda x: (1, 0) if x == 0 else (0, 1)
                    text_list += [[x, y] for x, y in zip(data['src_texts'], out_sents)]
                    """
                    out_list += [["({:.4f}, {:.4f}) -> ({:.4f}, {:.4f})".format(*num_to_one_hot_fn(l), *src_o), \
                                  "({:.4f}, {:.4f}) -> ({:.4f}, {:.4f})".format(*src_o, *o)] \
                                 for l, src_o, o in zip(data['cls_labels'].tolist(), src_out.tolist(), out.tolist())]
                    """
                    out_list += [[f"{colored_in_html(l, num_to_one_hot_fn(l))} -> {colored_in_html(l, src_o)}", f"{colored_in_html(l, o)}"] \
                                 for l, src_o, o in zip(data['cls_labels'].tolist(), src_out.tolist(), out.tolist())]
                else:
                    text_list += out_sents
                    #out_list += ["({:.4f}, {:.4f}) -> ({:.4f}, {:.4f})".format(*src_o, *o) for src_o, o in zip(src_out.tolist(), out.tolist())]
                    out_list += [f"{colored_in_html(l, o)}" for l, o in zip(data['cls_labels'].tolist(), out.tolist())]
                
                word_range_list += batch_word_range
        
            # html
            if len(np.array(text_list).shape) == 2: # init
                html_df = pd.DataFrame({k: np.array(text_list)[:,idx].tolist() \
                                        for k, idx in zip(['origin', f'ptb{step}={ptb_len}'], range(2))})
                html_out_df = pd.DataFrame({k: np.array(out_list)[:,idx].tolist() \
                                        for k, idx in zip(['origin', f'ptb{step}={ptb_len}'], range(2))})
                total_word_range_list = [word_range_list]
            else:
                html_df[f'ptb{step}={ptb_len}'] = text_list
                html_out_df[f'ptb{step}={ptb_len}'] = out_list
                total_word_range_list.append(word_range_list)

            # todo: plot
            """
            scatter_plot_info['ptb-size'] = ptb_len
            corr_num = 0
            if not total_num == sum([len(scatter_plot_info[x]) for x in ['label_0', 'label_1']]):
                print("warning: the total data recorded doesn't match the total num, the cls-acc value may be incorrect. ")
            for i, label in enumerate(['label_0', 'label_1']):
                pred = np.argmax(scatter_plot_info[label], axis=-1)
                corr_num += np.count_nonzero(pred == i)
            scatter_plot_info['cls-acc'] = corr_num / total_num
            self.wandb_log('scatter', scatter_plot_info)
            """
            #if step > 0 and step % 3 == 0:
            #    if show_html: show_texts_in_html(html_df, df_out=html_out_df, show_html=show_html, html_name=f'step={step}.html')
            #    html_df, html_out_df = html_df['origin'], html_out_df['origin']

            step += 1 # step inc

        if show_html: 
            os.makedirs('text', exist_ok=True)
            show_texts_in_html(
                html_df, ptb_len=ptb_len, 
                df_out=html_out_df, 
                word_range=total_word_range_list, 
                show_html=False, 
                html_name=f'text/p={self.config.ptb_param}.ln={self.config.perturb_inc_rate}.html'
            )
        # show wir html
        os.makedirs('wir', exist_ok=True)
        show_wir_in_html(
            opts=['write'], #, 'show'], 
            display_filename=f"wir/p={self.config.ptb_param}.ln={self.config.perturb_inc_rate}.html", 
        )


    def generate_batch_data(self, sents: List[str], labels: torch.tensor):
        labels = [str(l) for l in labels.tolist()]
        batch = [{"src_texts": src, "tgt_texts": tgt} for src, tgt in zip(sents, labels)]
        return self.collate_fn(batch)

    ''' 
        copy from dataset/ClassficationDataset 
        input batch should be list of dicts: 
        {"tgt_texts": tgt_line, "src_texts": source_line}
    '''
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        name_with_no_token_type_ids = [
            'textattack/roberta-base-rotten_tomatoes', 
            'textattack/distilbert-base-uncased-rotten-tomatoes', 
        ]

        source_inputs = [encode_line(self.eval_tokenizer, x["src_texts"], \
                        self.config.max_source_length, dataset_name=None) for x in batch]
        target_inputs = [label_to_tensor(x['tgt_texts']) for x in batch]

        source_ids = [x["input_ids"].squeeze() for x in source_inputs]
        src_mask = [x["attention_mask"].squeeze() for x in source_inputs]
        if self.eval_tokenizer.name_or_path in name_with_no_token_type_ids:
            src_token_type_ids = None
        else:
            src_token_type_ids = [x["token_type_ids"].squeeze() for x in source_inputs]
        target_ids = target_inputs
                    
        input_ids = torch.stack(source_ids)
        masks = torch.stack(src_mask)
        token_type_ids = None if src_token_type_ids is None else torch.stack(src_token_type_ids)
        target_ids = torch.stack(target_ids).squeeze().to(torch.long)
        pad_token_id = self.eval_tokenizer.pad_token_id
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

        # batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding