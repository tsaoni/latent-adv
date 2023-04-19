from lib import *
from utils import *

'''
latent-adv-exp1
../models/t=tgwv_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls_cls_m=Hazqeel.electra-small-finetuned-sst2-rotten_tomatoes-distilled
'''

class PerturbEval():
    def __init__(
        self, generator, eval_tokenizer, eval_model, data_loader, 
        embed_type='unify', 
        perturb_range=(0, 2), 
        use_wandb=False, 
        output_dir=None, 
        max_source_length=10, 
        ptb_mode='softmax', 
    ):
        self.config = argparse.Namespace(
            similarity_fn=None, 
            perturb_inc_rate=0.01, 
            max_perturb_step=10, 
            embed_type=embed_type, # unify, drop, 
            perturb_range=perturb_range, 
            use_wandb=use_wandb, #use_wandb, 
            output_dir=None, # for wandb proj name, beware not to exceed 128 chars
            max_source_length=max_source_length, 
            ptb_mode=ptb_mode, 
        )
        self.generator = generator
        self.eval_tokenizer = eval_tokenizer
        self.eval_model = eval_model
        self.data_loader = data_loader

    # eval metric: edit distance, cos sim, cel
    def calc_similarity_with_perturb_rate_inc(self):
        # input: (ori_sent, pred_sent), func: eval_model or other utility funcs
        pass

    def plot_scatter_result(self, samples: List[tuple]):
        # list of samples with info: (label, logit1, logit2)
        pass

    def perturb(self, data, ptb_len):
        cls_logits = data['cls_logits']
        bz, seq_len, _ = cls_logits.shape
        if self.config.embed_type == 'unify':
            for b_idx in range(bz):
                l = data['cls_labels'][b_idx].item()
                # binary cases
                if l == 0: 
                    ptb, bds, bd_check_fn = (ptb_len, -ptb_len), (torch.zeros, torch.ones), (torch.max, torch.min)
                else:
                    ptb, bds, bd_check_fn = (-ptb_len, ptb_len), (torch.ones, torch.zeros), (torch.min, torch.max)
                if self.config.ptb_mode in ['drop', 'raw']: # perturb without bound checking
                    for i, idx in enumerate(range(*self.config.perturb_range)):
                        cls_logits[b_idx, :, idx] = cls_logits[b_idx, :, idx] - ptb[i]
                else: # softmax
                    for i, idx in enumerate(range(*self.config.perturb_range)):
                        tmp = torch.stack((cls_logits[b_idx, :, idx] - ptb[i], bds[i](seq_len)), dim=-1)
                        cls_logits[b_idx, :, idx] = bd_check_fn[i](tmp, dim=-1).values
                
        return data
        
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
        """
        else:
            import matplotlib.pyplot as plt
            for l in range(0, 2):
                plt_data = data[f'label_{l}']
                x = [d[0] for d in plt_data]
                y = [d[1] for d in plt_data]
                label_name = '_'.join([f"l={l}"])
                plt.scatter(x, y, label=label_name)
            title = '_'.join([f"{k[0]}={data[k]}" for k in ['ptb-size', 'cls-acc']] + [f"m={self.config.ptb_mode}"])
            plt.title(title)
            plt.legend()
            plt.savefig('../img/'+title+'.png', dpi=300, bbox_inches='tight')
        """

    def eval_loop(self):
        step = 0
        eval_batch_num = 10
        while step < self.config.max_perturb_step:
            scatter_plot_info = {'label_0': [], 'label_1': []}
            total_num = 0
            for i, data in enumerate(self.data_loader):
                if i >= eval_batch_num: break
                total_num += data['input_ids'].shape[0]
                ptb_len = self.config.perturb_inc_rate * step
                perturbed_data = self.perturb(copy.deepcopy(data), ptb_len)
                out_sents = self.generator.generate(perturbed_data, return_ids=False)
                batch = self.generate_batch_data(out_sents, data['cls_labels'])
                #out = torch.nn.Softmax(dim=-1)(self.eval_model(**batch).logits)
                out = self.eval_model(**batch).logits # not to apply softmax
                # scatter info
                for l, dist in zip(data['cls_labels'].tolist(), out.tolist()):
                    scatter_plot_info[f'label_{l}'].append(dist)

            scatter_plot_info['ptb-size'] = ptb_len
            corr_num = 0
            if not total_num == sum([len(scatter_plot_info[x]) for x in ['label_0', 'label_1']]):
                print("warning: the total data recorded doesn't match the total num, the cls-acc value may be incorrect. ")
            for i, label in enumerate(['label_0', 'label_1']):
                pred = np.argmax(scatter_plot_info[label], axis=-1)
                corr_num += np.count_nonzero(pred == i)
            scatter_plot_info['cls-acc'] = corr_num / total_num
            self.wandb_log('scatter', scatter_plot_info)

            step += 1

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