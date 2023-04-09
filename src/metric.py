from lib import *
from utils import pickle_save

# to calculate train/eval metric per epoch
class Metric():
    DEFAULT_METRIC = "accuracy"
    ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]

    def __init__(
        self, 
        model_mode=None, 
        val_metric_name=None, 
        metric_save_path=None, 
        use_hf_eval=False, 
    ):
        assert model_mode in ['seq2seq', 'cls', 'tgwv']
        metric_name = dict(seq2seq="rouge", cls="accuracy")
        self.model_mode = model_mode
        self.use_hf_eval = use_hf_eval
        self.metric_save_path = metric_save_path
        self.val_metric_name = self.DEFAULT_METRIC if val_metric_name is None else val_metric_name
        # store metric per epoch
        if use_hf_eval: self.metric = evaluate.load(metric_name[model_mode])
        self.metric_dicts, self.result_dicts = dict(train=[], val=[], test=[]), []

        # store metric per batch
        self.batch_metric_dicts=dict(train=[], val=[], test=[])
        self.batch_result_dicts = []

    def add_batch_metric(
        self, 
        metric_dict: dict, 
        type_path='train', 
        result_kwargs=None, 
        seq2seq_kwargs=None, 
    ):
        if seq2seq_kwargs is not None:
            rouge_dict = self.calculate_rouge(**seq2seq_kwargs)
            if self.use_hf_eval: self.metric.add_batch(**seq2seq_kwargs)
            metric_dict.update(rouge_dict)
            self.batch_result_dicts.append(seq2seq_kwargs)
        elif result_kwargs is not None:
            self.batch_result_dicts.append(result_kwargs)
        self.batch_metric_dicts[type_path].append(metric_dict)
        
    def calc_metric_mean(self, metric_dict: dict, type_path='train') -> dict:
        if len(metric_dict[type_path]) == 0: return dict()
        mean_metric_dict = defaultdict(list)
        for x in metric_dict[type_path]:
            for key, value in x.items():
                mean_metric_dict[key].append(value)
        return {f'{type_path}_avg_{key}': np.mean(value) for key, value in mean_metric_dict.items()}

    # metadata: can be train/eval batch size, step count, gen_time, gen_len, 
    def calc_metric_per_period(self, metadata: dict = {}, prefix='all'):
        assert prefix in ['train', 'val', 'test', 'all']
        prefix_list = ['train', 'val'] if prefix == 'all' else [prefix]
        metric_dict, result_dict = dict(), defaultdict(list)
        for p in prefix_list:
            metric_dict.update(self.calc_metric_mean(self.batch_metric_dicts, type_path=p))
            if self.use_hf_eval: 
                metric_kwargs = dict(use_stemmer=True) if self.model_mode in ['seq2seq', 'tgwv'] else dict()
                eval_metric = self.metric.compute(**metric_kwargs)
                eval_metric = {f'{prefix}_avg_{key}': value for key, value in eval_metric.items()}
                metric_dict.update(eval_metric)

            metric_dict.update(metadata)
            if len(metric_dict) > 0:
                self.metric_dicts[p].append(metric_dict)
            # reset
            self.batch_metric_dicts[p] = []

        if 'val' in prefix_list:
            for result in self.batch_result_dicts:
                for key, value in result.items():
                    result_dict[key] += value

            if len(result_dict) > 0:
                self.result_dicts.append(result_dict)
            # reset
            self.batch_result_dicts = []

        
    def calculate_rouge(
        self, 
        predictions: List[str] = None, 
        references: List[str] = None, 
        rouge_keys: List[str] = None, 
        use_stemmer=True
    ) -> Dict:
        if rouge_keys is None: rouge_keys = self.ROUGE_KEYS
        scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
        aggregator = scoring.BootstrapAggregator()

        for reference_ln, output_ln in zip(references, predictions):
            scores = scorer.score(reference_ln, output_ln)
            aggregator.add_scores(scores)

        result = aggregator.aggregate()
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def calculate_rouge_by_evaluate(self, predictions: List[str], references: List[str]):
        self.metric.add_batch(predictions=predictions, references=references)

    def save_metric(self):
        json.dumps(self.metric_dicts, self.metric_save_path)