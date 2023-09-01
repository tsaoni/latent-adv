import torch
import numpy as np
from collections import defaultdict, OrderedDict
from textattack.goal_functions import (
    GoalFunction, 
    TargetedClassification, 
    UntargetedClassification, 
)
from textattack.shared.attacked_text import AttackedText

goal_function_type = {
    "targeted": TargetedClassification, 
    "untargeted": UntargetedClassification, 
}

def get_llm_goal_fn(*goal_fn_args, goal_type="targeted", **goal_fn_kwargs):
    goal_class = goal_function_type[goal_type]

    class LLMGoalFunction(goal_class):
        def __init__(self, *args, separator=AttackedText.SPLIT_TOKEN, **kwargs):
            super().__init__(*args, **kwargs)
            self.separator = separator
            self.process_res_queue = defaultdict(list)
            self.adv_samples = defaultdict(OrderedDict)

        def get_results(self, attacked_text_list, check_skip=False):
            """For each attacked_text object in attacked_text_list, returns a
            result consisting of whether or not the goal has been achieved, the
            output for display purposes, and a score.

            Additionally returns whether the search is over due to the query
            budget.
            """
            results = []
            if self.query_budget < float("inf"):
                queries_left = self.query_budget - self.num_queries
                attacked_text_list = attacked_text_list[:queries_left]
            self.num_queries += len(attacked_text_list)
            model_outputs = self._call_model(attacked_text_list)
           
            for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
                probs, logits = raw_output[0], raw_output[1]
                displayed_output = self._get_displayed_output(probs)
                goal_status = self._get_goal_status(
                    probs, attacked_text, check_skip=check_skip
                )
                goal_function_score = self._get_score(probs, attacked_text)
                result = self._goal_function_result_type()(
                    attacked_text,
                    probs,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
                result.final_logits = logits
                results.append(result)
            return results, self.num_queries == self.query_budget

        def _call_model_uncached(self, attacked_text_list):
            """Queries model and returns outputs for a list of AttackedText
            objects."""
            if not len(attacked_text_list):
                return []

            def context_format(qa_pair):
                q_title, a_title = "Question:", "Answer:"
                context = ""
                for i, x in enumerate(qa_pair):
                    if len(context) == 0:
                        context += f"{q_title} {x}"
                    elif i % 2 == 0:
                        context += f"\n\n{q_title} {x}"
                    else:
                        context += f"\n{a_title} {x}"
                context += f"\n{a_title}"
                return context
            
            inputs, outputs = [], []
            for at in attacked_text_list:
                at = at.get_query
                qa_pair = at.tokenizer_input #.split(self.separator)
                if len(qa_pair) > 51: 
                    n_choices = len(qa_pair) - 51 # 25 shots
                    continuations, context = qa_pair[0:n_choices], context_format(qa_pair[n_choices:])
                else:
                    n_choices = len(qa_pair) - 1
                    continuations, context = qa_pair[0:n_choices], qa_pair[n_choices:]
                inputs.append([(context, continuation) for continuation in continuations])
            i = 0
            while i < len(inputs):
                batch = inputs[i : i + self.batch_size]
                batch_preds = self.model(batch)

                # Some seq-to-seq models will return a single string as a prediction
                # for a single-string list. Wrap these in a list.
                if isinstance(batch_preds, str):
                    batch_preds = [batch_preds]

                # Get PyTorch tensors off of other devices.
                if isinstance(batch_preds, torch.Tensor):
                    batch_preds = batch_preds.cpu()

                if isinstance(batch_preds, list):
                    outputs.extend(batch_preds)
                elif isinstance(batch_preds, np.ndarray):
                    outputs.append(torch.tensor(batch_preds))
                else:
                    outputs.append(batch_preds)
                i += self.batch_size

            if isinstance(outputs[0], torch.Tensor):
                outputs = torch.cat(outputs, dim=0)

            assert len(inputs) == len(
                outputs
            ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

            for at, out in zip(attacked_text_list, outputs):
                task, doc_id = at._text_total["task"], int(at._text_total["doc_id"])
                self.process_res_queue[(task, doc_id)] = [(i, x.item()) for i, x in enumerate(out)]
                self.adv_samples[(task, doc_id)] = at._text_query

            return torch.stack((self._process_model_outputs(attacked_text_list, outputs), outputs), dim=1)
        
        def _call_model(self, attacked_text_list):
            """Gets predictions for a list of ``AttackedText`` objects.

            Gets prediction from cache if possible. If prediction is not in
            the cache, queries model and stores prediction in cache.
            """
            if not self.use_cache:
                return self._call_model_uncached(attacked_text_list)
            else:
                uncached_list = []
                for text in attacked_text_list:
                    if text in self._call_model_cache:
                        # Re-write value in cache. This moves the key to the top of the
                        # LRU cache and prevents the unlikely event that the text
                        # is overwritten when we store the inputs from `uncached_list`.
                        self._call_model_cache[text] = self._call_model_cache[text]
                    else:
                        uncached_list.append(text)
                uncached_list = [
                    text
                    for text in attacked_text_list
                    if text not in self._call_model_cache
                ]
                outputs = self._call_model_uncached(uncached_list)
                for text, output in zip(uncached_list, outputs):
                    self._call_model_cache[text] = output
                all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
                return all_outputs

    goal_fn = LLMGoalFunction(*goal_fn_args, **goal_fn_kwargs)
    return goal_fn
        