import numpy as np
from collections import defaultdict
from textattack import Attack
from textattack.attack_recipes.attack_recipe import AttackRecipe
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.shared import utils

from lm_eval.attack.goal_function import get_llm_goal_fn
from lm_eval.attack.attacked_text import LLMAttackedText

def build_attack(method, model_wrapper, attack_keys, attack_info):
    if method == "bert-attack":
        attack = BERTAttackLi2020.build(model_wrapper, attack_keys=attack_keys, attack_info=attack_info)
    return attack

class BERTAttackLi2020(AttackRecipe):
    @staticmethod
    def build(model_wrapper, attack_keys=[], attack_info={}):
        from textattack.constraints.overlap import MaxWordsPerturbed
        from textattack.constraints.pre_transformation import (
            RepeatModification,
            StopwordModification,
        )
        from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
        from textattack.goal_functions import UntargetedClassification
        from textattack.search_methods import GreedyWordSwapWIR
        from textattack.transformations import WordSwapMaskedLM
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxWordsPerturbed(max_percent=0.4))

        use_constraint = UniversalSentenceEncoder(
            threshold=0.2,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        constraints.append(use_constraint)
        goal_function = get_llm_goal_fn(model_wrapper, goal_type="untargeted")
        #goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return LLMAttack(goal_function, constraints, transformation, search_method, 
                         attack_keys=attack_keys, attack_info=attack_info, 
            )


class LLMAttack(Attack):
    def __init__(self, *args, attack_keys=[], attack_info={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_keys = attack_keys
        self.attack_info = attack_info

    def _attack(self, initial_result):
        """Calls the ``SearchMethod`` to perturb the ``AttackedText`` stored in
        ``initial_result``.

        Args:
            initial_result: The initial ``GoalFunctionResult`` from which to perturb.

        Returns:
            A ``SuccessfulAttackResult``, ``FailedAttackResult``,
                or ``MaximizedAttackResult``.
        """
        final_result = self.search_method(initial_result)
        # reset adv sample logits
        _text_total = final_result.attacked_text._text_total
        task, doc_id = _text_total["task"], int(_text_total["doc_id"])
        self.goal_function.process_res_queue[(task, doc_id)] = [(i, x.item()) for i, x in enumerate(final_result.final_logits)]
        self.goal_function.adv_samples[(task, doc_id)] = final_result.attacked_text._text_query

        self.clear_cache()
        if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            result = SuccessfulAttackResult(
                initial_result,
                final_result,
            )
        elif final_result.goal_status == GoalFunctionResultStatus.SEARCHING:
            result = FailedAttackResult(
                initial_result,
                final_result,
            )
        elif final_result.goal_status == GoalFunctionResultStatus.MAXIMIZING:
            result = MaximizedAttackResult(
                initial_result,
                final_result,
            )
        else:
            raise ValueError(f"Unrecognized goal status {final_result.goal_status}")
        return result

    def attack(self, example, ground_truth_output):
        from textattack.attack_results import (
            FailedAttackResult,
            MaximizedAttackResult,
            SkippedAttackResult,
            SuccessfulAttackResult,
        )
        from textattack.constraints import Constraint, PreTransformationConstraint
        from textattack.goal_function_results import GoalFunctionResultStatus
        from textattack.goal_functions import GoalFunction
        from textattack.models.wrappers import ModelWrapper
        from textattack.search_methods import SearchMethod
        from textattack.shared import AttackedText, utils
        from textattack.transformations import CompositeTransformation, Transformation
        from collections import OrderedDict

        assert isinstance(
            example, (str, OrderedDict, AttackedText)
        ), "`example` must either be `str`, `collections.OrderedDict`, `textattack.shared.AttackedText`."
        if isinstance(example, (str, OrderedDict)):
            example = LLMAttackedText(example, attack_keys=self.attack_keys)
        elif isinstance(example, AttackedText):
            if self.attack_info["mod_type"] == "choice_w/o_gt":
                attack_keys = []
                for k in example._text_input.keys():
                    if k.startswith("continuation") and k.split("_")[-1] != str(ground_truth_output):
                        attack_keys.append(k)
            elif self.attack_info["mod_type"] == "tgt_question":
                attack_keys = ["question_25"]
            else: 
                attack_keys = self.attack_keys 
            print("attack keys: ", attack_keys)

            example = LLMAttackedText(example._text_input, attack_keys=attack_keys)
        
        assert isinstance(
            ground_truth_output, (int, str)
        ), "`ground_truth_output` must either be `str` or `int`."
        goal_function_result, _ = self.goal_function.init_attack_example(
            example, ground_truth_output
        )
        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            result = self._attack(goal_function_result)
            return result

    def get_transformations(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        if not self.transformation:
            raise RuntimeError(
                "Cannot call `get_transformations` without a transformation."
            )

        if self.use_transformation_cache:
            cache_key = tuple([current_text] + sorted(kwargs.items()))
            if utils.hashable(cache_key) and cache_key in self.transformation_cache:
                # promote transformed_text to the top of the LRU cache
                self.transformation_cache[cache_key] = self.transformation_cache[
                    cache_key
                ]
                transformed_texts = list(self.transformation_cache[cache_key])
            else:
                transformed_texts = self._get_transformations_uncached(
                    current_text, original_text, **kwargs
                )
                if utils.hashable(cache_key):
                    self.transformation_cache[cache_key] = tuple(transformed_texts)
        else:
            transformed_texts = self._get_transformations_uncached(
                current_text, original_text, **kwargs
            )

        return self.filter_transformations(
            transformed_texts, current_text, original_text
        )

    def get_indices_to_order(self, current_text, **kwargs):
        """Applies ``pre_transformation_constraints`` to ``text`` to get all
        the indices that can be used to search and order.

        Args:
            current_text: The current ``AttackedText`` for which we need to find indices are eligible to be ordered.
        Returns:
            The length and the filtered list of indices which search methods can use to search/order.
        """
        indices_to_order = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            return_indices=True,
            **kwargs,
        )

        len_text = len(indices_to_order)

        # Convert indices_to_order to list for easier shuffling later
        return len_text, list(indices_to_order)