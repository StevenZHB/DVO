"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re
import json

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm.outputs import RequestOutput
import numpy as np

from mcts_math.nodes.base_node import BaseNode
from mcts_math.constants import (
    NO_VALID_CHILD,
    TOO_MANY_STEPS,
    TOO_MANY_CODE_ERRORS,
    SOLUTION_COLOR,
    OBSERVATION_COLOR,
)
from .tree import BaseTree, code_execution
from .step_tree_beam import STEP_BEAM_TREE
from .step_tree import STEP_TREE

from mcts_math.tools.python_parse_and_execution import parse_code, safe_execute


class STEP_BEAM_SEARCH_TREE(STEP_TREE):
    """
    Step-level Beam Search
    """

    current_top_num: int = 1
    current_logprob: float = 0
    current_nodes: List[Type[BaseNode]] = []
    final_answer_nodes: List[Type[BaseNode]] = []
    candidate_nodes: List[Type[BaseNode]] = []


    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.config.use_python_interpreter and self.config.initial_solution:
            leaf_node = self.init_code_solution()

        self.candidate_nodes.append(self.current_node)
        self.current_top_num = self.config.step_beam_width
        self.current_nodes.append(self.root)
        # self.select_next_step()


    def init_code_solution(self):
        pass


    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "sbs":
            raise ValueError(f"Wrong value for config mode, must be step_beam_tree")
        if not cfg.n_generate_sample >= 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be greater than 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg


    def create_llm(self) -> Callable[..., List[str]]:
        # we only implement the batch inference
        pass


    def is_ignored_node(self, node: Type[BaseNode]) -> bool:
        return node.is_terminal or node.depth > self.config.max_depth


    def should_generate_next(self) -> bool:
        need_generate = False
        if self.current_top_num < 1:
            return False
        for step_node in self.candidate_nodes:
            if not self.is_ignored_node(step_node):
                need_generate = True
                break
        return need_generate


    def create_prompt(
        self,
        is_value_only: bool = False,
    ) -> str:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        # current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        current_nodes = self.candidate_nodes
        for current_node in current_nodes:
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            partial_solution = self.collect_partial_solution(current_node)
            prompt = self.prompt_wrap(
                self.question,
                partial_solution
            )
            prompts.append(prompt)
        return prompts

    @staticmethod
    def is_valid_final_answer_node(node: Type[BaseNode]) -> bool:
        # by default, final_anwer = ""
        if node.is_terminal and node.state["final_answer"] and \
           node.state["final_answer"] != TOO_MANY_STEPS:
            return True
        return False

    def select_next_step(self, outputs: Optional[List[RequestOutput]] = None) -> None:
        """process output from vllm
        e.g.,
        prompts = tree.create_prompt(is_value_only=True)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                # assert self.question in output.prompt
                candidate_node.value = output.value_estimate if output.value_estimate is not None else -100

        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.current_nodes = self.candidate_nodes[:self.current_top_num]

        for current_node in self.current_nodes[:]:  # must shallow copy because of the remove in the loop
            if self.__class__.is_valid_final_answer_node(current_node):
                self.final_answer_nodes.append(current_node)
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
            elif current_node.is_terminal or current_node.depth > self.config.max_depth:
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1

    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        current_nodes = self.candidate_nodes
        self.candidate_nodes = []
        for current_node, output in zip(current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            self.current_node = current_node
            choices = output.choices if self.config.model_type=="server" else output.outputs
            if choices[0].value_estimate is not None:
                self.expand_node(choices, current_node)
                # self.candidate_nodes.extend(current_node.children)
            else:
                value_estimate = -100
                current_node.is_terminal = True
        # self.update_candidate_node()


    def expand_node(self, outputs, node):
        if self.config.remove_duplicate:
            dedup_outputs = []
            dedup_keys = set()
            for output in outputs:
                key = output.text.strip()
                if not key in dedup_keys:
                    dedup_keys.add(key)
                    dedup_outputs.append(output)
            outputs = dedup_outputs
        for idx, output in enumerate(outputs):
            token_num = 0
            if self.config.model_type == "server":
                token_num = len(output.logprobs.tokens)
            else:
                token_num = len(output.token_ids)
            prior_prob = output.q_pi / token_num

            output_tokens = self.add_step_delim(output)
            step_result, parser_result = self.step_unwrap(output_tokens, self.config.use_python_interpreter)

            # If the output ends with eos token and no parser result produced, yield NO_VALID_CHILD
            if parser_result['final_answer'] == "" and "FINISH_MATCHED_STR:" in output.finish_reason and self.config.stop[0] in output.finish_reason:
                parser_result['final_answer'] = NO_VALID_CHILD

            if output.value_estimate != -100:
                # output.value_estimate = -output.value_estimate # beam search 中，policy 和 ref 是反过来的
                value_estimate = 1 / (1 + np.exp(-output.value_estimate))  # sigmoid the value estimation to scale into a 0-1 scalar
            else:
                value_estimate = -100

            self.create_child(step_result, parser_result, node, prior_prob, idx, value_estimate, token_num)



    def create_child(
        self,
        step_result: str,
        parser_result: Dict[str, str],
        node: Type[BaseNode],
        prior_prob: float,
        idx: int,
        value_estimate: int=-100,
        token_num: int=0,
    ) -> None:
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))

        # initialize a new node
        new_node = self.create_node(parent=node)
        new_node.tag = f"{node.tag}.{idx}"
        new_node.depth = node.depth + 1
        # new_node.prior = prior_prob # Don't need prior in beam search
        new_node.value = value_estimate

        # update node state
        if parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            new_node.state["length"] = token_num
            if "<code>" in parser_result["final_answer"]:
                # TODO:
                # 这里需要把代码匹配出来，把代码放进 new_node.state["final_answer"]
                # 在 eval_final_answer 中，执行这个代码
                _node_trajectory = self.collect_partial_solution(new_node)
                # execution_result = parse_and_execute(_node_trajectory, parser_result["final_answer"])
                executable_code = parse_code(_node_trajectory, with_tag=True)
                new_node.state["final_answer"] = executable_code
                # input(f"parse_code: \n{executable_code}")
                # TODO END
            self.current_top_num -= 1
            # self.eval_final_answer(new_node)
        else:
            new_node.state["text"] = step_result
            new_node.state["length"] = token_num
            self.candidate_nodes.append(new_node)

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            # self.eval_final_answer(new_node)

        node.children.append(new_node)


    def eval_final_answer(self, node: Type[BaseNode]) -> None:
        # if node.state["final_answer"] == TOO_MANY_STEPS:
        #     node.update_recursive(self.config.negative_reward, self.root)
        #     return
        # * First, check the final answer:
        #   * If it's a regular answer (without <code>), it's either CoT or logic, so just match directly.
        #   * If it contains <code>, it needs to be executed:
        #       * During execution, MathPot's test cases have empty input, while others have normal inputs passed together.
        #       * In MathPoT's case, we have a test case where the input is empty, and the ground_truth is used as the output.
        # * Final answer format -> {"input": [""], "output": [Final answer]} (as a dict, not a JSON string)
        if "```python" not in node.state["final_answer"]:
            # CoT, Logic
            if self.ground_truth:
                final_answer = node.state["final_answer"]
                if "Math" in self.unique_id or "MATH" in self.unique_id or "GSM8K" in self.unique_id:
                    correct = is_equiv(self.ground_truth, final_answer)
                else:
                    correct = self.ground_truth in final_answer
                # backup
                reward = self.config.positive_reward if correct else self.config.negative_reward
                reward += node.get_all_prior()
                node.update_recursive(reward, self.root)
            else:
                # for testset, no ground_truth, put this node in candidate_nodes, then it will be evaluated by value model and backup in select_next_step().
                self.candidate_nodes.append(node)
        else:
            # Generated answer is the code.
            # 是pot还是coding通过测试样例来判断
            generated_code = node.state["final_answer"]
            if not self.ground_truth:
                self.candidate_nodes.append(node)
                return
            if type(self.ground_truth) == str:
                try:
                    self.ground_truth = json.loads(self.ground_truth)
                except:
                    self.ground_truth = {"input": [], "output": []}
            try:
                test_inputs = self.ground_truth.get("input", [])
                test_outputs = self.ground_truth.get('output', [])
            except Exception as e:
                print("Error: ", e)
                return
            test_cases = list(zip(test_inputs, test_outputs))
            correct = False
            for case in test_cases:
                # input(f"case: {case}")
                execute_result = safe_execute(generated_code, case[0])
                # input(f"execute_result: {execute_result}")
                # input(f"golden: {case[1]}")
                correct = is_equiv(case[1], execute_result)
            # input(f"correct: {correct}")
            reward = self.config.positive_reward if correct else self.config.negative_reward
            node.update_recursive(reward, self.root)



    def add_step_delim(self, output):
        if 'FINISH_MATCHED_STR: ' in output.finish_reason:
            output_text_w_stop_token = output.text+output.finish_reason.split("FINISH_MATCHED_STR: ")[-1].replace(self.config.stop[0], '')
            return output_text_w_stop_token
        # output_text_w_stop_token = ''.join(output.logprobs.tokens).replace(self.config.stop[0], '')
        else:
            return output.text

    def get_steps(self):
        final_answer_states = []
        for cur_node in self.final_answer_nodes:
            states = {
                "question": self.question,
                "ground_truth": self.ground_truth,
                "value": cur_node.value,
                "final_answer": cur_node.state["final_answer"],
                "solution": self.collect_partial_solution(cur_node),
                "tag": cur_node.tag,
            }
            final_answer_states.append(states)

        solutions = sorted(final_answer_states, key=lambda x: x['value'], reverse=True)
        return solutions

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            if node.has_children():
                candidates.extend(node.children)
        states["solutions"] = self.get_steps()
        return states


    def update_candidate_node(self):
        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.candidate_nodes = self.candidate_nodes[:self.current_top_num] if len(self.candidate_nodes) >= self.current_top_num else self.candidate_nodes
        return