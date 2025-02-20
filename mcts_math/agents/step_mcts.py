"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import random
import torch
import numpy as np

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Callable, Union
# from functools import partial
from pydantic import field_validator
import json

from vllm.outputs import CompletionOutput, RequestOutput

from mcts_math.agents.utils import math_is_equiv_no_timeout as is_equiv
import math
from mcts_math.nodes import MCTSNode
from mcts_math.constants import (
    TOO_MANY_CODE_ERRORS,
    TOO_MANY_STEPS,
    NO_VALID_CHILD,
    SOLUTION_COLOR,
    OBSERVATION_COLOR,
    WARNING_COLOR,
)
from mcts_math.tools.python_parse_and_execution import parse_code, safe_execute

from .tree import BaseTree, code_execution
from .step_tree_beam import STEP_BEAM_TREE
import re


def refine_code(response: str):
    """Refine the code.

    Repair
    ```
    Step x: from xxxx import xxx
    ```
    to
    ```
    Step x:
    from xxxx import xxx
    ```

    Args:
        response (str): _description_

    Returns:
        _type_: _description_
    """
    pattern = r"(Step \d+: )from[\w\s,]*import"
    replacement = r"\1\nfrom"
    result = re.sub(pattern, replacement, response)
    return result



class STEP_MCTS(STEP_BEAM_TREE):

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "step_mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def init_code_solution(self):
        for solution in self.solutions:
            if "```python" not in solution:
                break
            solution_steps = solution.split('Step')
            if solution_steps[0] == "":
                solution_steps = solution_steps[1:]
            solution_steps[0] = "Step" + solution_steps[0]
            current_node = self.root
            for step_idx in range(len(solution_steps)):

                public_parent_flag = False
                for child in current_node.children:
                    if child.state['text'] == solution_steps[step_idx]:
                        current_node = child
                        public_parent_flag = True
                        break
                if public_parent_flag:
                    continue

                new_node = self.create_node(parent=current_node)
                new_node.tag = f"{current_node.tag}.-1"
                new_node.depth = current_node.depth + 1
                new_node.prior = 0
                new_node.value = -100
                if step_idx != len(solution_steps)-1:
                    new_node.state['text'] = solution_steps[step_idx] + "Step" if "# " in solution_steps[step_idx] else solution_steps[step_idx] + "\n Step"
                else:
                    new_node.state['text'] = solution_steps[step_idx]
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = solution_steps[step_idx]
                    new_node.update_recursive(self.config.positive_reward, self.root)
                current_node.children.append(new_node)
                current_node = new_node
        return current_node


    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent,
            additional_state_keys=self.STEP_NODE_KEYS,
            c_puct=self.config.c_puct,
        )

    def generate(self) -> None:
        self.search()

    @torch.inference_mode()
    def search(self) -> None:
        for idx in range(self.config.iterations):
            # node selection starting from root
            node = self.selection()
            # expansion_evaluation_backpropagation
            self.expansion_evaluation_backpropagation(node)

    def selection(self) -> Optional[Type[MCTSNode]]:
        node = self.root
        while node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True
                if node.parent is not None:
                    next_node = node.parent
                else:
                    break
            node = next_node

        return None if node.is_terminal else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        # TO_DO: implement multi-strategy
        # select the best child according to the puct
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue

            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        return random.choice(best_childs) if best_childs else None

    def expansion_evaluation_backpropagation(self, node: Type[MCTSNode]) -> None:
        prompt = self.create_prompt()
        # expand and evaluate
        outputs, value_estimate = self.llm(prompt, n=self.n_generate_sample, stop=self.stop)
        if value_estimate is not None:  # input exceeds 4096, output '' and None
            self.expand_node(outputs, node)
        else:
            value_estimate = self.config.negative_reward
            node.is_terminal = True
        # backup
        node.update_recursive(value_estimate, self.root)


    def add_step_delim(self, output):
        if output.matched_stop is not None:
            output_text_w_stop_token = output.text+output.matched_stop.replace(self.config.stop[0], '')
            return output_text_w_stop_token
        # output_text_w_stop_token = ''.join(output.logprobs.tokens).replace(self.config.stop[0], '')
        else:
            return output.text

    def add_step_delim_old(self, output):
        output_text = output.text.strip()
        if output_text.endswith("."):
            final_token = "."
        elif output_text.endswith("python"):
            final_token = "python"
        else:
            output_text_tokens = output_text.split()
            final_token = output_text_tokens[-1]
        current_stop_tokens = []
        i = 0
        _reversed_output_text_tokens = list(reversed(output.logprobs.tokens.copy()))
        while i < len(_reversed_output_text_tokens):
            if _reversed_output_text_tokens[i] == final_token:
                break
            current_stop_tokens.append(_reversed_output_text_tokens[i])
            i += 1
        if len(current_stop_tokens) > 3:
            current_stop_tokens = output.logprobs.tokens[-1:-3:-1]
        # if current_stop_tokens[-2] == "":
        #     current_stop_tokens[-2] = " " # in Eurus tokenizer, "# Step" will be tokenized to ["#", "", "Step"]
        _stop_token_delim = ""
        for i in range(len(current_stop_tokens)):
            if current_stop_tokens[i] == "#" or current_stop_tokens[i] == ".":
                current_stop_tokens[i] += ' '
        stop_token_str = _stop_token_delim.join(list(reversed(current_stop_tokens)))

        return output_text + stop_token_str + ' '

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
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
            if self.config.model_type=="server":
                token_num = len(output.logprobs.tokens)
            else:
                token_num = len(output.token_ids)
            prior_prob = output.q_pi / token_num

            output_tokens = self.add_step_delim(output)
            step_result, parser_result = self.step_unwrap(output_tokens, self.config.use_python_interpreter)

            # If the output ends with eos token and no parser result produced, yield NO_VALID_CHILD
            if parser_result['final_answer'] == "" and "stop" in output.finish_reason and self.config.stop[0] in output.matched_stop:
                parser_result['final_answer'] = NO_VALID_CHILD
            
            if parser_result['final_answer'] == "" and "length" in output.finish_reason:
                parser_result['final_answer'] = NO_VALID_CHILD

            if output.value_estimate != -100:
                value_estimate = 1 / (1 + np.exp(-output.value_estimate))  # sigmoid the value estimation to scale into a 0-1 scalar
            else:
                value_estimate = -100
            # self.create_child(step_result, parser_result, node, prior_prob, idx)
            self.create_child(step_result, parser_result, node, prior_prob, idx, value_estimate, token_num)

    def create_child(
        self,
        step_result: str,
        parser_result: Dict[str, str],
        node: Type[MCTSNode],
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
        new_node.prior = prior_prob

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
            self.eval_final_answer(new_node)
        else:
            new_node.state["text"] = step_result
            new_node.state["length"] = token_num

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] == TOO_MANY_STEPS:
            node.update_recursive(self.config.negative_reward, self.root)
            return
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
                # reward += node.get_all_prior()
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
                # backup
                if candidate_node.is_terminal and self.ground_truth:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True
                if self.config.update_on_step:
                    candidate_node.update_recursive(value_estimate, self.root)
                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection()
        if selection_node is not None:
            self.current_nodes.append(selection_node)

    def evaluate_and_select_next_step(self) -> None:
        self.current_nodes = []
        selection_node = self.selection()
        if selection_node is not None:
            self.current_nodes.append(selection_node)

    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            if self.config.verbose:
                print(colored(f'value:{output.value_estimate}','green'))
            # value_estimate = output.value_estimate
            # if value_estimate is not None:
            choices = output.choices if self.config.model_type=="server" else output.outputs
            if choices[0].value_estimate is not None:  # input exceeds 4096, output '' and None
                self.expand_node(choices, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True
                # current_node.update_recursive(value_estimate, self.root)
            # self.expand_node(output.outputs, current_node)
            # self.candidate_nodes.extend(current_node.children)

            # backup
            if self.config.update_on_step:
                if self.config.update_leaf_value:
                    # child node will be put into candidate_nodes, then all candidate_nodes will be evaluated by value model and backup in select_next_step().
                    for value_node in current_node.children:
                        if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                            self.candidate_nodes.append(value_node)
                else:
                    current_node.update_recursive(value_estimate, self.root)

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["prior"] = node.prior
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states
