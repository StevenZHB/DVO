"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm import LLM, SamplingParams

from mcts_math.llms.local_llms import local_vllm
from mcts_math.nodes.base_node import BaseNode
from mcts_math.constants import (
    NO_VALID_CHILD,
    SOLUTION_COLOR,
    OBSERVATION_COLOR,
    WARNING_COLOR,
    FINAL_ANSWER_ACTION,
    FINAL_CODE_ACTION
)

from .tree import BaseTree, code_execution


class STEP_TREE(BaseTree):
    STEP_NODE_KEYS: List[str] = ["step", "final_answer", "length"]
    tokenizer: Optional[Any] = None

    def __init__(self, tokenizer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def prompt_wrap(self,question,partial_solution):
        step_delim = self.config.step_delim
        if partial_solution:
            inputs = f"{partial_solution}{step_delim}"
        else:
            inputs = f""
        if self.config.system_prompt is not None and self.config.system_prompt != "":
            question_conversation = [
                {'role':'system','content':self.config.system_prompt},
                {'role':'user','content':question}
                ]
        else:
            question_conversation = [{'role':'user','content':question}]
        prompt = self.tokenizer.apply_chat_template(question_conversation,add_generation_prompt=True,tokenize=False)
        prompt = prompt+inputs
        return prompt

    def step_unwrap(self, text, if_execute=False):
        """
        FINAL_ANSWER_ACTION = r"Final Answer:"
        FINAL_CODE_ACTION = r"\n(.*)\n```\s*(?!python)"

        For code generation, the parsing result is:
        * The goal of assignment, if the last line is an assignment statement.
        * The entire last line code(print(xxx)).
        """

        parser_result = {
            "final_answer": ""
        }
        parsed_answer = ""

        answer_match = re.search(FINAL_ANSWER_ACTION, text)
        if answer_match: # CoT leaf node for metamath
            parsed_answer = text[answer_match.end():].strip()
            parser_result['final_answer'] = parsed_answer
            return text, parser_result

        answer_match = re.search(r"Answer:(.*)", text, re.DOTALL)
        if answer_match: # CoT leaf node for math_cot
            parsed_answer = answer_match.group(1)
            parser_result['final_answer'] = parsed_answer
            return text, parser_result

        code_match = re.search(FINAL_CODE_ACTION, text)
        if code_match and if_execute: # Code leaf node
            parsed_answer = code_match.group(1)
            if "print" not in parsed_answer:
                parsed_answer = parsed_answer.split('=')[0].strip()
            parsed_answer = "<code>"+parsed_answer+"</code>"
        else: # No answer
            parsed_answer = ""
        parser_result['final_answer'] = parsed_answer

        return text, parser_result

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        super().validate_config(cfg)
        if not cfg.mode == "step_tree":
            raise ValueError(f"Wrong value for config mode, must be step_tree")
        if not cfg.n_generate_sample == 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        return BaseNode(
            parent=parent,
            additional_state_keys=self.STEP_NODE_KEYS,
        )

    def create_llm(self):
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        llm = LLM(
            model=self.config.model_dir,
            tensor_parallel_size=len(GPUS),
            trust_remote_code=True,
            seed=self.config.seed,
            swap_space=self.config.swap_space,
        )
        sampling_params = SamplingParams(
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            use_beam_search=self.config.use_beam_search,
            best_of=self.config.best_of,
            max_tokens=self.config.max_tokens,
            stop=self.stop,
            #seed=self.config.seed,
        )
        return partial(
            local_vllm,
            llm=llm,
            sampling_params=sampling_params,
            n=1,
            temperature=self.config.temperature,
        )

    def should_generate_next(self) -> bool:
        return not self.current_node.is_terminal and self.current_node.depth <= self.config.max_depth

    def generate(self) -> None:
        """
        generate as a linked list
        root -> x -> y -> z
        """
        while self.should_generate_next():
            step_result, parser_result = self.get_parsable_samples()
            self.update_current_node(step_result, parser_result)

    def update_current_node(
        self,
        step_result: str,
        parser_result: Dict[str, str],
    ) -> None:
        self._update_current_node(step_result, parser_result)
        self.current_node = self.current_node.children[0]

    def _update_current_node(
        self,
        step_result: str,
        parser_result: Dict[str, str],
        idx: int = 0,
    ) -> None:
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))

        # initialize a new node
        new_node = self.create_node(parent=self.current_node)
        new_node.tag = f"{self.current_node.tag}.{idx}"
        new_node.depth = self.current_node.depth + 1

        # update node state
        if parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
        else:
            new_node.state["text"] = step_result

        # update parent's children
        self.current_node.children.append(new_node)

    def get_parsable_samples(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = self.create_prompt()
        sampled_step_results = self.get_llm_samples(prompt)

        step_result = sampled_step_results[0]
        return self.step_unwrap(step_result)

    def create_prompt(
        self,
    ) -> str:
        partial_solution = self.collect_partial_solution(self.current_node)
        prompt = self.prompt_wrap(
            self.question,
            partial_solution,
            self.config,
        )
        return prompt

    def get_llm_samples(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> List[str]:
        if temperature is None:
            # default llm
            samples = self.llm(prompt, n=n)
        else:
            samples = self.llm(prompt, temperature=temperature, n=n)

        processed_samples = [sample.strip() for sample in set(samples)]
        return processed_samples
