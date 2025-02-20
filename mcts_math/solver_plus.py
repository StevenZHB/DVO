"""
author: lmp-decaderan
email: ldecaderan@gmail.com
"""
from __future__ import annotations

import os
import copy

from termcolor import colored
from functools import partial
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
from tqdm import tqdm
from abc import abstractmethod
from pydantic import BaseModel, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from pebble import ProcessPool
from concurrent.futures import TimeoutError

from .agents.tree import BaseTree

# from .llms.local_llms import local_generator, server_generator, local_server_generator
from .llms.local_llm_engine import llm_engine, bi_llm_engine, ref_llm_forward, get_sampling_params, ref_llm_server_forward, bi_llm_server
from .constants import TIMEOUT_SECONDS, ERROR_COLOR


class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Any

    stop: List[str] = None

    llm: Optional[Callable[..., List[str]]] = None
    ref_llm: Optional[Callable[..., List[str]]] = None

    tokenizer: Optional[Callable[..., List[str]]] = None

    llm_model_id: Optional[str] = None
    ref_llm_model_id: Optional[str] = None

    engine: Optional[LLM] = None
    ref_engine: Optional[Callable[..., List[str]]] = None

    generate_sampling_params: Optional[SamplingParams] = None
    ref_sampling_params: Optional[SamplingParams] = None
    value_sampling_params: Optional[SamplingParams] = None
    max_solver_steps: int = 1

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.llm_model_id = self.config.model_dir
        self.ref_llm_model_id = self.config.ref_model_dir

        if self.config.stop:
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)

        self.create_llm()

        if self.config.mode in ["sbs","step_beam_tree"]:
            self.max_solver_steps = self.config.max_depth
        elif self.config.mode in ["mcts", "step_mcts"]:
            self.max_solver_steps = self.config.iterations
            self.config.step_beam_width = 1

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg

        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")

    def create_llm(self) -> Callable[..., List[str]]:
        policy_model, policy_sampling_params, ref_model, ref_sampling_params, tokenizer = bi_llm_server(self.config)
        self.llm = policy_model
        self.ref_llm = ref_model
        self.generate_sampling_params = policy_sampling_params
        self.ref_sampling_params = ref_sampling_params
        self.tokenizer = tokenizer
        self.llm.test_connection()
        if self.config.need_ref_model:
            self.ref_llm.test_connection()

    @staticmethod
    def processor(solver: BaseTree, output: List[RequestOutput]) -> BaseTree:
        solver.generate_next_step(output)
        return solver

    @staticmethod
    def selector(solver: BaseTree) -> BaseTree:
        # solver.evaluate_and_select_next_step()
        solver.select_next_step(None)
        return solver

    def generate_preprocess(self, solvers: List[BaseTree]) -> Tuple[List[str], List[int], List[BaseTree], List[BaseTree]]:
        prompts = []
        prompts_span = [0]
        valid_solvers = []
        invalid_solvers = []

        for solver in solvers:
            if solver.should_generate_next():
                solver_prompts = solver.create_prompt()
                prompts.extend(solver_prompts)
                prompts_span.append(prompts_span[-1] + len(solver_prompts))
                valid_solvers.append(solver)
            else:
                invalid_solvers.append(solver)
        return prompts, prompts_span, valid_solvers, invalid_solvers

    def generate_postprocess(
        self,
        outputs: List[List[RequestOutput]],
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        post_solvers = []
        if self.config.verbose:
            print(colored(f"Start processing {len(valid_solvers)} solvers...", "green"))
        with ProcessPool(max_workers=min(len(valid_solvers), os.cpu_count())) as pool:
        # with ProcessPool(1) as pool:
            future = pool.map(self.__class__.processor, valid_solvers, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        if len(valid_solvers) > 100:
            progress_bar = tqdm(total=len(valid_solvers), desc="Execute")
        else:
            progress_bar = None

        while True:
            try:
                result = next(iterator)
                # input(f"result = {result}")
                post_solvers.append(result)
            except StopIteration:
                break
            except Exception as error:
                if self.config.verbose:
                    print(colored(f"{error}\n", ERROR_COLOR))
                # raise
                post_solvers.append(None)
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        # update solvers
        assert len(valid_solvers) == len(post_solvers), f"Data is not matched, {len(valid_solvers)} vs {len(post_solvers)}."
        updated_solvers = [
            post_solver if post_solver is not None else valid_solver
            for post_solver, valid_solver in zip(post_solvers, valid_solvers)
        ]
        return updated_solvers

    def value_preprocess(self, solvers: List[BaseTree]) -> Tuple[List[str], List[int]]:
        prompts = []
        prompts_span = [0]

        for solver in solvers:
            solver_prompts = solver.create_prompt(is_value_only=True)
            prompts.extend(solver_prompts)
            prompts_span.append(prompts_span[-1] + len(solver_prompts))
        return prompts, prompts_span

    def evaluation(
        self,
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        for solver in valid_solvers:
            if solver is not None:
                self.selector(solver)
        return valid_solvers

    def postprocess(
        self,
        valid_solvers: List[BaseTree],
        invalid_solvers: List[BaseTree],
    ) -> List[BaseTree]:

        # update solvers
        invalid_solvers.extend(valid_solvers)
        return invalid_solvers

    def solve(self, solvers: List[BaseTree]):

        for step in tqdm(range(self.max_solver_steps), desc="Step Processing"):

            prompts, prompts_span, valid_solvers, invalid_solvers = self.generate_preprocess(solvers)

            if len(valid_solvers) < 1:
                break

            # llm run for step generation
            if step == 0:
                n = self.config.n_generate_sample * self.config.step_beam_width
            else:
                n = self.config.n_generate_sample

            

            # outputs = self.llm.batch_generate(prompts, self.generate_sampling_params, use_tqdm=True if self.config.verbose else False)
            # use vllm server
            # Sampling_params in OpenAIModel is dict
            self.llm.sampling_params['n'] = n
            self.llm.sampling_params['best_of'] = n
            outputs = self.llm.batch_generate(prompts)

            if self.config.need_ref_model:
                outputs = self.ref_llm.ref_generate(input=prompts, outputs=outputs)
            else:
                for out in outputs:
                    for completions in out:
                        completions.q_pi = sum(completions.logprobs.token_logprobs)
                        completions.q_ref = -100
                        completions.value_estimate = -100
            
            # post-process outputs
            reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            # process output and run python interpreter
            # In cot settings, it's used for generate the new nodes with output
            valid_solvers = self.generate_postprocess(reconstructed_outputs, valid_solvers)

            valid_solvers = self.evaluation(valid_solvers)

            solvers = self.postprocess(valid_solvers, invalid_solvers)

        return self.output(solvers)

    def output(self, solvers: List[BaseTree]):
        jsonlines = {}
        for i, solver in enumerate(solvers):
            jsonlines[solver.question] = solver.return_states()

        return jsonlines
