from __future__ import annotations

import os

from termcolor import colored
from typing import Optional, Any, List, Callable, Tuple
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from omegaconf import OmegaConf

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from pebble import ProcessPool,ThreadPool
import concurrent.futures
from concurrent.futures import TimeoutError
from func_timeout import func_set_timeout, FunctionTimedOut

from .agents.tree import BaseTree
from .agents.step_tree_beam import STEP_BEAM_TREE
from .llms.server_llm_engine import bi_llm_engine, bi_llm_server
from .llms.local_llm_engine import local_generator,ref_llm_forward
from .constants import TIMEOUT_SECONDS, ERROR_COLOR
import time
import copy
from functools import partial

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

        if self.config.stop and not isinstance(self.config.stop, list):
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)

        self.create_llm()

        if self.config.mode in ["sbs","step_beam_tree"]:
            self.max_solver_steps = self.config.max_depth
        elif self.config.mode in ["mcts", "step_mcts"]:
            self.max_solver_steps = self.config.iterations
            self.config.step_beam_width = 1

    # @field_validator("config")
    # def validate_config(cls, cfg: Any):
    #     if issubclass(type(cfg), DictConfig):
    #         return cfg
    #     raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")

    def create_llm(self) -> Callable[..., List[str]]:
        if self.config.model_type == "server":
            policy_model, policy_sampling_params, ref_model, ref_sampling_params, tokenizer = bi_llm_server(self.config)
            self.llm = policy_model
            self.ref_llm = ref_model
            self.generate_sampling_params = policy_sampling_params
            self.ref_sampling_params = ref_sampling_params
            self.tokenizer = tokenizer
            self.llm.test_connection()
            if self.config.need_ref_model:
                self.ref_llm.test_connection()
        elif self.config.model_type == "local":
            policy_model,sampling_params,ref_model,tokenizer = bi_llm_engine(self.config)
            self.engine = policy_model
            self.ref_engine = ref_model
            self.tokenizer = tokenizer

            self.generate_sampling_params = sampling_params
            self.value_sampling_params = copy.deepcopy(sampling_params)
            self.value_sampling_params.max_tokens = 1
            self.value_sampling_params.n = 1
            self.llm =  partial(
                local_generator,
                engine=self.engine,
                use_tqdm = False
            )
            self.ref_llm = partial(
                ref_llm_forward,
                model=self.ref_engine,
                pad_token_id = self.tokenizer.pad_token_id,
                batch_size = self.config.ref_batch_size
            )
        else:
            raise NotImplementedError

    @staticmethod
    def processor(solver: BaseTree, output: List[RequestOutput]) -> BaseTree:
        solver.generate_next_step(output)
        return solver

    @staticmethod
    @func_set_timeout(TIMEOUT_SECONDS)
    def selector(solver: BaseTree) -> BaseTree:
        solver.evaluate_and_select_next_step()
        return solver

    def generate_preprocess(self, agents: List[BaseTree]) -> Tuple[List[str], List[int], List[BaseTree], List[BaseTree]]:
        # only generate prompt for the self.candidate_nodes
        prompts = []
        prompts_span = [0]
        valid_solvers = []
        invalid_solvers = []
        for agent in agents:
            if agent.should_generate_next():
                solver_prompts = agent.create_prompt()
                prompts.extend(solver_prompts)
                prompts_span.append(prompts_span[-1] + len(solver_prompts))
                valid_solvers.append(agent)
            else:
                invalid_solvers.append(agent)
        return prompts, prompts_span, valid_solvers, invalid_solvers

    def generate_postprocess(
        self,
        outputs: List[List[RequestOutput]],
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        post_solvers = []
        max_workers = min(len(valid_solvers), os.cpu_count()//8+1)
        for valid_solver, output in zip(valid_solvers, outputs):
            try:
                result = self.__class__.processor(valid_solver, output)
                post_solvers.append(result)
            except:
                post_solvers.append(None)

        updated_solvers = [
            post_solver if post_solver is not None else valid_solver
            for post_solver, valid_solver in zip(post_solvers, valid_solvers)
        ]
        return updated_solvers

    def generate_postprocess_old(
        self,
        outputs: List[List[RequestOutput]],
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        # Update the current agents
        post_solvers = []
        if self.config.verbose:
            print(colored(f"Start processing {len(valid_solvers)} solvers...", "green"))

        max_workers = min(len(valid_solvers), os.cpu_count()//8+1)
        if self.config.parallel_type == "thread":
            decorated_timeout_processor = func_set_timeout(TIMEOUT_SECONDS)(self.__class__.processor)
            with ThreadPool(max_workers=max_workers) as pool:
                futures = [
                    pool.schedule(decorated_timeout_processor, args=(valid_solver, output))
                    for valid_solver, output in zip(valid_solvers, outputs)
                ]

                if len(valid_solvers) > 100:
                    progress_bar = tqdm(total=len(valid_solvers), desc="Execute")
                else:
                    progress_bar = None
                for future in futures:
                    try:
                        result = future.result()
                        post_solvers.append(result)
                    except FunctionTimedOut:
                        post_solvers.append(None)
                        print(colored(f"Task exceeded the timeout of {TIMEOUT_SECONDS} seconds and was terminated.\n", "red"))
                    except SystemExit as error:
                        post_solvers.append(None)
                        print(colored(f"{error}\n", "red"))
                    except Exception as error:
                        post_solvers.append(None)
                        print(colored(f"An unexpected error occurred: {error}\n", "red"))
                        # raise
                    if progress_bar is not None:
                        progress_bar.update(1)
                if progress_bar is not None:
                    progress_bar.close()
        # For process parallel, we cannot use func_timeout, we use the timeout of future.result()
        elif self.config.parallel_type == "process":
            with ProcessPool(max_workers=max_workers) as pool:
                futures = [
                    pool.schedule(self.__class__.processor, args=(valid_solver, output),timeout=TIMEOUT_SECONDS)
                    for valid_solver, output in zip(valid_solvers, outputs)
                ]
                if len(valid_solvers) > 100:
                    progress_bar = tqdm(total=len(valid_solvers), desc="Execute")
                else:
                    progress_bar = None
                for future in futures:
                    try:
                        result = future.result()
                        post_solvers.append(result)
                    except TimeoutError:
                        post_solvers.append(None)
                        print(colored(f"Task exceeded the timeout of {TIMEOUT_SECONDS} seconds and was terminated.\n", "red"))
                    except SystemExit as error:
                        post_solvers.append(None)
                        print(colored(f"{error}\n", "red"))
                    except Exception as error:
                        post_solvers.append(None)
                        print(colored(f"An unexpected error occurred: {error}\n", "red"))
                    if progress_bar is not None:
                        progress_bar.update(1)

                if progress_bar is not None:
                    progress_bar.close()
        else:
            raise NotImplementedError

        # update solvers
        assert len(valid_solvers) == len(post_solvers), f"Data is not matched, {len(valid_solvers)} vs {len(post_solvers)}."
        updated_solvers = [
            post_solver if post_solver is not None else valid_solver
            for post_solver, valid_solver in zip(post_solvers, valid_solvers)
        ]
        return updated_solvers

    def evaluation(
        self,
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        def selector_wrapper(solver) -> None:
            if solver is not None:
                self.selector(solver)

        max_workers = min(len(valid_solvers), os.cpu_count())  # 根据需求调整线程数
        with ThreadPool(max_workers=max_workers) as pool:
            future = pool.map(selector_wrapper, valid_solvers)
        results = list(future.result())
        return valid_solvers

    def update_valid_solvers(
        self,
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        def update_valid_solvers_wrapper(solver) -> None:
            if solver is not None:
                solver.update_candidate_node()
        max_workers = min(len(valid_solvers), os.cpu_count())  # 根据需求调整线程数
        with ThreadPool(max_workers=max_workers) as pool:
            future = pool.map(update_valid_solvers_wrapper, valid_solvers)
        results = list(future.result())
        return valid_solvers


    def postprocess(
        self,
        valid_solvers: List[BaseTree],
        invalid_solvers: List[BaseTree],
    ) -> List[BaseTree]:

        # update solvers
        invalid_solvers.extend(valid_solvers)
        return invalid_solvers

    def solve(self, agents: List[BaseTree]):
        # for solver in solvers:
        #     solver.init_code_solution()

        for step in tqdm(range(self.max_solver_steps), desc="Step Processing"):

            step_start_time = time.time()
            prompts, prompts_span, valid_solvers, invalid_solvers = self.generate_preprocess(agents)

            if len(valid_solvers) < 1:
                break

            if self.config.verbose:
                print(f"Preprocessing time: {time.time() - step_start_time:.2f} seconds")
            step_start_time = time.time()

            # llm run for step generation
            if step == 0:
                n = self.config.n_generate_sample * self.config.step_beam_width
            else:
                n = self.config.n_generate_sample
            if self.config.model_type == "server":
                # use vllm server
                # Sampling_params in OpenAIModel is a dict
                self.llm.sampling_params['n'] = n
                self.llm.sampling_params['best_of'] = n
                outputs = self.llm.batch_generate(prompts)

                if self.config.need_ref_model:
                    outputs = self.ref_llm.ref_generate(input=prompts, outputs=outputs)
                else:
                    for out in outputs:
                        for completions in out.choices:
                            completions.q_pi = sum(completions.logprobs.token_logprobs)
                            completions.q_ref = -100
                            completions.value_estimate = -100
            elif self.config.model_type == "local":
                self.generate_sampling_params.n = n
                self.generate_sampling_params.best_of = n
                outputs = self.llm(prompts, self.generate_sampling_params)
                if self.config.need_ref_model:
                    outputs = self.ref_llm(outputs=outputs)
                else:
                    for out in outputs:
                        for completions in out.outputs:
                            completions.q_pi = completions.cumulative_logprob
                            completions.q_ref = -100
                            completions.value_estimate = -100
            else:
                raise NotImplementedError

            if self.config.verbose:
                print(f"LLM step generation time: {time.time() - step_start_time:.2f} seconds")
            step_start_time = time.time()

            # post-process outputs
            reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]

            # process output and run python interpreter
            # In cot settings, it's used for generate the new nodes with output


            valid_solvers = self.generate_postprocess(reconstructed_outputs, valid_solvers)

            if self.config.verbose:
                print(f"Post-processing time: {time.time() - step_start_time:.2f} seconds")
            step_start_time = time.time()

            # valid_solvers = self.evaluation(valid_solvers)
            valid_solvers = self.update_valid_solvers(valid_solvers)

            if self.config.verbose:
                print(f"Evaluation and further post-processing time: {time.time() - step_start_time:.2f} seconds")

            agents = self.postprocess(valid_solvers, invalid_solvers)

        return self.output(agents)

    def output(self, solvers: List[BaseTree]):
        jsonlines = {}
        for i, solver in enumerate(solvers):
            jsonlines[solver.question] = solver.return_states()

        return jsonlines
