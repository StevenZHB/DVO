"""
author: lovecambi
email: interfk@gmail.com
"""
import json
from typing import List, Optional, Literal
from enum import Enum, EnumMeta

from dataclasses import dataclass, field
from omegaconf import II


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


_SEARCH_CHOICES = [
    "react",
    "mcts",         # monte carlo tree search
    "sbs",    # step-level beam search
    "step_mcts",
    "step_beam_tree"
]

SEARCH_CHOICES = ChoiceEnum(_SEARCH_CHOICES)


_PROMPT_CHOICES = [
    "react",          # thought/action/observation with examples for round 1
    "react_sft",      # xml format in SFT
]

PROMPT_CHOICES = ChoiceEnum(_PROMPT_CHOICES)


_MCTS_INFER_CHOICES = [
    "value",
    "puct",
    "q_value",
    "visit_count",
]

MCTS_INFER_CHOICES = ChoiceEnum(_MCTS_INFER_CHOICES)

@dataclass
class BaseConfig:

    round_name: Optional[str] = field(
        metadata={"help": "round name for file name"}
    )
    mode: SEARCH_CHOICES = field(
        default="mcts", metadata={"help": "search mode for inference"}
    )
    model_dir: Optional[str] = field(
        default=None, metadata={"help": "llm model dir"}
    )
    ref_model_dir: Optional[str] = field(
        default=None, metadata={"help": "reference llm model dir"}
    )
    model_api: Optional[str] = field(
        default=None, metadata={"help": "llm model api url"}
    )
    ref_model_api: Optional[str] = field(
        default=None, metadata={"help": "reference api url"}
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "model type"}
    )
    ref_batch_size: Optional[int] = field(
        default=8, metadata={"help": "ref llm model forward batch size"}
    )
    few_shot_path: Optional[str] = field(
        default=None, metadata={"help": "few shot data json"}
    )
    prompt_path: Optional[str] = field(
        default=None, metadata={"help": "prompt config json"}
    )
    num_few_shot: int = field(
        default=2, metadata={"help": "the number of few-shot examples"}
    )
    create_local_llm: bool = field(
        default=False, metadata={"help": "not for batch inference"}
    )
    # prompt args
    # prompt_wrap: PROMPT_CHOICES = field(
    #     default="react", metadata={"help": "prompt wrap type"}
    # )
    # result_unwrap: PROMPT_CHOICES = field(
    #     default="react", metadata={"help": "result unwrap"}
    # )

    # Deprecated.
    step_delim: str = field(
        default="", metadata={"help": "delimiter between two steps"}
    )# By default, we will append a stop token after each step, which is typically the required step_delim. An exception is the `EOS` token, which we handle in the Final answer parser section.

    # vllm args
    temperature: float = field(
        default=0.7, metadata={"help": "control diversity of llm generation"}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens."}
    )
    top_k: float = field(
        default=-1.0, metadata={"help": "Float that controls the probability of other highly-scored candidates to be chosen"}
    )
    use_beam_search: bool = field(
        default=False, metadata={"help": "whether to enable beam search decoding"}
    )
    best_of: int = field(
        default=1, metadata={"help": "Integer that controls the number of candidate considered in the beam search decoding process"}
    )
    max_tokens: int = field(
        default=1024, metadata={"help": "Maximum number of tokens to generate per output sequence."}
    )
    max_seq_length: int = field(
        default=4096, metadata={"help": "Maximum number of tokens for model forward."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "seed of llm generation for reproducible"}
    )
    swap_space: Optional[int] = field(
        default=16, metadata={"help": "swap space for vllm"}
    )
    n_generate_sample: int = field(
        default=1, metadata={"help": "how many samples generated for each step. B2 in paper."}
    )
    stop: Optional[List[str]] = field(
        default=None, metadata={"help": "possible stop tokens for each step"}
    )
    # Deprecated
    include_stop_str_in_output: str = field(
        default=False, metadata={"help": "include stop tokens in output.choices[x].text"}
    )# Sglang would ignore this para. By default, we will append a stop token after each step.

    # agent (mcts, step_beam) args
    step_beam_width: int = field(
        default=1, metadata={"help": "beam width for each step. B1 in paper."}
    )
    max_depth: int = field(
        default=4, metadata={"help": "maximum depth of the tree, ie., maximum steps of completion."}
    )
    iterations: int = field(
        default=1, metadata={"help": "number of simulations in mcts"}
    )
    max_total_time: int = field(
        default=300, metadata={"help": "maximum program runing time."}
    )
    reward_weight: float = field(
        default=0.5, metadata={"help": "balance value and reward in mcts"}
    )
    positive_reward: float = field(
        default=1.0, metadata={"help": "reward for positive example"}
    )
    negative_reward: float = field(
        default=-1.0, metadata={"help": "reward for negative example"}
    )
    errors_threshold: int = field(
        default=0, metadata={"help": "maximum code errors allowed, ie., if errors_count > errors_threshold, the tree growth from this node should be terminated."}
    )

    initial_solution: bool = field(
        default=False, metadata={"help": "init the tree with the golden solution"}
    )
    # need_value_func: bool = field(
    #     default=False, metadata={"help": "whether to use value head in decoding"}
    # )
    need_ref_model: bool = field(
        default=True, metadata={"help": "whether to use reference model"}
    )
    update_leaf_value: bool = field(
        default=False, metadata={"help": "update leaf value in mcts"}
    )
    update_on_step: bool = field(
        default=False, metadata={"help": "update on every step"}
    )
    c_puct: float = field(
        default=1.5, metadata={"help": "weight of c_puct in mcts"}
    )
    is_sampling: bool = field(
        default=False, metadata={"help": "solution generation in mcts"}
    )
    remove_duplicate: bool = field(
        default=False, metadata={"help": "remove duplicate children nodes in step beam or mcts"}
    )
    # offline inferene args
    prune: bool = field(
        default=False, metadata={"help": "prune the tree in a complete mcts tree"}
    )
    # mcts_infer_strategy: MCTS_INFER_CHOICES = field(
    #     default="q_value", metadata={"help": "the strategy to select node in a complete mcts tree"}
    # )
    # other args
    batch_size: int = field(
        default=-1, metadata={"help": "batch size for batch inference"}
    )
    verbose: bool = field(
        default=False, metadata={"help": "print intermediate steps on screen"}
    )
    model_api: str = field(
        default="http://0.0.0.0:5555/v1", metadata={"help": "The port for policy model."}
    )
    ref_model_api: str = field(
        default="http://0.0.0.0:5555/v1", metadata={"help": "The port for ref model."}
    )
    ref_gpu: int = field(
        default=1, metadata={"help": "The gpu id of the reference model."}
    )
    system_prompt: str = field(
        default = ""
    )
    parallel_type: str = field(
        default = "thread", metadata={"help": "The parallel type for generate postprocess, thread or process."}
    )
    use_python_interpreter: bool = field(
        default=False, metadata={"help": "If use the python interpreter to get the final answer."}
    )


if __name__ == '__main__':
    # example usage
    from omegaconf import OmegaConf

    config = OmegaConf.structured(
        BaseConfig(
            mode="react",
            stop=[
                "\nObservation:",
                "Observation:",
            ],
        )
    )
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)
