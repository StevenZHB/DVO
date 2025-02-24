#%%
from __future__ import annotations

import argparse
import json
import numpy as np

from typing import Any, Dict, Type, Optional, List, Tuple
from pydantic import BaseModel
from omegaconf import OmegaConf
from tqdm import tqdm

from mcts_math.constants import (
    NO_VALID_CHILD,
    TOO_MANY_STEPS,
    TOO_MANY_CODE_ERRORS,
)
from mcts_math.config import BaseConfig
from mcts_math.agents.utils import math_is_equiv
from glob import glob
import os
from pebble import ThreadPool, ProcessPool
from concurrent.futures import TimeoutError
#%%
class InferNode(BaseModel):

    tag: str = "0"

    text: str = ""
    extra_info: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    c_puct: float = 1.25
    depth: int = 0

    prior: float = 1.0
    value: float = 0
    q_value: float = 0
    visit_count: int = 0

    parent: Optional[Any] = None
    children: List[Any] = []

    prune: bool = False

    def puct(self) -> float:
        q_value = self.q_value if self.visit_count > 0 else 0
        if self.value == -100:
            u_value = self.c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        else: # self.value = old q_pi-q_credit
            u_value = self.c_puct * self.value * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value

#%%
def rebuild_tree(
    tree_dict: Dict[str, Any],
    max_num_children: int,
    c_puct: float,
    root_tag: str = "0",
) -> Tuple[Type[InferNode], int, List[Type[InferNode]]]:
    root = InferNode(
        parent=None,
        tag=root_tag,
        c_puct=c_puct,
        **tree_dict[root_tag],
    )
    candidates = [root]
    leaf_nodes = []
    max_depth = 0
    while candidates:
        node = candidates.pop(0)
        for idx in range(max_num_children):
            tag = f"{node.tag}.{idx}"
            depth = node.depth + 1
            if tag in tree_dict:
                child = InferNode(
                    parent=node,
                    tag=tag,
                    depth=depth,
                    c_puct=c_puct,
                    **tree_dict[tag],
                )
                max_depth = max(max_depth, depth)
                node.children.append(child)
                candidates.append(child)
        if len(node.children)==0:
            leaf_nodes.append(node)
    return root, max_depth, leaf_nodes

#%%
def is_valid_final_answer_node(node: Type[InferNode]) -> bool:
    if not node.children and node.final_answer and \
        node.final_answer not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
        return True
    return False


def prune_node(node: Type[InferNode]) -> bool:
    if node.children:
        children_prune = []
        for child in node.children:
            children_prune.append(prune_node(child))
        if all(children_prune):
            node.prune = True
    else:
        # for leaf node
        if not is_valid_final_answer_node(node):
            node.prune = True
    return node.prune


def select_non_prune(current_nodes: List[Type[InferNode]]) -> List[Type[InferNode]]:
    candidate_nodes = []
    for node in current_nodes:
        candidate_nodes.extend([child for child in node.children if not child.prune])
    return candidate_nodes


def sort_by_strategy(
    candidate_nodes: List[Type[InferNode]],
    strategy: str = "q_value",
) -> List[Type[InferNode]]:
    if strategy == "value":
        return sorted(candidate_nodes, key=lambda x: x.value, reverse=True)
    elif strategy == "q_value":
        return sorted(candidate_nodes, key=lambda x: x.q_value, reverse=True)
    elif strategy == "visit_count":
        return sorted(candidate_nodes, key=lambda x: x.visit_count, reverse=True)
    elif strategy == "puct":
        return sorted(candidate_nodes, key=lambda x: x.puct(), reverse=True)
    else:
        raise NotImplementedError(f"strategy {strategy} not implemented")

def get_solution(
    full_tree_dict: Dict[str, Any],
    prune: bool = False,
    b1: int = 1,
    b2: int = 5,
    strategy: str = "q_value",
    c_puct: float = 1.25,
) -> Optional[Dict[str, Any]]:
    """
    This function is used to extract solution from a built tree.
    It is mainly used for MCTS, but also works for saved tree from step_beam.
    """
    question = full_tree_dict["question"]
    ground_truth = full_tree_dict.get("answer", None)
    tree_dict = full_tree_dict["step_mcts"]

    # rebuild tree
    root, tree_depth, leaf_nodes = rebuild_tree(tree_dict, max_num_children=b1*b2, c_puct=c_puct)

    # pruning tree
    if prune:
        prune_node(root)
        if root.prune:
            # no valid leaf node for the entire tree
            return None

    # search in tree
    final_answer_nodes = []
    current_top_num = b1
    current_nodes = [root]

    for _ in range(tree_depth):
        candidate_nodes = select_non_prune(current_nodes)
        candidate_nodes = sort_by_strategy(candidate_nodes, strategy)
        current_nodes = candidate_nodes[:current_top_num]

        for current_node in current_nodes[:]:
            if is_valid_final_answer_node(current_node):
                final_answer_nodes.append(current_node)
                current_nodes.remove(current_node)
                current_top_num -= 1
            elif not current_node.children:
                current_nodes.remove(current_node)
                current_top_num -= 1

    if not final_answer_nodes:
        return None

    final_answer_nodes = sort_by_strategy(final_answer_nodes, strategy)
    top_final_answer_node = final_answer_nodes[0]

    return {
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": top_final_answer_node.final_answer,
        "tag": top_final_answer_node.tag,
    }

def get_all_solutions(
    full_tree_dict: Dict[str, Any],
    prune: bool = False,
    max_num_children: int = 15,
    strategy: str = "q_value",
    c_puct: float = 1.25,
) -> List[Optional[Dict[str, Any]]]:
    """
    This function is used to extract all possible solutions from a built tree.
    It is mainly used for MCTS, but also works for saved tree from step_beam.
    """
    question = full_tree_dict["question"]
    ground_truth = full_tree_dict.get("answer", None)
    tree_dict = full_tree_dict["step_mcts"]

    # rebuild tree
    root, tree_depth, leaf_nodes = rebuild_tree(tree_dict, max_num_children=max_num_children, c_puct=c_puct)

    # pruning tree
    if prune:
        prune_node(root)
        leaf_nodes = [node for node in leaf_nodes if not node.prune]
        if not leaf_nodes:
            # no valid leaf node for the entire tree
            return []

    solutions = []
    for leaf_node in leaf_nodes:
        path = []
        node = leaf_node
        while node is not None:
            path.append({
                "text": node.text,
                "value": node.value,
                "q_value": node.q_value,
                "prior": node.prior,
                "visit_count": node.visit_count,
            })
            node = node.parent
        path.reverse()
        # assert leaf_node.q_value in [-1,1]
        solutions.append({
            "question": question,
            "ground_truth": ground_truth,
            "final_answer": leaf_node.final_answer,
            "tag": leaf_node.tag,
            "acc": math_is_equiv(ground_truth,leaf_node.final_answer),
            # "acc": leaf_node.q_value==1,
            "path": path,
        })

    solutions = sorted(solutions, key=lambda x: x["path"][-1][strategy], reverse=True)

    return solutions

def process_line(line):
    full_tree_dict = json.loads(line)
    solutions = get_all_solutions(
        full_tree_dict,
        prune=True,
        max_num_children=config.step_beam_width*config.n_generate_sample,
        strategy="q_value",
        c_puct=config.c_puct
    )
    return solutions
#%%
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument('--qaf', type=str, default="MetaMathQA_train_20K_RL.json")
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)
    #%%
    llm_version = os.path.basename(config.model_dir.rstrip("/"))
    # qaf_basename = 'trainset_annotation.json'
    qaf_basename = os.path.basename(args.qaf)

    config_dir = os.path.dirname(args.custom_cfg)
    model_name = os.path.basename(config_dir)

    tree_jsonl = f"results/{model_name}/{qaf_basename}.step_mcts.{llm_version}.round_{config.round_name}*.jsonl"


    cnt, total = 0, 0
    sampled_pair_solutions = []
    files = glob(tree_jsonl)
    saved_jsonl_file = tree_jsonl.replace(f"results/{model_name}/",f"results/{model_name}/sampled_").replace("*.jsonl",".jsonl")
    print(f"tree_jsonl: {tree_jsonl}")
    print(f"Saved sampled data to: {saved_jsonl_file}")
    for file in files:
        print(f"Read in file:",file)
        with open(file, "r") as f:
            lines = f.readlines()
        with ProcessPool(max_workers=96) as pool:
            future = pool.map(process_line, lines,timeout=15)

            iterator = future.result()
            for _ in tqdm(range(len(lines)), desc=f"Processing file"):
                try:
                    result = next(iterator)
                    sampled_pair_solutions.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(f"Function timeout")
                except Exception as error:
                    print(f"Function raised {error}")

    with open(saved_jsonl_file, "w") as writer:
        for solution_pairs in sampled_pair_solutions:
            writer.write(json.dumps(solution_pairs, ensure_ascii=False) + '\n')
            writer.flush()

    print(f"Sampled data saved to: {saved_jsonl_file}")