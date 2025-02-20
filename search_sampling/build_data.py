#%%
from __future__ import annotations

import argparse
import json
import numpy as np

from typing import Any, Dict, Type, Optional, List
from pydantic import BaseModel
from omegaconf import OmegaConf
from tqdm import tqdm
import os
from mcts_math.constants import (
    NO_VALID_CHILD,
    TOO_MANY_STEPS,
    TOO_MANY_CODE_ERRORS,
)
from mcts_math.config import BaseConfig
from mcts_math.agents.utils import math_is_equiv
from glob import glob
import json
from itertools import product
import random

#%%
# pair_data_file_path = "results/math_trainset_annotation.json.step_mcts.deepstepmath-value-2.*.8_tmp_sampled.jsonl"
SYSTEM_PROMPT = """You are tasked with solving the provided math word problem by following these instructions:\n1. Formulate and solve the equations using algebraic methods, ensuring each equation is written strictly in LaTeX syntax.\n2. Document each step of your process clearly. Use double line breaks '\n\n' to separate each step and ensure that there are no double line breaks within a single step.\n3. Ensure the number of reasoning steps is within 8 steps.\n4. Conclude your solution by stating the final answer after the phrase 'Final Answer:'."""
def remove_duplicates(data, key):
    seen = set()
    unique_data = []
    for item in data:
        value = item[key]
        if value not in seen:
            seen.add(value)
            unique_data.append(item)
    return unique_data

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', '-D', type=str, required=True)
    args.add_argument('--threshold', '-T', type=int, default=8)
    args.add_argument('--max_choosing_num', '-M', type=int, default=8)
    args.add_argument('--system_prompt', '-S', type=bool, default=False)
    args = args.parse_args()
    return args

#%%
args = parse_args()
#%%
pair_data_read_in = []
with open(args.data_path, "r") as f:
    for line in f:
        pair_data = json.loads(line)
        pair_data_read_in.append(pair_data)

print(f"Read in {len(pair_data_read_in)} questions in total.")

data_dir = os.path.dirname(args.data_path)
model_name = os.path.basename(data_dir)

output_path = f"generated_data/{model_name}/"+os.path.basename(args.data_path)
output_path = output_path.replace('.','_').replace('_jsonl','') + "_unbalanced"
#%%
random.shuffle(pair_data_read_in)
pref_data_train = []
pref_data_test = []

chosen_num_per_pair = []
rejected_num_per_pair = []

pair_data_train = pair_data_read_in
# pair_data_test = pair_data_read_in[-100:]

# pair_data_train = pair_data_read_in[:1000]
# pair_data_test = pair_data_read_in[1000:1100]

threshold = args.threshold
max_choosing_num = args.max_choosing_num

def sample_data(pair_data, threshold=4, max_choosing_num=4):
    chosen_data = [p for p in pair_data if p["acc"]]
    rejected_data = [p for p in pair_data if not p["acc"]]

    chosen_data_num = len(chosen_data)
    rejected_data_num = len(rejected_data)
    sample_num = min(chosen_data_num, rejected_data_num, max_choosing_num)
    sample_chosen_num = min(max_choosing_num, chosen_data_num)
    sample_rejected_num = min(max_choosing_num, rejected_data_num)

    sample_chosen = []
    sample_rejected = []

    if sample_num == 0:
        if chosen_data_num > 0:
            weights_chosen = np.array([p['all_visit_num'] for p in chosen_data])
            if weights_chosen.sum() > 0:
                weights_chosen = weights_chosen / weights_chosen.sum()
                sample_chosen = list(np.random.choice(chosen_data, min(threshold, chosen_data_num), replace=False, p=weights_chosen))

        if rejected_data_num > 0:
            weights_rejected = np.array([p['all_visit_num'] for p in rejected_data])
            if weights_rejected.sum() > 0:
                weights_rejected = weights_rejected / weights_rejected.sum()
                sample_rejected = list(np.random.choice(rejected_data, min(threshold, rejected_data_num), replace=False, p=weights_rejected))
    else:
        weights_chosen = np.array([p['all_visit_num'] for p in chosen_data])
        if weights_chosen.sum() > 0:
            weights_chosen = weights_chosen / weights_chosen.sum()
            sample_chosen = list(np.random.choice(chosen_data, sample_chosen_num, replace=False, p=weights_chosen))
        # else:
        #     sample_chosen = list(np.random.choice(chosen_data, 1, replace=False))

        weights_rejected = np.array([p['all_visit_num'] for p in rejected_data])
        if weights_rejected.sum() > 0:
            weights_rejected = weights_rejected / weights_rejected.sum()
            # sample_rejected = list(np.random.choice(rejected_data, sample_num, replace=False, p=weights_rejected))
            sample_rejected = list(np.random.choice(rejected_data, sample_rejected_num, replace=False, p=weights_rejected))

    return sample_chosen, sample_rejected

only_chosen_data = []
only_rejected_data = []
both_data = []
ignored_data = []
for pair_data in tqdm(pair_data_train,desc="training sampling"):
    for p in pair_data:
        p['all_visit_num'] = sum(_['visit_count'] for _ in p['path'])
        p['all_q_value'] = abs(p['path'][-1]['q_value']-p['path'][0]['q_value'])
        # p['all_visit_num'] = p['all_q_value']
    sample_chosen, sample_rejected = sample_data(pair_data, threshold, max_choosing_num)

    if len(sample_chosen) == 0 and len(sample_rejected) == 0:
        ignored_data.append(pair_data)
    elif len(sample_chosen) == 0 and len(sample_rejected) > 0:
        only_rejected_data.append(pair_data)
    elif len(sample_chosen) > 0 and len(sample_rejected) == 0:
        only_chosen_data.append(pair_data)
    else:
        both_data.append(pair_data)


    chosen_num_per_pair.append(len(sample_chosen))
    pref_data_train.extend(sample_chosen)
    if len(sample_chosen) > 0:
        rejected_num_per_pair.append(len(sample_rejected))
        pref_data_train.extend(sample_rejected)

print(f"One pair data has both chosen and rejected: {len(both_data)}, only chosen: {len(only_chosen_data)}, only rejected: {len(only_rejected_data)}, ignored: {len(ignored_data)}")
# for pair_data in tqdm(pair_data_test,desc="testing sampling"):
#     for p in pair_data:
#         p['all_visit_num'] = sum(_['visit_count'] for _ in p['path'])
#         p['all_q_value'] = abs(p['path'][-1]['q_value']-p['path'][0]['q_value'])

#     sample_chosen, sample_rejected = sample_data(pair_data, threshold, max_choosing_num)

#     chosen_num_per_pair.append(len(sample_chosen))
#     rejected_num_per_pair.append(len(sample_rejected))
#     pref_data_test.extend(sample_chosen)
#     pref_data_test.extend(sample_rejected)
random.shuffle(pref_data_train)
pref_data_test = pref_data_train[-200:]
pref_data_train = pref_data_train[:-200]

print(f"Sampled data: {len(pref_data_train)} pieces for training, {len(pref_data_test)} pieces for testing.")

# %%
# Q(at|st) = log(Pi*(at|st))-log(Pi_ref(at|st))
# V(st+1) = log(Pi*(at|st))-log(Pi_ref(at|st))
# Delta_Q(at|st) = Q(at|st)-Q(at-1|st-1) = Q(at|st)-V(st)

# Not Q credit, more as an adavantage function:
# We need calculate v(st+1) - v(st) + r(st,at) which will be aligned to \beta log(pi*(at|st)/pi_ref(at|st))
# Q* = logits_Pi*
# V* = \beta log sum(softmax(logits_Pi*)*logits_Pi*)
# At the first time, we do not use V in MCTS-Backpropogation
# NOTE: In the data, we need adv = v(st+1) - v(st) + r(st,at), For step before T, r(st,at) = 0, r(sT-1,aT-1)=r
# then here adv == q_value - q_value_before
# q_value_t = value_t-1
LAMBDA = 0
BETA = 0
assemble_pref_data_train = []

for d in pref_data_train:
    if args.system_prompt:
        prompt = [{"role":"system","content":SYSTEM_PROMPT}]
    else:
        prompt = []
    prompt.append({"role":"user","content":d["question"]})
    last_round_msgs = []
    last_round_q_credits = []
    assert d["path"][0]["text"] == '', d
    for i in range(1,len(d["path"])):
        #
        last_q = d["path"][i-1]["q_value"]
        q = d["path"][i]["q_value"]
        q_credit = q-last_q
        text = d["path"][i]["text"]
        # if i != len(d["path"])-1 and not text.endswith("\n\n"):
        #     text += "\n\n"
        last_round_msgs.append(text)
        # Weighted sum with the log ratio
        q_credit = (LAMBDA)*BETA*d["path"][i]["value"] + (1-LAMBDA)*q_credit
        last_round_q_credits.append(q_credit)
    # last_round = {"role":"assistant","content":last_round_msgs, "q_credits":last_round_q_credits}
    copy_d = d.copy()
    del(copy_d["path"])
    copy_d["prompt"] = prompt
    copy_d["completions"] = last_round_msgs
    copy_d["q_credits"] = last_round_q_credits
    if not isinstance(copy_d["ground_truth"],str):
        copy_d["ground_truth"] = json.dumps(copy_d["ground_truth"])

    # assert sum(last_round_q_credits) == 1 or sum(last_round_q_credits) == -1
    assemble_pref_data_train.append(copy_d)

assemble_pref_data_test = []

for d in pref_data_test:
    if args.system_prompt:
        prompt = [{"role":"system","content":SYSTEM_PROMPT}]
    else:
        prompt = []
    prompt.append({"role":"user","content":d["question"]})
    last_round_msgs = []
    last_round_q_credits = []
    assert d["path"][0]["text"] == '', d
    for i in range(1,len(d["path"])):
        #
        last_q = d["path"][i-1]["q_value"]
        q = d["path"][i]["q_value"]
        q_credit = q-last_q
        text = d["path"][i]["text"]
        # if i != len(d["path"])-1 and not text.endswith("\n\n"):
        #     text += "\n\n"
        last_round_msgs.append(text)
        # Weighted sum with the log ratio
        q_credit = (LAMBDA)*BETA*d["path"][i]["value"] + (1-LAMBDA)*q_credit
        last_round_q_credits.append(q_credit)
    # last_round = {"role":"assistant","content":last_round_msgs, "q_credits":last_round_q_credits}

    copy_d = d.copy()
    del(copy_d["path"])
    copy_d["prompt"] = prompt
    copy_d["completions"] = last_round_msgs
    copy_d["q_credits"] = last_round_q_credits
    # assert sum(last_round_q_credits) == 1 or sum(last_round_q_credits) == -1
    if not isinstance(copy_d["ground_truth"],str):
        copy_d["ground_truth"] = json.dumps(copy_d["ground_truth"])

    assemble_pref_data_test.append(copy_d)

# %%
random.shuffle(assemble_pref_data_train)
random.shuffle(assemble_pref_data_test)
# assemble_pref_data_train = assemble_pref_data_train[:5000]

print("Example data as below:")
print(random.choice(assemble_pref_data_train))
# %%
import datasets

assemble_pref_dataset_train = datasets.Dataset.from_list(assemble_pref_data_train)
assemble_pref_dataset_test =  datasets.Dataset.from_list(assemble_pref_data_test)
# %%
assemble_pref_datadict = datasets.DatasetDict({"train_prefs":assemble_pref_dataset_train,"test_prefs":assemble_pref_dataset_test})

# %%
print("Saving data to",output_path)
assemble_pref_datadict.save_to_disk(output_path)
# %%
