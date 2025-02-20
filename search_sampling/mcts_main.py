from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from tqdm import tqdm

from omegaconf import OmegaConf

from mcts_math.agents import STEP_MCTS
from mcts_math.solver import Solver
from mcts_math.config import BaseConfig
from react_demo import load_qaf, batch
from transformers import AutoTokenizer
import random
from pathlib import Path
from glob import glob

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument(
        "--qaf", "--question-answer-file",
        type=str,
        required=True,
        help="the file includes question / partial solution (optional) / answer (optional)")
    args.add_argument(
        "--bucket",
        type=str,
        default="0,1",
        help="The bucket index and the number of all buckets for the question-answer data")

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
    # print('cuda visible:',os.environ['CUDA_VISIBLE_DEVICES'])

    bucket_index = int(args.bucket.split(",")[0])
    num_buckets = int(args.bucket.split(",")[1])
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(bucket_index)

    llm_version = os.path.basename(config.model_dir.rstrip("/"))
    qaf_basename = os.path.basename(args.qaf.rstrip("/"))

    data = load_qaf(args.qaf)

    config_dir = os.path.dirname(args.custom_cfg)
    model_name = os.path.basename(config_dir)

    random.seed(config.seed)
    # data = random.sample(data,2000)

    split_data = [data[i::num_buckets] for i in range(num_buckets)]
    data = split_data[bucket_index]

    print(f"Total data num: {len(data)}")

    solver = Solver(config=config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)


    Path(f"results/{model_name}").mkdir(parents=True, exist_ok=True)
    saved_jsonl_file = f"results/{model_name}/{qaf_basename}.{config.mode}.{llm_version}.round_{config.round_name}.jsonl"
    if num_buckets > 1:
        saved_jsonl_file = f"results/{model_name}/{qaf_basename}.{config.mode}.{llm_version}.round_{config.round_name}.{bucket_index}.{num_buckets}.jsonl"

    exist_jsonl_pattern = f"results/{model_name}/{qaf_basename}.{config.mode}.{llm_version}.round_{config.round_name}.*.jsonl"
    exist_jsonl_files = glob(exist_jsonl_pattern)
    processed_data_unique_id = set()
    for exist_jsonl_file in exist_jsonl_files:
        if os.path.exists(exist_jsonl_file):
            with open(exist_jsonl_file, "r") as reader:
                for line in reader:
                    json_line = json.loads(line)
                    processed_data_unique_id.add(json_line["unique_id"])
    print(f"Processed data unique id: {len(processed_data_unique_id)}")
    unprocessed_data = []
    for d in data:
        if d["unique_id"] not in processed_data_unique_id:
            unprocessed_data.append(d)
    print(f"Unprocessed data num: {len(unprocessed_data)}")
    data = unprocessed_data

    with open(saved_jsonl_file, "a") as writer:
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing",total=(len(data)//config.batch_size)+1):
            agents = [
                STEP_MCTS(
                    config=config,
                    tokenizer=tokenizer,
                    question=d["question"],
                    unique_id=d['unique_id'],
                    ground_truth=d["answer"] if config.is_sampling else None,
                    solutions=d['solution'] if 'solution' in d else "") for d in cur_data ]
            jsonlines = solver.solve(agents)
            for d in cur_data:
                question = d["question"]
                d[config.mode] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()

