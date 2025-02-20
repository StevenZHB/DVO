# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Literal, Optional, Union

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
import os

from .configs import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{'<|user|>: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|user|>: ' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>: '  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>: ' }}{% endif %}{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})

def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo", "dqo"] = "sft", auto_insert_empty_system_msg: bool = True,
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            # Now we extract the final turn to define chosen/rejected responses
            # chosen_messages = example["chosen"][-1:]
            # rejected_messages = example["rejected"][-1:]
            # example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            # example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            # example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            text_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            text_chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False, add_generation_prompt=False)
            text_rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False, add_generation_prompt=False)
            assert text_chosen.startswith(text_prompt) and text_rejected.startswith(text_prompt), f"Prompt {text_prompt} not found in chosen {text_chosen} or rejected {text_rejected} responses"
            example["text_prompt"] = text_prompt
            example["text_chosen"] = _strip_prefix(text_chosen, text_prompt)
            example["text_rejected"] = _strip_prefix(text_rejected, text_prompt)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dqo":
        assert "q_credits" in example.keys(), f"q_credits not found in example keys {example.keys()}"
        assert "completions" in example.keys(), f"completions not found in example keys {example.keys()}"

        prompt_messages = example["prompt"]
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(prompt_messages, tokenizer)
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        example["prompt"] = prompt


    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    data_config: Union[DataArguments , dict],
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            if os.path.isdir(ds):
                if "train" in split:
                    raw_train_datasets.append(
                        load_from_disk(
                            ds
                        )[split]
                    )
                elif "test" in split:
                    raw_val_datasets.append(
                        load_from_disk(
                            ds
                        )[split]
                    )
                else:
                    raise ValueError(f"Split type {split} not recognized as one of test or train.")
            else:
                if "train" in split:
                    raw_train_datasets.append(
                        load_dataset(
                            ds,
                            split=split,
                            cache_dir='/zhanghongbo/alignment-handbook/datasets',
                        )
                    )
                elif "test" in split:
                    raw_val_datasets.append(
                        load_dataset(
                            ds,
                            split=split,
                            cache_dir='/zhanghongbo/alignment-handbook/datasets',
                        )
                    )
                else:
                    raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        val_subsets = []
        for dataset, frac in zip(raw_val_datasets, fracs):
            val_subset = dataset.select(range(int(frac * len(dataset))))
            val_subsets.append(val_subset)
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(val_subsets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(val_subsets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
