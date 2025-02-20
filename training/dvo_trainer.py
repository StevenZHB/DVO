# KTO Authors: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, has_length

from transformers import is_wandb_available
from transformers.utils import is_peft_available
from trl.models import PreTrainedModelWrapper, create_reference_model

from trl.trainer.utils import (
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
from transformers.utils import(
    is_accelerate_available
)
import logging
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from accelerate.utils import (
        GradientAccumulationPlugin
    )

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _get_kl_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Creates mismatched pairs of prompts and completions for the KL dataset by reversing the order of completions."""
    batch["kl_input_ids"] = batch["input_ids"][::-1]
    batch["kl_attention_mask"] = batch["attention_mask"][::-1]
    batch["kl_labels"] = batch["labels"][::-1]
    batch["kl_q_credits"] = batch["q_credits"][::-1]
    batch["kl_acc"] = batch["acc"][::-1]
    return batch

@dataclass
class DVODataCollatorWithPadding:
    r"""
    DVO Data Collator class that pads the tokenized inputs, attention masks, labels,
    and q_credits to the maximum length of the batch or to the longest item in the batch,
    ensuring that all batch items have uniform size for processing by models.
    NOTE: We will pad the first batch to the largest length to pre allocate memory.

    Args:
        pad_token_id (int, defaults to 0):
            The pad token ID used by the tokenizer to fill in the input_ids.
            This ID is necessary to maintain consistent sequence lengths in input_ids.

        label_pad_token_id (int, defaults to -100):
            The pad token ID used to fill in the labels for masking unwanted computations,
            typically in loss calculations where certain elements should not contribute to the loss.

    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    input_ids_max_length: int = -100
    first_step_done: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for prefix in ['','kl_']:
            input_ids_to_pad = [torch.LongTensor(ex[f"{prefix}input_ids"]) for ex in features]
            input_ids_padded_batch = pad_sequence(input_ids_to_pad, batch_first=True, padding_value=self.pad_token_id)
            attention_mask_to_pad = [torch.LongTensor(ex[f"{prefix}attention_mask"]) for ex in features]
            attention_mask_padded_batch = pad_sequence(attention_mask_to_pad, batch_first=True, padding_value=0)

            if not self.first_step_done:
                max_seq_length = self.input_ids_max_length
                input_ids_padded_batch = torch.full((len(features), self.input_ids_max_length), self.pad_token_id, dtype=torch.long)
                for i, input_ids in enumerate(input_ids_to_pad):
                    length = len(input_ids)
                    input_ids_padded_batch[i, :length] = input_ids
                attention_mask_padded_batch = torch.zeros((len(features), self.input_ids_max_length), dtype=torch.long)
                for i, mask in enumerate(attention_mask_to_pad):
                    length = len(mask)
                    attention_mask_padded_batch[i, :length] = mask
                self.first_step_done = True

            labels_to_pad = [torch.LongTensor(ex[f"{prefix}labels"]) for ex in features]
            max_labels_num = max([len(ex[f"{prefix}labels"]) for ex in features])
            max_seq_length = input_ids_padded_batch.shape[-1]
            labels_padded_batch = torch.full((len(features), max_labels_num, max_seq_length), self.label_pad_token_id, dtype=torch.long)

            for i, example in enumerate(labels_to_pad):
                for j, label in enumerate(example):
                    length = len(label)
                    labels_padded_batch[i, j, :length] = torch.LongTensor(label)

            q_credits_to_pad = [torch.tensor(ex[f"{prefix}q_credits"]) for ex in features]
            q_credits_padded_batch = torch.full((len(features), max_labels_num), self.label_pad_token_id, dtype=torch.float)
            for i, q_credits in enumerate(q_credits_to_pad):
                length = len(q_credits)
                q_credits_padded_batch[i, :length] = q_credits
            if prefix == '':
                if 'reference_logps' in features[0]:
                    reference_logps_to_pad = [torch.tensor(ex['reference_logps'][:len(ex[f"{prefix}q_credits"])]) for ex in features]
                    reference_logps_padded_batch = torch.full((len(features), max_labels_num), self.label_pad_token_id, dtype=torch.float)
                    for i, ref_logps in enumerate(reference_logps_to_pad):
                        length = len(ref_logps)
                        reference_logps_padded_batch[i, :length] = torch.tensor(ref_logps, dtype=torch.float)
                    padded_batch.update({
                        f"reference_logps": reference_logps_padded_batch
                    })

            padded_batch.update({
                f"{prefix}input_ids": input_ids_padded_batch,
                f"{prefix}attention_mask": attention_mask_padded_batch,
                f"{prefix}labels": labels_padded_batch,
                f"{prefix}q_credits": q_credits_padded_batch,
                f"{prefix}acc": torch.tensor([ex["acc"] for ex in features]),
            })

        return padded_batch

def tokenize_row(feature,tokenizer,max_length,label_pad_token_id,step_delim) -> Dict:
    """Tokenize a batch from a DVO specific dataset."""
    batch = {}
    prompt = feature["prompt"]
    completions = feature["completions"]
    q_credits = feature["q_credits"]

    prompt = prompt.lstrip(tokenizer.bos_token)

    prompt_tokenized = tokenizer(prompt,add_special_tokens=False)
    prompt_input_ids = prompt_tokenized['input_ids']
    prompt_attention_mask = prompt_tokenized['attention_mask']

    # Initialize list for all input_ids and attention masks, start with prompt
    full_input_ids = prompt_input_ids
    full_attention_masks = prompt_attention_mask

    current_position = len(prompt_input_ids)
    full_labels = []

    delim_tokens_len = {s_delim:len(tokenizer.tokenize(s_delim)) for s_delim in step_delim}

    for compl_id, completion in enumerate(completions):
        if compl_id == len(completion)-1 and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        completion_tokenized = tokenizer(completion, add_special_tokens=False)
        completion_input_ids = completion_tokenized['input_ids']
        completion_attention_mask = completion_tokenized['attention_mask']
        if len(completion_input_ids) + len(full_input_ids) > max_length:
            q_credits = q_credits[:len(full_labels)]
            break

        # Calculate start and end positions for the current completion
        current_labels = [-100]*current_position
        current_labels.extend(completion_input_ids)
        for delim in delim_tokens_len:
            if completion.endswith(delim):
                end_index = len(current_labels)
                start_index = end_index - delim_tokens_len[delim]
                current_labels[start_index:end_index] = [-100] * delim_tokens_len[delim]
        full_labels.append(current_labels)

        # Append completion tokens to the overall list
        full_input_ids.extend(completion_input_ids)
        full_attention_masks.extend(completion_attention_mask)

        # Update current position for the next completion
        current_position = current_position + len(completion_input_ids)


    assert len(full_labels) == len(q_credits), f"Num of labels must equals to num of q_credits, but found {len(full_labels)} and {len(q_credits)}."

    input_ids_length = len(full_input_ids)
    full_labels = [single_label + [label_pad_token_id] * (input_ids_length - len(single_label)) for single_label in full_labels]

    # add BOS token to head of prompt
    if full_input_ids[0] != tokenizer.bos_token_id:
        full_input_ids = [tokenizer.bos_token_id] + full_input_ids
        full_attention_masks = [1] + full_attention_masks
        full_labels = [[label_pad_token_id]+single_label for single_label in full_labels]

    batch = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_masks,
        "labels": full_labels,
        "q_credits": q_credits,
        "input_ids_length": len(full_input_ids),
        "acc": feature['acc'],
    }
    return batch

class DVOTrainer(Trainer):
    r"""
    Initialize DVOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`DVOConfig`):
            The arguments to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator (`DVODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
    """

    _tag_names = ["trl", "kto"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["mse", "kto", "policy_gradient"] = "kto",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None, # type: ignore
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
    ):
        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the DVOTrainer, the value you passed will override the one in the `DVOConfig`."
            )
            args.model_init_kwargs = model_init_kwargs
        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DVOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DVOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            model_init_kwargs["torch_dtype"] = torch_dtype

        # if ref_model_init_kwargs is not None:
        #     warnings.warn(
        #         "You passed `ref_model_init_kwargs` to the DVOTrainer, the value you passed will override the one in the `DVOConfig`."
        #     )
        args.ref_model_init_kwargs = ref_model_init_kwargs


        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str) and not precompute_ref_log_probs:
            raise ValueError(
                "You passed ref_model_kwargs to the DVOTrainer. But your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DVOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DVOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DVOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it with `pip install peft` to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install with `pip install wandb` to resolve."
            )


        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError(
                "max_length or a tokenizer must be specified when using the default DVODataCollatorWithPadding"
            )
        if max_length is None:
            warnings.warn(
                "When using DVODataCollatorWithPadding, you should set `max_length` in the DVOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512


        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.truncation_mode = truncation_mode
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self.loss_type = loss_type
        self.dataset_num_proc = dataset_num_proc

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # DVO parameter
        self.beta = beta
        self.min_weight = args.min_weight

        # dataset stastic
        self.input_ids_max_length = 0

        self.step_delim = args.step_delim if args.step_delim is not None else [self.tokenizer.eos_token]
        if self.tokenizer.eos_token not in self.step_delim:
            self.step_delim.append(self.tokenizer.eos_token)

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(tokenize_row, fn_kwargs={"tokenizer":self.tokenizer,"max_length":self.max_length,"label_pad_token_id":self.label_pad_token_id,"step_delim":self.step_delim}, num_proc=self.dataset_num_proc,desc="Processing tokenized train dataset",load_from_cache_file=False)
            train_dataset = train_dataset.map(
                _get_kl_dataset, batched=True, batch_size=len(train_dataset), desc="Extracting KL train dataset",load_from_cache_file=False
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(tokenize_row, fn_kwargs={"tokenizer":self.tokenizer,"max_length":self.max_length,"label_pad_token_id":self.label_pad_token_id,"step_delim":self.step_delim}, num_proc=self.dataset_num_proc, desc="Processing tokenized eval dataset",load_from_cache_file=False)
                eval_dataset = eval_dataset.map(
                    _get_kl_dataset, batched=True, batch_size=len(eval_dataset), desc="Extracting KL eval dataset",load_from_cache_file=False
                )

            length_list = [length for length in train_dataset['input_ids_length']]
            self.input_ids_max_length = max(length_list)

            if self.input_ids_max_length > self.max_length:
                self.input_ids_max_length = self.max_length
            warnings.warn(f"Input ids max length: {self.input_ids_max_length},mean length: {sum(length_list)/len(length_list)} will preallocate memory at the first step.")

        if data_collator is None:
            data_collator = DVODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                input_ids_max_length=self.input_ids_max_length
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DVODataCollatorWithPadding, you should set `remove_unused_columns=False` in your DVOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs and 'reference_logps' not in train_dataset.column_names:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if 'reference_logps' in train_dataset.column_names:
            self._precomputed_train_ref_log_probs = True
        if 'reference_logps' in eval_dataset.column_names:
            self._precomputed_eval_ref_log_probs = True

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model


    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_logp = self.compute_reference_log_probs(padded_batch)
                reference_logp = self.accelerator.gather_for_metrics(
                    reference_logp
                )
                reference_logps.append(reference_logp.cpu())

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=reference_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_logp = self.compute_reference_log_probs(padded_batch)
                reference_logp = self.accelerator.gather_for_metrics(
                    reference_logp
                )
                reference_logps.append(reference_logp.cpu())

            eval_dataset = eval_dataset.add_column(
                name="reference_logps", column=reference_logps
            )

            # Save calculated reference_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DVO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    completion_logits = self.model(padded_batch['input_ids'],padded_batch['attention_mask']).logits
            else:
                completion_logits = self.ref_model(padded_batch['input_ids'],padded_batch['attention_mask']).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        return completion_logps

    def dvo_loss(
        self,
        policy_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        q_credits: torch.FloatTensor,
        KL_q_credits: torch.FloatTensor,
        policy_denominator: torch.FloatTensor,
        KL_denominator: torch.FloatTensor,
        label_pad_token_id: int = -100,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DVO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_logps: Log probabilities of the policy model for the responses. Shape: (batch_size,)
            reference_logps: Log probabilities of the reference model for the responses. Shape: (batch_size,)
            q_credits: Q credits of the span. Shape: (batch_size, )

        Returns:
            A tuple of seven tensors: (loss, chosen_credits, rejected_credits, credits_distance, pi_rewards, q_rewards, full_logps).
            The losses tensor contains the DVO loss for each example in the batch.
        """

        pi_credits = policy_logps - reference_logps # (batch_size, max_span_num)

        # flatten_pi_credits = pi_credits.view(-1) # (batch_size*max_span_num, )
        # flatten_q_credits = q_credits.view(-1) # (batch_size*max_span_num, )

        valid_mask = q_credits != label_pad_token_id
        # flatten_valid_mask = valid_mask.view(-1)

        # pi_KL = pi_credits.mean().clamp(min=0)

        # The beta is a temperature parameter for the DVO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DVO loss.
        masked_pi_credits = pi_credits * valid_mask
        masked_q_credits = q_credits * valid_mask
        if self.loss_type == 'kto':
            # kl_valid_mask = KL_q_credits != label_pad_token_id
            kl_valid_mask = KL_q_credits != label_pad_token_id
            kl_values = (policy_KL_logps - reference_KL_logps)[kl_valid_mask].flatten() # (1, )

            kl_mean = kl_values.mean()
            kl_var = kl_values.var()

            gather_kl_mean = self.accelerator.gather(kl_mean)
            global_variance = self.accelerator.gather(kl_var)

            kl_mean = gather_kl_mean.mean()
            kl_variance = (global_variance+(gather_kl_mean-kl_mean)**2).mean().detach()
            kl_mean = kl_mean.clamp(min=0).detach()
            kl_std = torch.sqrt(kl_variance).detach()

            modified_pi_credits = torch.where(q_credits >= 0, (pi_credits-kl_mean), (kl_mean-pi_credits))
            losses = 1 - F.sigmoid(self.beta * modified_pi_credits)
            # losses = losses * valid_mask.float()
            losses = q_credits.abs().clamp(min=self.min_weight) * losses * valid_mask.float()
            # losses = self.beta * losses * q_credits.abs() * valid_mask.float()

        elif self.loss_type == 'mse':
            # βlog(pi*(at|st)/pi_ref(at|st)) = advantage = r + v(t+1) - v(t)
            mse_loss = nn.MSELoss(reduction='none')
            # mse_loss = nn.L1Loss(reduction='none')

            q_credits = q_credits.to(pi_credits.dtype)
            # masked_pi_credits = (pi_credits - kl_mean) / (kl_std + 1e-8) # For normalize
            masked_pi_credits = self.beta * masked_pi_credits
            losses = mse_loss(masked_pi_credits, q_credits) * valid_mask
            kl_mean = None
        elif self.loss_type == 'policy_gradient':
            log_pi_theta = self.beta * policy_logps
            log_pi_ref = reference_logps
            adv = q_credits + log_pi_ref*self.beta
            # masked_adv = adv * valid_mask
            # masked_pi_credits = pi_credits * valid_mask
            # βlog(pi*(at|st)/pi_ref(at|st)) = advantage = r + v(t+1) - v(t)
            adv = adv.to(pi_credits.dtype)
            # masked_pi_credits = (pi_credits - kl_mean) / (kl_std + 1e-8) # For normalize
            # masked_pi_theta = torch.exp(policy_logps)
            # loss = - masked_pi_theta * masked_adv * valid_mask

            # losses = - log_pi_theta * adv * valid_mask
            # losses = - torch.exp(log_pi_theta) * adv * valid_mask
            # Reproduce MSE
            losses = (log_pi_theta ** 2 - 2 * log_pi_theta * adv + adv ** 2) * valid_mask
            # policy gradient loss
            # losses = - log_pi_theta * adv * valid_mask

            # + policy_logps * policy_logps
            # losses = losses.sum(-1)
            kl_mean = None

        else:
            raise ValueError("Unsupported loss type. Choose 'kto' or 'mse'.")

        valid_rows = valid_mask.sum(dim=1) > 0
        losses = losses[valid_rows].sum(dim=1) / valid_mask[valid_rows].sum(dim=1)
        # losses = losses.mean()
        # losses = losses.sum(dim=1)/valid_mask.sum(dim=1)

        # calculate credits of each span
        mask_credits_chosen = masked_q_credits > 0
        mask_credits_rejected = masked_q_credits < 0
        mask_credits_zero = (q_credits==0)

        chosen_credits = pi_credits[mask_credits_chosen].mean(dim=-1) if mask_credits_chosen.any() else torch.tensor(0.0).to(self.accelerator.device)
        rejected_credits = pi_credits[mask_credits_rejected].mean(dim=-1) if mask_credits_rejected.any() else torch.tensor(0.0).to(self.accelerator.device)
        zero_credits = pi_credits[mask_credits_zero].mean(dim=-1) if mask_credits_zero.any() else torch.tensor(0.0).to(self.accelerator.device)

        credits_distance = ((masked_pi_credits - masked_q_credits) ** 2).mean().sqrt()

        # calculate rewards of each data
        pi_rewards = masked_pi_credits.sum(dim=-1) # Sum of q credits equals to rewards, (batch_size,)
        q_rewards = masked_q_credits.sum(dim=-1)

        # calculate logps of each data
        pi_logps = (policy_logps * valid_mask.float()).sum(dim=-1)
        ref_logps = (reference_logps * valid_mask.float()).sum(dim=-1)

        return losses, chosen_credits.detach(), rejected_credits.detach(), zero_credits.detach(), credits_distance.detach(), pi_rewards.detach(), q_rewards.detach(), pi_logps.detach(),ref_logps.detach(), kl_mean

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, max_span_num, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size, max_span_num ) containing the average/sum log probabilities of the given labels under the given logits.
        """
        batch_size, max_span_num, seq_length = labels.shape
        vocab_size = logits.shape[-1]

        if (logits.shape[0] != labels.shape[0]) or (logits.shape[1] != labels.shape[-1]):
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        logits = logits.unsqueeze(1).expand(-1, max_span_num, -1, -1)  # Expand logits to match labels shape
        logits = logits.contiguous().view(batch_size * max_span_num, seq_length, vocab_size)

        labels = labels.view(batch_size * max_span_num, seq_length)


        labels = labels[:, 1:].clone() # (bs * msn * sl * vs)
        logits = logits[:, :-1, :] # (bs * msn * sl)

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        per_token_logps = per_token_logps.view(batch_size, max_span_num, seq_length-1) # Note that seq_length is changed to seq_length -1
        loss_mask = loss_mask.view(batch_size, max_span_num, seq_length-1)

        if average_log_prob:
            denominator = loss_mask.sum(-1)
            # 避免除以零
            denominator = torch.clamp(denominator, min=1e-10)
            return (per_token_logps * loss_mask).sum(-1) / denominator
            # return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            denominator = loss_mask.sum(-1)
            # 避免除以零
            denominator = torch.clamp(denominator, min=1e-10)
            return (per_token_logps * loss_mask).sum(-1), denominator

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self.loss_type=='kto':
            # calculate the kl part
            with torch.no_grad():
                KL_logits = model(
                            batch["kl_input_ids"],
                        attention_mask=batch["kl_attention_mask"],
                    ).logits
                KL_logps, KL_denominator = self.get_batch_logps(
                    KL_logits,
                    batch["kl_labels"],
                    average_log_prob=False,
                    label_pad_token_id=self.label_pad_token_id,
                )
        else:
            KL_logits = None
            KL_logps = None
            KL_denominator = None

        completion_logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        completion_logps, completion_denominator = self.get_batch_logps(
            completion_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        return (completion_logits, KL_logits, completion_logps, KL_logps, completion_denominator, KL_denominator)


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DVO loss for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        q_credits = batch['q_credits']
        KL_q_credits = batch['kl_q_credits']

        (
            _,
            _,
            policy_logps,
            policy_KL_logps,
            policy_denominator,
            KL_denominator
        ) = self.forward(model, batch)

        # if reference_logps in batch use them, otherwise use the reference model
        if "reference_logps" in batch and self.loss_type == 'mse':
            reference_logps = batch["reference_logps"]
            reference_KL_logps = None
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            _,
                            _,
                            reference_logps,
                            reference_KL_logps,
                            policy_denominator,
                            KL_denominator
                        ) = self.forward(self.model, batch)
                else:
                    (
                        _,
                        _,
                        reference_logps,
                        reference_KL_logps,
                        policy_denominator,
                        KL_denominator
                    ) = self.forward(self.ref_model, batch)

        losses, chosen_credits, rejected_credits, zero_credits, credits_distance, pi_rewards, q_rewards, pi_logps, ref_logps, kl_mean = self.dvo_loss(
            policy_logps,
            policy_KL_logps,
            reference_logps,
            reference_KL_logps,
            q_credits,
            KL_q_credits,
            policy_denominator,
            KL_denominator
        )

        chosen_mask = q_rewards > 0
        rejected_mask = q_rewards < 0

        chosen_mask = (batch['acc'] == 1) # mask is from data accuracy
        rejected_mask = (batch['acc'] != 1)

        num_chosen = torch.tensor([chosen_mask.sum()]).to(self.accelerator.device)
        num_rejected = torch.tensor([rejected_mask.sum()]).to(self.accelerator.device)

        # mean_rewards = pi_rewards.mean()
        reward_accuracies = (torch.sign(pi_rewards) == torch.sign(q_rewards)).float()
        # reward_accuracies = (torch.sign(pi_rewards-mean_rewards) == torch.sign(q_rewards)).float()

        prefix = "eval_" if train_eval == "eval" else ""
        if num_chosen > 0:
            metrics[f"{prefix}rewards/chosen"] = ((pi_rewards*chosen_mask.float()).sum() / num_chosen).cpu()
            metrics[f"{prefix}pi_logps/chosen"] = ((pi_logps*chosen_mask.float()).sum() / num_chosen).cpu()
            metrics[f"{prefix}ref_logps/chosen"] = ((ref_logps*chosen_mask.float()).sum() / num_chosen).cpu()

        if num_rejected > 0:
            metrics[f"{prefix}rewards/rejected"] = ((pi_rewards*rejected_mask).sum() / num_rejected).cpu()
            metrics[f"{prefix}pi_logps/rejected"] = ((pi_logps*rejected_mask).sum() / num_rejected).cpu()
            metrics[f"{prefix}ref_logps/rejected"] = ((ref_logps*rejected_mask).sum() / num_rejected).cpu()

        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}credits/chosen"] = chosen_credits.cpu()
        metrics[f"{prefix}credits/rejected"] = rejected_credits.cpu()
        metrics[f"{prefix}credits/zero"] = zero_credits.cpu()
        metrics[f"{prefix}credits/distance"] = credits_distance.cpu()
        if kl_mean is not None:
            metrics[f"{prefix}kl"] = kl_mean.cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DVODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DVODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


    def generate_from_model_and_ref(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DVODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DVODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, 'eval')

        # force log the metrics
        # if self.accelerator.is_main_process:
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["logits/chosen"],
            "eval_logits/rejected": metrics["logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            target_indicies = [i for i in range(len(random_batch["kl"])) if random_batch["kl"][i] is False]
            target_batch = {
                "prompt_input_ids": itemgetter(*target_indicies)(random_batch["prompt_input_ids"]),
                "prompt_attention_mask": itemgetter(*target_indicies)(random_batch["prompt_attention_mask"]),
                "prompt": itemgetter(*target_indicies)(random_batch["prompt"]),
            }
            policy_output_decoded, ref_output_decoded = self.generate_from_model_and_ref(self.model, target_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                target_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "kto" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
