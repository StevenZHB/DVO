#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import logging
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_path)
from training import (
    DataArguments,
    DVOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    get_checkpoint,
    is_adapter_model,
    evaluate_GSM8K_callback,
    EvaluateFirstStepCallback,
    EmptyCachePerLoggingStepCallback,
    OtherConfig,
    DVOTrainer,
    reserve_max_memory,
)
from peft import PeftConfig, PeftModel
import wandb
from tqdm import tqdm
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'
logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DVOConfig, OtherConfig))
    model_args, data_args, training_args, other_args = parser.parse()

    # set output direction with run name
    output_dir = training_args.output_dir
    run_name = training_args.run_name
    if run_name not in output_dir:
        output_dir = os.path.join(output_dir,run_name)
        training_args.output_dir = output_dir


    if isinstance(data_args.dataset_mixer,str):
        import json
        data_args.dataset_mixer = json.loads(data_args.dataset_mixer)

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    # accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200000))])
    # accelerator = Accelerator()

    # if accelerator.is_main_process:
    #     wandb.require("core")
    #     wandb.init(project=training_args.run_name,name=training_args.run_sub_name,settings=wandb.Settings(_disable_stats=True, _disable_meta=True))

    ###############
    # Load datasets
    ###############
    # /zhanghongbo/cuihan/Super_MARIO_TINY/generated_data/deepseekmath-instruct/sampled_trainset_annotation_json_step_mcts_deepseekmath-instruct-pattern-sft_round_1_precompute_ref_logps
    keys_to_update = []
    if training_args.precompute_ref_log_probs:
        for key in data_args.dataset_mixer.keys():
            new_key = key + '_precompute_ref_logps'
            if os.path.exists(new_key):
                logger.info(f"{new_key} exists and precompute_ref_log_probs is True, will use data from {new_key}")
                keys_to_update.append((key, new_key))
            else:
                raise ValueError(f"{new_key} does not exist and precompute_ref_log_probs is True")

        for old_key, new_key in keys_to_update:
            data_args.dataset_mixer[new_key] = data_args.dataset_mixer[old_key]
            del data_args.dataset_mixer[old_key]

    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names = [name for name in column_names if name not in ['q_credits', 'prompt', 'completions','reference_logps','reference_logits','acc']]

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    if other_args.test_oom:
        #Sample 1000 data for quickly begin training
        raw_datasets['train'] = raw_datasets['train'].select(list(range(1000)))
        raw_datasets['test'] = raw_datasets['test'].select(list(range(100)))

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dvo","auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # ##
    # if self.state.global_step == 20:
    #     import torch.distributed as dist
    #     import code
    #     import os
    #     if dist.get_rank() == 0:
    #         # GPU 0 enters interactive mode
    #         print("Entering interactive mode on GPU 0...")
    #         os.system('stty sane')
    #         code.interact(banner="start debug",local=dict(globals(), **locals()),exitmsg="end debug")
    #         os.system('stty sane')
    #     # Barrier: All GPUs wait here, GPU 0 will reach here after interactive session
    #     dist.barrier()
    # ##

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Completions sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['completions']}")
        logger.info(f"Q_credits sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['q_credits']}")


    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
        model_args.model_name_or_path,
        revision=model_args.model_revision
        )
        model_kwargs = None

    ref_model = training_args.ref_model_name_or_path if training_args.ref_model_name_or_path is not None else model_args.model_name_or_path
    ref_model = None if training_args.precompute_ref_log_probs is True else ref_model
    ref_model_kwargs = model_kwargs if training_args.precompute_ref_log_probs is not True else None

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    ########################
    # Make Callback
    ########################

    if 'llama' in model_args.model_name_or_path.lower():
        eos_token_id = 128009
    else:
        eos_token_id = tokenizer.eos_token_id
    GSMcallback = evaluate_GSM8K_callback('/zhanghongbo/cuihan/Super_MARIO_TINY/data/testset_annotation_2057.json',
            # system_prompt="You are tasked with solving the provided math word problem by following these instructions:\n1. Formulate and solve the equations using algebraic methods, ensuring each equation is written strictly in LaTeX syntax.\n2. Document each step of your process clearly. Use double line breaks '\n\n' to separate each step and ensure that there are no double line breaks within a single step.\n3. Ensure the number of reasoning steps is within 8 steps.\n4. Conclude your solution by stating the final answer after the phrase 'Final Answer:'.",
            eos_token_id=eos_token_id
        )


    #########################
    # Instantiate DVO trainer
    #########################
    # Reserve max cuda memory
    # reserve_max_memory(0.9)

    dvo_trainer = DVOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        precompute_ref_log_probs=training_args.precompute_ref_log_probs,
        callbacks=[
                    # GSMcallback,
                    # EvaluateFirstStepCallback(),
                    EmptyCachePerLoggingStepCallback(),
                   ]
    )

    if dvo_trainer.accelerator.is_main_process:
        wandb.require("core")
        wandb.init(project=training_args.run_name,name=training_args.run_sub_name,settings=wandb.Settings(_disable_stats=True, _disable_meta=True))



    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = dvo_trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    dvo_trainer.log_metrics("train", metrics)
    dvo_trainer.save_metrics("train", metrics)
    dvo_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    dvo_trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if dvo_trainer.accelerator.is_main_process:
        dvo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dvo_trainer.model.config.use_cache = True
        dvo_trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = dvo_trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["test"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["test"]))
        dvo_trainer.log_metrics("eval", metrics)
        dvo_trainer.save_metrics("eval", metrics)


    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        dvo_trainer.push_to_hub(**kwargs)


    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    dvo_trainer.accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()

