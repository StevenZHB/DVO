from transformers import (
    TrainerCallback,
)
from accelerate import Accelerator
from tqdm import tqdm
import datasets
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb
import logging
import pandas as pd
import json
from typing import Union
from math_evaluation import is_equiv
import signal
from tqdm import tqdm
from .grading.grader import grade_answer
logger = logging.getLogger(__name__)

class evaluation_generation_callback(TrainerCallback):
    def __init__(self, query_dataset):
        self.query_dataset = datasets.load_from_disk(query_dataset)
        # .select(range(64))
        self.overall_len = len(self.query_dataset)
        self.batch_size = 4

    def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
        # 创建 DataLoader
        sampler = DistributedSampler(self.query_dataset, shuffle=False)
        dataloader = DataLoader(self.query_dataset, sampler=sampler, batch_size=self.batch_size)
        model.eval()

        qa_pairs = []
        tqdm_progress_bar = tqdm(dataloader,total=len(dataloader)) if state.is_world_process_zero else dataloader

        for batch in tqdm_progress_bar:
            batch_question = batch['prompt']
            ground = batch['answer']
            prompt_message = [tokenizer.apply_chat_template([{'role':'user','content':question}], tokenize=False, add_generation_prompt=True) for question in batch_question]
            inputs = tokenizer(prompt_message, return_tensors="pt", padding=True,add_special_tokens=False)['input_ids'].cuda()
            outputs = model.generate(inputs,max_new_tokens = 512,pad_token_id=tokenizer.eos_token_id)
            input_length = inputs.shape[1]
            answer = tokenizer.batch_decode(outputs[:,input_length:], skip_special_tokens=True)
            qa_pairs.extend(zip(batch_question,answer,ground))

        # collect all output
        gathered_qa_pairs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_qa_pairs, qa_pairs)
        if state.is_world_process_zero:
            output_log = []
            cor = 0
            gathered_qa_pairs = [_ for qa in gathered_qa_pairs for _ in qa]
            for qa_pair in gathered_qa_pairs:
                (question,answer,ground) = qa_pair
                answer_choice = self.parse_answer(answer)
                output_log.append({'prompt':question,'output': answer,'choice': answer_choice,'gound': ground})
                if ground == answer_choice:
                    cor+=1
            overall_acc = cor/self.overall_len
            output_df = pd.DataFrame(output_log)
            # build a wandb.Table
            wandb.log({'eval/Overall_acc': overall_acc,
                       'Summ': wandb.Table(dataframe=output_df)},
                         step=state.global_step)
            # wandb.log(evaluate_log, step=state.global_step)
            logger.info(f"Accuracy in evaluation: {overall_acc}")
            logger.info(output_df)
        model.train()


    def parse_answer(self,answer):
        answer_choice_list = ['A','B','C']
        try:
            answer = answer.split('The correct option is: ')[-1].strip()
            if answer[0] in answer_choice_list:
                answer_choice = answer[0]
            else:
                answer_choice = 'N'
        except Exception as e:
            print(str(e))
            answer_choice = 'N'
        return answer_choice



def remove_single_dollar(s):
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s

# def math_is_equiv(grt: Union[str, list[str]], prd: str):
#     prd = remove_single_dollar(prd)
#     if isinstance(grt, list):
#         for g in grt:
#             if is_equiv(remove_single_dollar(g), prd):
#                 return True
#         return False
#     else:
#         return is_equiv(remove_single_dollar(grt), prd)
def math_is_equiv(grt: Union[str, list[str]], prd: str):
    prd = remove_single_dollar(prd)
    if isinstance(grt, list):
        for g in grt:
            if grade_answer(prd,remove_single_dollar(g)):
                return True
        return False
    else:
        return grade_answer(prd,remove_single_dollar(grt))


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

class evaluate_GSM8K_callback(TrainerCallback):
    def __init__(self, query_dataset,system_prompt=None,eos_token_id=None):
        self.query_dataset = json.load(open(query_dataset))
        self.overall_len = len(self.query_dataset)
        self.batch_size = 4
        self.system_prompt = system_prompt
        self.eos_token_id = eos_token_id

    def on_evaluate(self, args, state, control, model, processing_class, **kwargs):
        # 创建 DataLoader
        sampler = DistributedSampler(self.query_dataset, shuffle=False)
        dataloader = DataLoader(self.query_dataset, sampler=sampler, batch_size=self.batch_size)
        model.eval()

        qa_pairs = []
        tqdm_progress_bar = tqdm(dataloader,total=len(dataloader)) if state.is_world_process_zero else dataloader
        original_padding_side = processing_class.padding_side
        processing_class.padding_side ='left'
        eos_token_id = self.eos_token_id if self.eos_token_id is not None else processing_class.eos_token_id


        for batch in tqdm_progress_bar:
            batch_question = batch['question']
            ground = batch['answer']
            data_type = [u_id.split('_')[0] for u_id in batch['unique_id']]
            if self.system_prompt is None:
                prompt_message = [processing_class.apply_chat_template([{'role':'user','content':question}], tokenize=False, add_generation_prompt=True) for question in batch_question]
            else:
                prompt_message = [processing_class.apply_chat_template([{'role':'system','content':self.system_prompt},{'role':'user','content':question}], tokenize=False, add_generation_prompt=True) for question in batch_question]
            encoding = processing_class(prompt_message, return_tensors="pt", padding=True,add_special_tokens=False)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            outputs = model.generate(input_ids = input_ids, attention_mask = attention_mask,max_new_tokens = 512,pad_token_id=processing_class.eos_token_id,temperature=0.01,top_p=0.9,do_sample=False,eos_token_id=eos_token_id)
            input_length = input_ids.shape[1]
            answer = processing_class.batch_decode(outputs[:,input_length:], skip_special_tokens=True)
            qa_pairs.extend(zip(prompt_message,answer,ground,data_type))

        # collect all output
        gathered_qa_pairs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_qa_pairs, qa_pairs)
        if state.is_world_process_zero:
            output_log = []
            type_cor = {}
            type_num = {}
            gathered_qa_pairs = [_ for qa in gathered_qa_pairs for _ in qa]
            for qa_pair in tqdm(gathered_qa_pairs,desc="checking acc"):
                (question,answer,ground,data_type) = qa_pair
                try:
                    # signal.alarm(30)
                    answer_span = self.parse_answer(answer)
                    output_log.append({'prompt':question,'output': answer,'final_answer': answer_span,'gound': ground,'data_type': data_type})
                    if data_type not in type_cor:
                        type_cor[data_type] = 0
                        type_num[data_type] = 0
                    if math_is_equiv(ground, answer_span):
                        type_cor[data_type]+=1
                    type_num[data_type]+=1
                except:
                    print(f"Skipping QA pair due to timeout: {qa_pair}")
                    continue
                # finally:
                #     signal.alarm(0)
            output_df = pd.DataFrame(output_log)
            for d_type in type_cor:
                type_cor[d_type] /= type_num[d_type]
                wandb.log({f'eval/{d_type}_acc': type_cor[d_type],'train/global_step': state.global_step})
            # build a wandb.Table
            wandb.log({'Summ': wandb.Table(dataframe=output_df),'train/global_step': state.global_step})
            # wandb.log(evaluate_log, step=state.global_step)
            logger.info(output_df)
            logger.info(f"Accuracy in evaluation: {type_cor}")
        processing_class.padding_side = original_padding_side
        model.train()


    def parse_answer(self,answer):
        try:
            if 'Final Answer: ' in answer:
                answer = answer.split('Final Answer: ')[-1].strip()
            else:
                answer = answer.split('Answer:\n')[-1].strip()
        except Exception as e:
            print(str(e))
            answer = 'None'
        return answer


class EmptyCachePerLoggingStepCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()
    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     torch.cuda.empty_cache()

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


