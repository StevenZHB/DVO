import os
import platform
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import re
import argparse
from accelerate import Accelerator
import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
import itertools
from torch.utils.data import DataLoader, DistributedSampler
import logging
from math_evaluation import is_equiv
from typing import Union
import subprocess
import time
import openai
import backoff
import concurrent.futures
from functools import partial
import atexit
from collections import defaultdict
from func_timeout import func_timeout, FunctionTimedOut
from typing import Union
# 确保程序结束时终止后台进程
def cleanup(process):
    process.terminate()
    print("Terminated the server process.")

def start_vllm_backend(args):
    """
    nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "MODEL_PATH" \
    --served-model-name "MODEL_NAME" \
    --chat-template "TEMPLATE_PATH" \
    --port 9554 \
    > logs/vllm_openai_server.log 2>&1 &
    Application startup complete.
    nohup python3 -m sglang.launch_server \
    --model-path "MODEL_PATH" \
    --served-model-name "MODEL_NAME" \
    --chat-template "TEMPLATE_PATH" \
    --port 9554 \
    > logs/vllm_openai_server.log 2>&1 &
    """
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    num_devices = len(cuda_visible_devices.split(',')) if cuda_visible_devices else 0
    print(f'{num_devices} gpus available.')
    dp = str(int(num_devices//args.tp))
    tp = str(args.tp)
    if args.backend == 'sglang':
        command = [
            "nohup", "python3", "-m", "sglang.launch_server",
            "--model-path", args.model_path,
            "--served-model-name", "vllm_model_name",
            "--mem-fraction-static", "0.9",
            # "--chat-template", args.template_path,
            "--disable-cuda-graph",
            "--port", str(args.port),
            "--dp", dp,
            "--tp", tp
        ]
    elif args.backend == 'vllm':
        command = [
        "nohup", "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--served-model-name", "vllm_model_name",
        "--enforce-eager",
        # "--chat-template", args.template_path,
        "--port", str(args.port),
        "--tensor_parallel_size", str(num_devices)
    ]

    log_file = f'logs/test_math_{args.backend}_{os.path.basename(args.model_path)}.log'
    # os.makedirs(log_file, exist_ok=True)

    # 打开日志文件
    with open(log_file, 'w') as lf:
        # 启动后台进程
        process = subprocess.Popen(
            command,
            stdout=lf,
            stderr=subprocess.STDOUT
        )
    print(f"Started process with PID: {process.pid}")

    atexit.register(cleanup,process)

    start_time = time.time()
    started = False
    timeout = 1800
    while not started and (time.time() - start_time) < timeout:
        with open(log_file, 'r') as lf:
            for line in lf:
                if "Application startup complete." in line:
                    started = True
                    break
        if not started:
            print("Waiting for the server to start...")
            time.sleep(2)  # 等待2秒后再次检查
    assert started, "VLLM server backend failed."

    client = openai.OpenAI(
            api_key="API_KEY",
            base_url = f"http://localhost:{args.port}/v1"
        )
    print("Server started successfully.")
    return client, process

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client,**kwargs):
    return client.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def chat_completions_with_backoff(client,**kwargs):
    return client.chat.completions.create(**kwargs)

def chat_generate(input_string, client, system_prompt, stop, temperature,best_of):
    if system_prompt:
        messages=[
                    {"role": "system", "content": "You are tasked with solving the provided math word problem by following these instructions:\n1. Formulate and solve the equations using algebraic methods, ensuring each equation is written strictly in LaTeX syntax.\n2. Document each step of your process clearly. Use double line breaks '\n\n' to separate each step and ensure that there are no double line breaks within a single step.\n3. Ensure the number of reasoning steps is within 8 steps.\n4. Conclude your solution by stating the final answer after the phrase 'Final Answer:'."},
                    {"role": "user", "content": input_string}
                    ]
    else:
        messages=[
                    {"role": "user", "content": input_string}
                    ]
    response = chat_completions_with_backoff(
            client = client,
            model = "vllm_model_name",
            messages=messages,
            max_tokens = 1024,
            temperature = temperature,
            top_p = 1.0,
            stop = stop,
            extra_body={
                "use_beam_search": True,
                "best_of": best_of
                } if best_of>1 else {}
    )

    generated_text = response.choices[0].message.content.strip()

    return generated_text

def chat_batch_generate(input_string_list, client, system_prompt, stop, temperature,best_of,tokenizer):
    if system_prompt:
        messages=[tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are tasked with solving the provided math word problem by following these instructions:\n1. Formulate and solve the equations using algebraic methods, ensuring each equation is written strictly in LaTeX syntax.\n2. Document each step of your process clearly. Use double line breaks '\n\n' to separate each step and ensure that there are no double line breaks within a single step.\n3. Ensure the number of reasoning steps is within 8 steps.\n4. Conclude your solution by stating the final answer after the phrase 'Final Answer:'."},
                    {"role": "user", "content": input_string}
                    ],add_generation_prompt=True,tokenize=False) for input_string in input_string_list]
    else:
        messages=[tokenizer.apply_chat_template([
                    {"role": "user", "content": input_string}
                    ],add_generation_prompt=True,tokenize=False) for input_string in input_string_list]
    response = completions_with_backoff(
            client = client,
            model = "vllm_model_name",
            prompt=messages,
            max_tokens = 1024,
            temperature = temperature,
            top_p = 0.1,
            stop = stop,
            extra_body={
                "use_beam_search": True,
                "best_of": best_of
                } if best_of>1 else {}
    )

    generated_text = [response.choices[i].text.strip() for i in range(len(input_string_list))]

    return generated_text


def parse_answer(answer):
    try:
        answer = answer.split('Final Answer: ')[-1].strip()
    except Exception as e:
        print(str(e))
        answer = 'None'
    return answer

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

    def safe_is_equiv(g, prd):
        try:
            return func_timeout(30, is_equiv, args=(remove_single_dollar(g), prd))
        except FunctionTimedOut:
            return False

    if isinstance(grt, list):
        for g in grt:
            if safe_is_equiv(g, prd):
                return True
        return False
    else:
        return safe_is_equiv(grt, prd)

def calculate_statistics(data_list):
    if not data_list:
        return {}, {}, {}

    keys = data_list[0].keys()

    # 初始化字典以存储总和、最大值和最小值
    sums = defaultdict(float)
    max_values = defaultdict(lambda: float('-inf'))
    min_values = defaultdict(lambda: float('inf'))

    count = len(data_list)

    for item in data_list:
        for key in keys:
            value = item.get(key, 0.0)  # 如果某个键缺失，默认值为 0.0
            sums[key] += value
            if value > max_values[key]:
                max_values[key] = value
            if value < min_values[key]:
                min_values[key] = value

    # 计算平均值
    averages = {key: total / count for key, total in sums.items()}
    max_values = dict(max_values)
    min_values = dict(min_values)

    return averages, max_values, min_values

def main(args):
    inp_file = args.input_path

    model_path = args.model_path

    client,process = start_vllm_backend(args)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if args.stop is None and '<|eot_id|>' in tokenizer.vocab:
        args.stop = ['<|eot_id|>',tokenizer.eos_token]
        print(f"Set <|eot_id|> as s top token.")

    # generate_function = partial(chat_generate,client=client,system_prompt=args.system_prompt,stop=args.stop,temperature=0.0,best_of=args.beams)
    generate_function = partial(chat_batch_generate,client=client,system_prompt=args.system_prompt,stop=args.stop,temperature=args.temp,best_of=args.beams,tokenizer=tokenizer)

    query_dataset = json.load(open(inp_file))

    type_cor_list = []
    eval_num = 3 if args.temp!=0 else 1
    for eval_id in range(eval_num):
        print(f"Temp: {args.temp}, evaluate at {eval_id}/{eval_num}...")
        if 'checkpoint' in model_path:
            model_path_split = model_path.split('/')[-2:]
            if args.beams==1:
                out_file = os.path.join("./test_results",'_'.join(model_path_split)+'_temp_'+str(args.temp).replace('.', '_')+f"_eval_{eval_id}_{eval_num}_"+"_"+str(os.path.basename(inp_file)))
            else:
                out_file = os.path.join("./test_results",f"beams_{args.beams}"+'_'.join(model_path_split)+'_temp_'+str(args.temp).replace('.', '_')+f"_eval_{eval_id}_{eval_num}_"+"_"+str(os.path.basename(inp_file)))
        else:
            if args.beams==1:
                out_file = os.path.join("./test_results",str(os.path.basename(model_path))+'_temp_'+str(args.temp).replace('.', '_')+f"_eval_{eval_id}_{eval_num}_"+"_"+str(os.path.basename(inp_file)))
            else:
                out_file = os.path.join("./test_results",f"beams_{args.beams}"+str(os.path.basename(model_path))+'_temp_'+str(args.temp).replace('.', '_')+f"_eval_{eval_id}_{eval_num}_"+"_"+str(os.path.basename(inp_file)))

        batch_size = 512
        results = []
        for i in tqdm(range(0, len(query_dataset), batch_size), desc="Processing batches"):
            batch = query_dataset[i:i+batch_size]
            batch_questions = [data['question'] for data in batch]

            try:
                batch_answers = generate_function(batch_questions)

                for data, answer in zip(batch, batch_answers):
                    data['final_answer'] = answer
                    results.append(data)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                for data in batch:
                    data['final_answer'] = f"Error: {str(e)}"
                    results.append(data)


        w = open(out_file,'w')
        output_log = []
        type_cor = {}
        type_num = {}
        output_log = []
        for result in tqdm(results):
            question = result['question']
            answer = result['final_answer']
            ground = result['answer']
            data_type = result['unique_id'].split('_')[0]
            answer_span = parse_answer(answer)
            if data_type not in type_cor:
                type_cor[data_type] = 0
                type_num[data_type] = 0
            if math_is_equiv(ground, answer_span):
                type_cor[data_type]+=1
                acc=1
            else:
                acc = 0
            type_num[data_type]+=1
            log = {'prompt':question,'output': answer,'final_answer': answer_span,'ground': ground,'data_type':data_type,'acc':acc}
            output_log.append(log)
            w.write(json.dumps(log)+"\n")

        for d_type in type_cor:
            type_cor[d_type] /= type_num[d_type]
        print(f"Accuracy in evaluation: {type_cor}")
        log = {"model_name":args.model_path,"inp_file":args.input_path}
        log.update(type_cor)
        type_cor_list.append(type_cor)
        w.write(json.dumps(log)+'\n')

    average_values, maximum_values, minimum_values = calculate_statistics(type_cor_list)
    print("平均值 (Averages):")
    print(average_values)
    log = {"corr_type":"Mean"}
    log.update(average_values)
    w.write(json.dumps(log)+'\n')

    print("\n最大值 (Maximums):")
    print(maximum_values)
    log = {"corr_type":"Max"}
    log.update(maximum_values)
    w.write(json.dumps(log)+'\n')

    print("\n最小值 (Minimums):")
    print(minimum_values)
    log = {"corr_type":"Min"}
    log.update(minimum_values)
    w.write(json.dumps(log)+'\n')

    cleanup(process)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the desired checkpoint folder, e.g., path/checkpoint-12",
        default=""),
    parser.add_argument(
        "--template_path",
        type=str,
        default="",
        help="jinja file path to the chat template"
        ),
    parser.add_argument(
        "--input_path",
        type=str,
        default="./eval/test_set/math_test.json",
        help="path to the input data",
    )
    parser.add_argument(
        "--port",
        type=int,
        default= 9554,
        help="vllm port",
    )
    parser.add_argument(
        "--system_prompt",
        action="store_true",
        help="chat model",
    )
    parser.add_argument(
        "--stop",
        nargs='+',
        default=None,
        help="stop tokens",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="beam search num",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="generation temperature",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default='sglang',
        help="vllm or sglang",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor parallel",
    )
    args = parser.parse_args()

    main(args)
'''

accelerate launch test_math.py \
--model_path=/mntcephfs/data/med/zhanghongbo/MOSS/ckpts/ppo_all_new_huatuo_300K_self_train_it1/latest_tfmr  \
--input_path=/mntcephfs/data/med/zhanghongbo/huatuo/data_v2/chat_yaoshi/2021real_for_generate.json \
--output_path=/mntcephfs/data/med/zhanghongbo/huatuo/data_v2/chat_yaoshi/2021real_huatuo_answer_self_it1.json \
--num_return=5 \
--batch_size=2
'''