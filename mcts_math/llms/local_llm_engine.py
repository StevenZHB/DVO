import os, sys
import os.path as osp
import time
import numpy as np
import pynvml
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import openai
# from utils_api import OpenAIModel
from .utils_api import OpenAIModel
logger = logging.getLogger(__name__)

TIMEOUT_PROCESS = 1800
TIME_SPAN_LLM = 0.5
MAX_SOLUTIONS_COUNT = 24

# LLM check time
CHECK_INTERVAL = 60 # 1800  # half hour
UNIT = 1024**3

BAR_TIME = 30 # 30


def llm_init(config):
    GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    llm = LLM(
        model=config.model_dir,
        tensor_parallel_size=len(GPUS),
        trust_remote_code=True,
        seed=config.seed,
        swap_space=config.swap_space
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        use_beam_search=config.use_beam_search,
        best_of=config.best_of,
        max_tokens=config.max_tokens,
        n=config.n_generate_sample,
        stop=config.stop,
        logprobs=0,
        prompt_logprobs=0,
        #seed=config.seed,
    )
    return llm, sampling_params

def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def ref_llm_server_forward(ref_model, ref_sampling_params, outputs):
    for out_index, output in enumerate(outputs):
        response = output.choices[0].text
        q_pi = sum(output.choices[0].logprobs.token_logprobs)
        response_tokens_len = len(output.choices[0].logprobs.tokens)-2 # cut out the "\n\n" at the end
        forward_input = output.prompt + response
        reference_output = ref_model.generate(forward_input, ref_sampling_params)[0]
        reference_logprobs = reference_output.prompt_logprobs[(-1) * response_tokens_len:]
        q_ref = sum([list(token.values())[0].logprob for token in reference_logprobs])
        if q_ref == 0 or response == "":
            output.value_estimate = None
        else:
            output.value_estimate = q_pi - q_ref
    return outputs


def ref_llm_forward(model,pad_token_id,batch_size, outputs):
    """
    forward and get the logprobs of outputs span
    outputs: list[RequestOutput]
    """
    inputs = []
    labels = []
    labels_length = []

    output_map = []  # To keep track of the mapping to original output structure
    # Flatten all outputs for processing
    for out_index, out in enumerate(outputs):
        inp_token_ids = out.prompt_token_ids
        for output in out.outputs:
            out_token_ids = list(output.token_ids)
            inputs.append(inp_token_ids + out_token_ids)
            label = [-100] * len(inp_token_ids) + out_token_ids
            labels.append(label)
            labels_length.append(len(out_token_ids))
            output_map.append(out_index)


    sum_log_probs = []
    original_structure = [[] for _ in outputs]
    for (batch_inputs, batch_labels, batch_labels_length) in zip(batch(inputs,batch_size), batch(labels,batch_size),batch(labels_length,batch_size)):
        max_length = max(len(ids) for ids in batch_inputs)
        batch_inputs_padded = [ids + [pad_token_id] * (max_length - len(ids)) for ids in batch_inputs]
        batch_labels_padded = [ids + [-100] * (max_length - len(ids)) for ids in batch_labels]
        input_tensors = torch.tensor(batch_inputs_padded).to(model.device)
        label_tensors = torch.tensor(batch_labels_padded).to(model.device)
        batch_labels_length = torch.tensor(batch_labels_length)
        with torch.no_grad():
            output = model(input_tensors, labels=label_tensors, return_dict=True)
            losses = output.loss.cpu()
            logits = output.logits

        label_tensors = label_tensors[:, 1:].clone() # (bs * msn * sl * vs)
        logits = logits[:, :-1, :]
        loss_mask = label_tensors != -100
        label_tensors[label_tensors == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=label_tensors.unsqueeze(2)).squeeze(2)
        sum_log_prob = (per_token_logps * loss_mask).sum(-1).cpu()

        sum_log_prob = sum_log_prob.tolist()
        sum_log_probs.extend(sum_log_prob)

    for log_prob, map_index in zip(sum_log_probs, output_map):
        original_structure[map_index].append(log_prob)

    for output,ref_output in zip(outputs,original_structure):
        for o,r in zip(output.outputs,ref_output):
            if r == 0:
                o.value_estimate = None
            else:
                o.value_estimate = o.cumulative_logprob - r
                o.q_ref = r
                o.q_pi = o.cumulative_logprob

    return outputs


def get_sampling_params(config):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        use_beam_search=config.use_beam_search,
        best_of=config.best_of,
        max_tokens=config.max_tokens,
        n=config.n_generate_sample,
        stop=config.stop,
        #seed=config.seed,
    )

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.ref_gpu)

    float_type = "bfloat16"
    torch_dtype = torch.bfloat16
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0]<8:
        float_type = "float16"
        torch_dtype = torch.float16

    ref_llm = LLM(
        model=config.ref_model_dir,
        # pipeline_parallel_size=split_num_GPUS,
        trust_remote_code=True,
        seed=config.seed,
        swap_space=config.swap_space,
        dtype = torch_dtype,
        tensor_parallel_size=1
        # gpu_memory_utilization=gpu_util
    )

    ref_sampling_params = SamplingParams(
        temperature=0,
        top_k=-1,
        top_p=1,
        max_tokens=1,
        n=1,
        logprobs=1,
        prompt_logprobs=1
    )

    return sampling_params, ref_llm, ref_sampling_params


def bi_llm_engine(config):
    GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0,1")
    GPUS = GPUS.split(',')
    num_GPUS = len(GPUS)
    float_type = "bfloat16"
    torch_dtype = torch.bfloat16
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0]<8:
        float_type = "float16"
        torch_dtype = torch.float16
    if config.need_ref_model:
        if num_GPUS > 1:
            available_GPUS = list(range(len(GPUS)))
            split_num_GPUS = num_GPUS // 2
            gpu_util = 0.85
            ref_GPUS = available_GPUS[-split_num_GPUS:]
            ref_gpu_list = {i: torch.cuda.mem_get_info(i)[0] for i in ref_GPUS}
        else:
            available_GPUS = list(range(len(GPUS)))
            split_num_GPUS = 1
            gpu_util = 0.6
            ref_GPUS = available_GPUS[-1:]
            ref_gpu_list = {i: torch.cuda.mem_get_info(i)[0]*0.4 for i in ref_GPUS}

        # 加载模型并放置到指定的 GPU 上
        ref_llm = AutoModelForCausalLM.from_pretrained(
            config.ref_model_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="sequential",
            max_memory=ref_gpu_list
        )

        ref_llm.eval()
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPUS)
    else:
        split_num_GPUS = num_GPUS
        gpu_util = 0.85
        ref_llm=None
    llm = LLM(
            model=config.model_dir,
            pipeline_parallel_size=split_num_GPUS,
            trust_remote_code=True,
            seed=config.seed,
            swap_space=config.swap_space,
            dtype = float_type,
            gpu_memory_utilization=gpu_util
        )
    sampling_params = SamplingParams(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            use_beam_search=config.use_beam_search,
            best_of=config.best_of,
            max_tokens=config.max_tokens,
            n=config.n_generate_sample,
            stop=config.stop,
            #seed=config.seed,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.ref_model_dir,
        trust_remote_code=True
    )

    return llm,sampling_params,ref_llm,tokenizer



def llm_engine(config):
    llm, sampling_params = llm_init(config)
    return llm, sampling_params


## For server
def bi_llm_server(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_dir,
        trust_remote_code=True
    )
    policy_client = openai.OpenAI(
            api_key="API_KEY",
            base_url = config.model_api
        )
    ref_client = openai.OpenAI(
            api_key="API_KEY",
            base_url = config.ref_model_api
        )
    policy_sampling_params = {
        "temperature": config.temperature,
        # "top_k": config.top_k,
        "top_p": config.top_p,
        # "use_beam_search": config.use_beam_search,
        "best_of": config.best_of,
        "max_tokens": config.max_tokens,
        "n": config.n_generate_sample,
        "stop": list(config.stop),
        "logprobs": 1,
    }
    ref_sampling_params = {
        "temperature": 0,
        # "top_k": config.top_k,
        "top_p": config.top_p,
        # "use_beam_search": config.use_beam_search,
        "best_of": 1,
        "max_tokens": 1,
        "echo": True,
        "n": 1,
        "stop": list(config.stop),
        "logprobs": 1
    }
    # print(policy_sampling_params)
    # print(ref_sampling_params)

    policy_model = OpenAIModel(client=policy_client,model_name=config.model_dir,sampling_params=policy_sampling_params,tokenizer=tokenizer)
    ref_model = OpenAIModel(client=ref_client,model_name=config.ref_model_dir,sampling_params=ref_sampling_params,tokenizer=tokenizer)
    return policy_model, policy_sampling_params, ref_model, ref_sampling_params, tokenizer

def local_generator(
    prompts: List[str],
    sampling_params: SamplingParams,
    engine: LLM,
    use_tqdm: True
):

    outputs = engine.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)    # return List[RequestOutput]
    return outputs