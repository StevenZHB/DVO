#%%
from vllm import LLM, SamplingParams
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
#%%
compute_capability = torch.cuda.get_device_capability()
float_type = "bfloat16"
torch_dtype = torch.bfloat16
attn_implementation = "flash_attention_2"
if compute_capability[0]<8:
    float_type = "float16"
    torch_dtype = torch.float16
    attn_implementation = None

GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0,1").split(',')
num_GPUS = len(GPUS)
split_num_GPUS = num_GPUS // 2
llm_GPUS = GPUS[:split_num_GPUS]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(_gpu) for _gpu in llm_GPUS])
llm_1 = LLM("/zhanghongbo/pretrained_models/Sheared-LLaMA-2.7B",max_model_len=50,gpu_memory_utilization=0.9,dtype=float_type,tensor_parallel_size=split_num_GPUS)

# %%

ref_GPUS = GPUS[-split_num_GPUS:]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(_gpu) for _gpu in ref_GPUS])
ref_llm = AutoModelForCausalLM.from_pretrained("/zhanghongbo/pretrained_models/Sheared-LLaMA-2.7B",trust_remote_code=True,torch_dtype=torch_dtype,attn_implementation=attn_implementation,device_map="balanced_low_0")
tokenizer = AutoTokenizer.from_pretrained("/zhanghongbo/pretrained_models/Sheared-LLaMA-2.7B")
# %%
samling_params = SamplingParams(logprobs=0,prompt_logprobs=0,max_tokens=3)
o = llm_1.generate("hello "*4096,samling_params)
# %%
inputs = {'input_ids':torch.tensor([[1,2,3,4,5,6,7]]),'labels':torch.tensor([[-100,-100,-100,-100,-100,-100,-100]])}

#%%
inputs = {k:v.to(ref_llm.device) for k,v in inputs.items()}
# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def reserve_max_memory(percentage=0.9):
    """
    查找当前所有 GPU，申请指定百分比的显存，然后释放以确保不占用显存。
    """
    torch.cuda.empty_cache()  # 清空缓存以获得最大的可用显存
    num_devices = torch.cuda.device_count()
    max_memories = {}

    for device in range(num_devices):
        total_memory = torch.cuda.get_device_properties(device).total_memory  # 获取总显存
        print(f"Device {device} - Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
        
        alloc_memory = total_memory * percentage  # 设定初始分配显存为总显存的指定百分比
        step = 1024 ** 3  # 以1GB为步长

        while alloc_memory > 0:
            try:
                # 尝试分配显存
                with torch.cuda.device(device):
                    tensor = torch.cuda.FloatTensor(int(alloc_memory // 4))  # 申请alloc_memory字节的显存
                    print_memory_usage()
                    del tensor  # 立即释放分配的显存
                break  # 如果分配成功，退出循环
            except RuntimeError:
                print_memory_usage()
                alloc_memory -= step  # 如果分配失败，减少分配大小
        
        max_memories[device] = alloc_memory
        print(f"Device {device} - Max allocatable GPU memory at {percentage*100}%: {alloc_memory / (1024 ** 3):.2f} GB")


# %%
reserve_max_memory(0.9)
# %%
CUDA_VISIBLE_DEVICES=6 nohup python3 -m vllm.entrypoints.openai.api_server \
--model /zhanghongbo/Super_MARIO_STEVEN/build_deepstepmath/llama3-70B-instruct-step-sft/llama3-70B-instruct-pattern-sft \
--served-model-name default_vllm_model \
--port 8805 \
--pipeline-parallel-size 8 \
--enforce-eager \
> default_vllm_model.log 2>&1 &

/zhanghongbo/Super_MARIO_STEVEN/build_deepstepmath/llama3-70B-instruct-step-sft/llama3-70B-instruct-pattern-sft
#%%
import openai
from utils_api_server import OpenAIModel
from transformers import AutoTokenizer
# client, model_name, sampling_params
api_base = "http://localhost:9554/v1"
api_key = "EMPTY"
model_name = "default_vllm_model"
sampling_params = {"temperature":1,"n":5, "logprobs":1,"stop": ["<｜end▁of▁sentence｜>","\n\n"],'max_tokens':1000}
client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)
models = client.models.list()
model_name = models.data[0].id
tok = AutoTokenizer.from_pretrained("/storage/zhanghongbo/pretrained_models/deepseek-math-7b-instruct")
llm = OpenAIModel(client=client,model_name=model_name,sampling_params=sampling_params,tokenizer=tok)
ref_sampling_params = {"temperature":1,"logprobs":5,"n":1,"max_tokens":1,"echo":True}
ref_llm = OpenAIModel(client=client,model_name=model_name,sampling_params=ref_sampling_params,tokenizer=tok)
# %%
batch_queries = ["hello","hi"]
# %%
batch_queries = ["<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe sum of three numbers $a, b$ and $c$ is 60. If we decrease $a$ by 7, we get the value $N$. If we increase $b$ by 7, we get the value $N$. If we multiply $c$ by 7, we also get the value $N$. What is the value of $N$?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"]
# batch_queries = ["The sum of three numbers $a, b$ and $c$ is 60. If we decrease $a$ by 7, we get the value $N$. If we increase $b$ by 7, we get the value $N$. If we multiply $c$ by 7, we also get the value $N$. What is the value of $N$?"]


# %%
import openai
from utils_api_server import OpenAIModel
# client, model_name, sampling_params
api_base = "http://localhost:8801/v1"
api_key = "EMPTY"
sg_client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)
models = sg_client.models.list()
model_name = models.data[0].id
sampling_params = {"temperature":1,"n":1}
sampling_params = {'temperature': 0, 'echo':True,'top_p': 1.0, 'best_of': 1, 'max_tokens': 1000, 'n': 1, 'stop': ['<|eot_id|>', '\n\n'], 'logprobs': 1}
sglang_llm = OpenAIModel(client=sg_client,model_name=model_name,sampling_params=sampling_params,tokenizer=None)

#%%
api_base = "http://localhost:9554/v1"
api_key = "EMPTY"
vllm_client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)
models = vllm_client.models.list()
model_name = models.data[0].id
vllm_llm = OpenAIModel(client=vllm_client,model_name=model_name,sampling_params=sampling_params,tokenizer=None)
# %%
batch_queries = ["<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe sum of three numbers $a, b$ and $c$ is 60. If we decrease $a$ by 7, we get the value $N$. If we increase $b$ by 7, we get the value $N$. If we multiply $c$ by 7, we also get the value $N$. What is the value of $N$?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"]
# %%
import datasets
d = datasets.load_from_disk("/zhanghongbo/cuihan/Super_MARIO_TINY/generated_data/deepseekmath-instruct/for_beta_scan")
# %%
d_train = d["train_prefs"]
# %%
