from __future__ import annotations

import os
import argparse
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from mcts_math.config import BaseConfig
import subprocess
import time
from urllib.parse import urlparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument('--tp', type=int, default=None)
    args.add_argument('--quantization', action='store_true', default=False)
    args = args.parse_args()
    return args

def check_startup_complete(log_files, timeout=6000):
    start_time = time.time()
    found = {log_file: False for log_file in log_files}

    while time.time() - start_time < timeout and not all(found.values()):
        for log_file in log_files:
            if not found[log_file]:
                try:
                    with open(log_file, 'r') as file:
                        if "Application startup complete." in file.read():
                            found[log_file] = True
                except FileNotFoundError:
                    pass
        time.sleep(10)

    return all(found.values())

def cleanup():
    print("Cleaning up...")
    subprocess.run(f"pkill -9 -f 'python3 -m sglang.launch_server'", shell=True)


def start_vllm_backend(args):
    """
    nohup python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --chat-template $TEMPLATE_PATH \
    --port 9554 \
    > logs/vllm_openai_server.log 2>&1 &
    Application startup complete.
    """
    try:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
        num_devices = len(cuda_visible_devices.split(',')) if cuda_visible_devices else 0
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0]==9:
            enable_fp8 = True
        else:
            enable_fp8 = False
        print(f'{num_devices} gpus available.')
        if args.need_ref_model:
            assert num_devices>=2
        all_devices = cuda_visible_devices.split(',')

        if args.need_ref_model:
            ref_model_gpu_num = max(num_devices//3,1)
            ref_model_devices = ','.join(all_devices[-ref_model_gpu_num:])

            policy_model_gpu_num = num_devices-ref_model_gpu_num
            policy_model_devices = ','.join(all_devices[:policy_model_gpu_num])

            policy_port = urlparse(args.model_api).port
            ref_port = urlparse(args.ref_model_api).port
        else:
            policy_model_gpu_num = num_devices
            policy_model_devices = ','.join(all_devices)
            policy_port = urlparse(args.model_api).port

        # 清除代理相关的环境变量
        proxy_env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
        for var in proxy_env_vars:
            if var in os.environ:
                del os.environ[var]  # 删除环境变量
        print(f"tp value: {args.tp}")
        if args.tp is None:
            dp = str(policy_model_gpu_num)
            tp = '1'
        else:
            dp = str(int(policy_model_gpu_num//args.tp))
            tp = str(args.tp)

        policy_command = [
            f"CUDA_VISIBLE_DEVICES={policy_model_devices}",
            "nohup", "python3", "-m", "sglang.launch_server",
            "--model-path", args.model_dir,
            "--port", str(policy_port),
            "--dp", dp,
            "--tp", tp,
            "--mem-fraction-static", "0.9",
            "--max-prefill-tokens", "12800",
            "--disable-cuda-graph",
            # "--disable-flashinfer-sampling",
            "--dtype", "half",
            "--host","0.0.0.0",
            "--disable-cuda-graph"
        ]
        if args.quantization:
            policy_command.append("--quantization")
            policy_command.append("fp8")
        policy_log_file = f'logs/sglang_server_pol_{os.path.basename(args.model_dir)}.log'
        policy_command.append(f"> {policy_log_file} 2>&1 &")
        print(" ".join(policy_command))
        subprocess.run(' '.join(policy_command), shell=True)
        print(f"Policy model server started on GPU(s) {policy_model_devices} at port {policy_port}.")

        log_files_to_check = [policy_log_file]

        if args.need_ref_model:
            time.sleep(10)
            if args.tp is None:
                dp = str(ref_model_gpu_num)
                tp = 'None'
            else:
                dp = str(int(ref_model_gpu_num//args.tp))
                tp = str(args.tp)
            ref_command = [
                f"CUDA_VISIBLE_DEVICES={ref_model_devices}", "VLLM_ATTENTION_BACKEND=XFORMERS",
                "nohup", "python3", "-m", "sglang.launch_server",
                "--model-path", args.ref_model_dir,
                "--port", str(ref_port),
                "--dp", dp,
                "--dp", dp,
                "--mem-fraction-static", "0.9",
                "--max-prefill-tokens", "12800",
                "--disable-cuda-graph",
                # "--disable-flashinfer-sampling",
                "--disable-cuda-graph",
                "--host","0.0.0.0",
                # "--quantization fp8"
            ]
            if enable_fp8:
                ref_command.append("--quantization")
                ref_command.append("fp8")
            ref_log_file = f'logs/sglang_server_ref_{os.path.basename(args.ref_model_dir)}.log'

            ref_command.append(f"> {ref_log_file} 2>&1 &")
            print(" ".join(ref_command))
            subprocess.run(' '.join(ref_command), shell=True)
            print(f"Reference model server started on GPU(s) {ref_model_devices} at port {ref_port}.")
            log_files_to_check.append(ref_log_file)


        if check_startup_complete(log_files_to_check):
            print("Both servers started successfully.")
        else:
            print("Failed to start one or both servers within the timeout period.")
            raise RuntimeError("Server startup failed; processes will be terminated at exit.")
    except KeyboardInterrupt:
        print("Program was interrupted by user (Ctrl+C).")
        cleanup()
    except Exception as e:
        print("An error occurred:", e)
        cleanup()



if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    config.tp = args.tp
    config.quantization = args.quantization
    start_vllm_backend(config)



"""
python step_1_solver_demo.py --custom_cfg configs/step_mcts_round1.yaml --qaf MARIO_EVAL/data/math_trainset_annotation.json

"""