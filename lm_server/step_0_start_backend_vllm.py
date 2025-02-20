from __future__ import annotations

import os
import argparse

from omegaconf import OmegaConf
from tqdm import tqdm
from mcts_math.config import BaseConfig
import subprocess
import time
from urllib.parse import urlparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args = args.parse_args()
    return args

def check_startup_complete(log_files, timeout=300):
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
    subprocess.run(f"pkill -f 'python3 -m vllm.entrypoints.openai.api_server'", shell=True)


def start_vllm_backend(args):
    """
    nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --chat-template $TEMPLATE_PATH \
    --port 9554 \
    > logs/vllm_openai_server.log 2>&1 &
    Application startup complete.
    """
    try:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        num_devices = len(cuda_visible_devices.split(',')) if cuda_visible_devices else 0
        print(f'{num_devices} gpus available.')
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

        policy_command = [
            f"CUDA_VISIBLE_DEVICES={policy_model_devices}",
            "nohup", "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model_dir,
            "--served-model-name", "policy_model",
            "--enforce-eager",
            "--port", str(policy_port),
            "--pipeline-parallel-size", str(policy_model_gpu_num)
        ]
        policy_log_file = f'logs/vllm_server_pol_{os.path.basename(args.model_dir)}.log'
        policy_command.append(f"> {policy_log_file} 2>&1 &")
        subprocess.run(' '.join(policy_command), shell=True)
        print(f"Policy model server started on GPU(s) {policy_model_devices} at port {policy_port}.")

        log_files_to_check = [policy_log_file]

        if args.need_ref_model:
            time.sleep(30)
            ref_command = [
                f"CUDA_VISIBLE_DEVICES={ref_model_devices}", "VLLM_ATTENTION_BACKEND=XFORMERS",
                "nohup", "python3", "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.ref_model_dir,
                "--served-model-name", "reference_model",
                "--enforce-eager",
                "--port", str(ref_port),
                "--gpu-memory-utilization", "0.7",
                "--pipeline-parallel-size", str(ref_model_gpu_num)
            ]
            ref_log_file = f'logs/vllm_server_ref_{os.path.basename(args.ref_model_dir)}.log'

            ref_command.append(f"> {ref_log_file} 2>&1 &")
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
    start_vllm_backend(config)



"""
python step_1_solver_demo.py --custom_cfg configs/step_mcts_round1.yaml --qaf MARIO_EVAL/data/math_trainset_annotation.json

"""