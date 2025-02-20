#!/bin/bash


MODEL_PATH=$1
TEMPLATE_PATH=$2

python eval/test_math_vllm.py --model_path "$MODEL_PATH" --template_path "$TEMPLATE_PATH"

python eval/test_reasoning_vllm.py --model_path "$MODEL_PATH" --template_path "$TEMPLATE_PATH"