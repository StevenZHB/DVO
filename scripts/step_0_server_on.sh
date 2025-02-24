#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH


mkdir -p ./logs

CONFIG_FILE=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python lm_server/step_0_start_backend_sglang.py --custom_cfg "$CONFIG_FILE"
