

if [ "$1" = "sglang" ]; then
    SERVER_TYPE="$1"
else
    SERVER_TYPE="vllm"
fi

CONFIG_FILE="configs/beam_search/beam_search.yaml"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [ "$SERVER_TYPE" = "sglang" ]; then
    python lm_server/step_0_start_backend_sglang.py --custom_cfg "$CONFIG_FILE"
else
    python lm_server/step_0_start_backend_vllm.py --custom_cfg "$CONFIG_FILE"
fi