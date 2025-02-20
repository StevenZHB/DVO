# 检查输入参数
if [ $# -lt 1 ]; then
    echo "Please follow the usage:"
    echo "    bash $0 config_file [mode]"
    echo "    mode: 2 or 3 (optional, default is 2)"
    exit
fi

# 获取输入参数
input_config_file=$1
mode=${2:-2}

# You Could Modify the config file according to your own setting
if [ "$mode" == "1" ]; then
    config_file="configs/accelerate_configs/8_gpus/deepspeed_zero_1_off.yaml"
elif [ "$mode" == "2" ]; then
    config_file="configs/accelerate_configs/8_gpus/deepspeed_zero_2_off.yaml"
elif [ "$mode" == "3" ]; then
    config_file="configs/accelerate_configs/8_gpus/deepspeed_zero_3_no_off.yaml"
else
    echo "Invalid mode: $mode. Please use 2 or 3."
    exit
fi


base_name=$(basename "$input_config_file" .yaml)

parent_dir=$(basename "$(dirname "$input_config_file")")

job_name="${parent_dir}_${base_name}"

command="ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file $config_file \
    training/run_dvo.py $input_config_file"

# 打印命令
echo "Running command: $command"

# 使用 nohup 运行命令并将输出重定向到日志文件
nohup bash -c "$command" > "./logs/training_${job_name}.log" 2>&1 &
