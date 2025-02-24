#!/bin/bash

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 config_file data_file total_buckets"
    exit
fi

export PYTHONPATH=./:$PYTHONPATH
# run command
config_file=$1
data_file=$2
total_buckets=$3

config_dir=$(dirname "$config_file")
parent_folder_name=$(basename "$config_dir")

for ((i=0; i<total_buckets; i++))
do
    echo "Starting bucket $i of $total_buckets..."
    log_file="logs/${parent_folder_name}_bucket_${i}_of_${total_buckets}.log"
    echo ""
    # CUDA_VISIBLE_DEVICES=$i nohup python step_1_solver_demo_new.py --custom_cfg "$config_file" --qaf "$data_file" --bucket "${i},${total_buckets}" > "$log_file" 2>&1 &
    command="CUDA_VISIBLE_DEVICES=$i nohup python search_sampling/mcts_main.py --custom_cfg \"$config_file\" --qaf \"$data_file\" --bucket \"${i},${total_buckets}\" > \"$log_file\" 2>&1 &"
    # Print the command
    echo "Executing command: $command"
    # Execute the command
    eval $command
    echo "Sleeping for 30 seconds..."
    sleep 30
done

echo "All $total_buckets buckets are running in parallel."