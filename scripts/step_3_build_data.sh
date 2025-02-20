#! /bin/bash

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 data_path threshold(Optional) max_choosing_num(Optional) system_prompt(Optional)"
    exit
fi


data_path=$1
if [ ! -z "$2" ]; then
    threshold=$2
fi

if [ ! -z "$3" ]; then
    max_choosing_num=$3
fi

if [ ! -z "$4" ]; then
    system_prompt=$4
fi


# run command
# Build the command string first - eval will execute the final command string
# eval is needed to properly handle the quotes and spaces in the arguments
command="python search_sampling/build_data.py --data_path \"$data_path\""

if [ ! -z "$threshold" ]; then
    command="$command --threshold \"$threshold\""
fi

if [ ! -z "$max_choosing_num" ]; then
    command="$command --max_choosing_num \"$max_choosing_num\""
fi

if [ ! -z "$system_prompt" ]; then
    command="$command --system_prompt \"$system_prompt\""
fi

# eval executes the command string as a shell command
# This ensures proper handling of quotes and spaces in arguments
eval $command
