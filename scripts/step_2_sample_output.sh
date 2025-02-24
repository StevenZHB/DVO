#!/bin/bash

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 config_file qaf_file"
    exit
fi

export PYTHONPATH=./:$PYTHONPATH

# run command
config_file=$1
qaf_file=$2

python search_sampling/sample_output.py\
    --custom_cfg $config_file\
    --qaf $qaf_file

