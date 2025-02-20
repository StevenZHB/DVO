#!/bin/bash

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 config_file"
    exit
fi

# run command
config_file=$1


python search_sampling/sample_output.py\
    --custom_cfg $config_file

