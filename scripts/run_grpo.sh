#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <deepspeed_stage> <model_config_name>"
    exit 1
fi

stage=$1
config=$2

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file ../configs/accelerate/"$stage"_config.yaml \
    ../src/train/grpo.py ../configs/train/sft/"$config".toml