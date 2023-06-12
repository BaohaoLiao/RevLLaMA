#!/bin/bash

source activate llama_adapter

TARGET_FOLDER=/ivi/ilps/projects/ltl-mt/llama
ADAPTER_FOLDER=/ivi/ilps/personal/bliao/llama_adapter/01_revllama/not_tune_output


torchrun --nproc_per_node 1 example.py \
     --ckpt_dir $TARGET_FOLDER/7B \
     --tokenizer_path $TARGET_FOLDER/tokenizer.model \
     --adapter_dir $ADAPTER_FOLDER