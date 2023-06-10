#!/bin/bash

source activate llama_adapter

TARGET_FOLDER=/ivi/ilps/projects/ltl-mt/llama
DATA_PATH=/ivi/ilps/personal/bliao/data/stanford_alpaca
SAVE_DIR=/ivi/ilps/personal/bliao/llama_adapter/01_revllama/checkpoint

torchrun --nproc_per_node 8 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_PATH/alpaca_data.json \
    --adapter_layer 32 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $SAVE_DIR
