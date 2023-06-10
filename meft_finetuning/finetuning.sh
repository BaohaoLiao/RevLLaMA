#!/bin/bash

source activate llama_adapter

TARGET_FOLDER=/ivi/ilps/projects/ltl-mt/llama
DATA_PATH=/ivi/ilps/personal/bliao/data/stanford_alpaca
SAVE_DIR=/ivi/ilps/personal/bliao/llama_adapter/01_revllama/checkpoint

torchrun --nproc_per_node 1 finetuning.py \
    --model RevLlama7B \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_PATH/alpaca_data.json \
    --adapter_layer 32 \
    --x1_factor 0.1 \
    --x2_factor 1 \
    --sum_factor 0 \
    --finetune_output_layer \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $SAVE_DIR
