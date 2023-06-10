#!/bin/bash

## SLURM setting
#SBATCH --job-name=llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:2 #volta, pascal or p40
##SBATCH --nodelist=ilps-cn117
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=4-10

source activate llama_adapter

TARGET_FOLDER=/ivi/ilps/projects/ltl-mt/llama
DATA_PATH=/ivi/ilps/personal/bliao/data/stanford_alpaca
SAVE_DIR=/ivi/ilps/personal/bliao/llama_adapter/00_reproduce/checkpoints

torchrun --nproc_per_node 2 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_PATH/alpaca_data.json \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 4 \
    --epochs 5 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $SAVE_DIR/ 2>&1 | tee $SAVE_DIR/out

