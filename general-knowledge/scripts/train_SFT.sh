#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%A_%x.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- User-editable ---------------------------------------------- #
MODEL_NAME="Qwen/Qwen2.5-7B"  # Put the (n-1)'th RL checkpoint. This script then trains the n'th checkpoint. The 0'th checkpoint is the base model.
TRAIN_FILE="general-knowledge/data/synthetic_data/EM_SFT/sft_best1of5_iter0.jsonl"  # Path to training data output by src/EM/build_SFT_dataset.py
OUTPUT_DIR="models/iter1"
mkdir -p "${OUTPUT_DIR}"

PER_DEVICE_BATCH_SIZE=1
GRAD_ACC=5
EPOCHS=2
LR=3e-4
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LOG_STEPS=1
# --------------------------------------------------------------------- #

# export NCCL_P2P_DISABLE=1  # fixes hangs on some setups

echo "Launching SFT run on $(hostname)..."
accelerate launch \
    --num_processes 2 \
    --deepspeed_config_file general-knowledge/src/EM/config/deepspeed_stage3.json \
    --mixed_precision bf16 \
    general-knowledge/src/EM/train_SFT.py \
    --train_file "${TRAIN_FILE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --logging_steps ${LOG_STEPS}

echo "Job finished."
