#!/bin/bash
#SBATCH --job-name=request
#SBATCH --output=logs/%A_%x.log
#SBATCH --gres=gpu:0

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

set -a
source .env
set +a

# --------------------------------------------------------------------- #
python general-knowledge/src/data_generation/make_squad_data_openai.py \
    --dataset_in general-knowledge/data/squad_val.json \
    --dataset_out general-knowledge/data/synthetic_data/eval/gpt4_1_val.json \
    --n 200

echo "Job finished."
