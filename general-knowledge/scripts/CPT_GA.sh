#!/bin/bash
#SBATCH --job-name=cptGA
#SBATCH --output=logs/%A_%a_%x.log
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

OUTPUT_DIR="general-knowledge/results/cpt"
mkdir -p "${OUTPUT_DIR}"

set -a
source .env
set +a

# -------- Shared hyperparams ---------------------------------------------- #
# Note: this script requires setting your HF_TOKEN in the .env file
MODEL_NAME="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"  # HF model name or path to RL checkpoint (e.g. models/iter1)

# -------- Hyperparams via Slurm array ------------------------------- #
# Columns:
#  TAG  DATASET  N_ARTICLES  BASELINE_EVAL
EXPERIMENTS=(
    "base  general-knowledge/data/synthetic_data/eval/base_val.json  200 1"
)

EXP="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

read -r TAG DATASET N_ARTICLES BASELINE_EVAL <<< "${EXP}"

LOG_FILE="logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}.log"
echo "Run tag: ${TAG}" | tee -a "$LOG_FILE"

BASELINE_FLAG=""
if [[ "${BASELINE_EVAL}" == "1" ]]; then
    BASELINE_FLAG="--baseline_eval"
fi

python -u -m general-knowledge.src.query.CPT_GA \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL_NAME}" \
    --tf32 \
    --n_articles "${N_ARTICLES}" \
    --eval_question_limit 500 \
    ${BASELINE_FLAG} \
    >> "${LOG_FILE}" 2>&1
