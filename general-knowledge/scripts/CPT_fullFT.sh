#!/bin/bash
#SBATCH --job-name=cpt
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
MODEL="Qwen/Qwen2.5-7B"
MAXLEN=2048
EVALTOK=64
SPLIT_NEWLINES=0

# -------- Hyperparams via Slurm array ------------------------------- #
# Columns:
#  TAG  DATASET  K_COMP  EPOCHS  LR  BS  GA  N_ARTICLES  BASELINE_EVAL
EXPERIMENTS=(
    "base general-knowledge/data/synthetic_data/eval/base_val.json  5 1 7e-5 4 2 200 1"
    "base general-knowledge/data/synthetic_data/eval_full/base_val.json  5 1 7e-5 4 2 2067 1"
)

EXP="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

read -r TAG DATASET K_COMP EPOCHS LR BS GA N_ARTICLES BASELINE_EVAL <<< "${EXP}"

LOG_FILE="logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}.log"
echo "Run tag: ${TAG}" | tee -a "$LOG_FILE"

SN_FLAG=""
if [[ "${SPLIT_NEWLINES}" == "1" ]]; then
    SN_FLAG="--split_newlines"
fi

BASELINE_FLAG=""
if [[ "${BASELINE_EVAL}" == "1" ]]; then
    BASELINE_FLAG="--baseline_eval"
fi

python -u -m general-knowledge.src.query.CPT_fullFT \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --k_completions "${K_COMP}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --batch_size "${BS}" \
    --gradient_accumulation_steps "${GA}" \
    --max_seq_length "${MAXLEN}" \
    --eval_max_tokens "${EVALTOK}" \
    --bf16 \
    --tf32 \
    --n_articles "${N_ARTICLES}" \
    --eval_question_limit 500 \
    ${SN_FLAG} \
    ${BASELINE_FLAG} \
    >> "${LOG_FILE}" 2>&1
