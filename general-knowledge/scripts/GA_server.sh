#!/bin/bash
#SBATCH --job-name=GAserver
#SBATCH --output=logs/%A_%a_%x.log
#SBATCH --gres=gpu:1

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- User-editable ---------------------------------------------- #
# Note: this script requires setting your HF_TOKEN in the .env file
MODEL_NAME="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"  # HF model name or path to RL checkpoint (e.g. models/iter1)
ZMQ_PORT=5555

MAX_SEQ_LENGTH=2048  # Max sequence length for training
EVAL_MAX_TOKENS=64  # Max generated tokens for evaluation completions
EVAL_TEMPERATURE=0.0
EVAL_TOP_P=1.0
# --------------------------------------------------------------------- #
mkdir -p logs
echo "Launching GA server on $(hostname) ($(hostname -i))..."

set -a
source .env
set +a

echo "Starting Inner Loop server..."
python3 -m general-knowledge.src.inner.GA_server \
    --model "${MODEL_NAME}" \
    --zmq_port ${ZMQ_PORT} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --eval_max_tokens ${EVAL_MAX_TOKENS} \
    --eval_temperature ${EVAL_TEMPERATURE} \
    --eval_top_p ${EVAL_TOP_P} \
    > logs/${SLURM_JOB_ID}_GA_server.log 2>&1 &

ZMQ_PID=$!
echo "    Inner Loop Server started with PID ${ZMQ_PID}."
echo "Ready to accept requests on port ${ZMQ_PORT}."
wait

echo "Job finished."
