#!/bin/bash
#BSUB -P rl_evtols
#BSUB -J server                     # Job name
#BSUB -o logs/%J_server.log         # Stdout log (%J = LSF Job ID)
#BSUB -e logs/%J_server.err         # Stderr log
#BSUB -q gpu_h100                     # Queue name (adjust if needed)
#BSUB -W 8:00                       # Max runtime: 8 hours
#BSUB -n 2                          # Total cores/slots
#BSUB -gpu "num=2"                  # Request 2 GPUs
#BSUB -R "rusage[mem=48000]"        # Memory request per node (16GB here)

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
#module load mambaforge
conda activate seal_env
cd ~/SEAL

# -------- User-editable ---------------------------------------------- #
MODEL_NAME="Qwen/Qwen2.5-7B"
VLLM_SERVER_GPUS="0"
INNER_LOOP_GPU="1"
PORT=8001
ZMQ_PORT=5555

MAX_SEQ_LENGTH=2048
EVAL_MAX_TOKENS=64
EVAL_TEMPERATURE=0.0
EVAL_TOP_P=1.0

MAX_LORA_RANK=32
# --------------------------------------------------------------------- #
echo "Launching TTT server on $(hostname)..."

set -a
source .env
set +a

VLLM_HOST=$(hostname -i)
VLLM_API_URL="http://${VLLM_HOST}:${PORT}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "Launching vLLM on GPU ${VLLM_SERVER_GPUS}"
CUDA_VISIBLE_DEVICES=${VLLM_SERVER_GPUS} vllm serve "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port ${PORT} \
    --max-model-len ${MAX_SEQ_LENGTH} \
    --enable-lora \
    --max-lora-rank ${MAX_LORA_RANK} \
    --trust-remote-code \
    > "logs/${LSB_JOBID}_vllm_server.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM..."
until curl --silent --fail ${VLLM_API_URL}/health >/dev/null; do sleep 3; done
echo "    vLLM ready at ${VLLM_API_URL}"

echo "Starting Inner Loop server on GPU ${INNER_LOOP_GPU}..."
CUDA_VISIBLE_DEVICES=${INNER_LOOP_GPU} python3 -m general-knowledge.src.inner.TTT_server \
    --vllm_api_url "${VLLM_API_URL}" \
    --model "${MODEL_NAME}" \
    --zmq_port ${ZMQ_PORT} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --eval_max_tokens ${EVAL_MAX_TOKENS} \
    --eval_temperature ${EVAL_TEMPERATURE} \
    --eval_top_p ${EVAL_TOP_P} \
    > logs/${LSB_JOBID}_TTT_server.log 2>&1 &

ZMQ_PID=$!
echo "    Inner Loop Server started with PID ${ZMQ_PID}."
echo "Ready to accept requests on port ${ZMQ_PORT}."

trap "echo 'Shutting down...'; kill ${ZMQ_PID} ${VLLM_PID}" EXIT
wait

echo "Job finished."