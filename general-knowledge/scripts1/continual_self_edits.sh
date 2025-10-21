#scripts1/continual_self_edits.sh
#!/bin/bash

# Kill any existing servers before starting
echo "Cleaning up any existing processes..."
pkill -f "vllm serve"
pkill -f "TTT_server"
sleep 2

eval "$(conda shell.bash hook)"
# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL
mkdir -p general-knowledge/results/continual_self_edits
mkdir -p models


# -------- User-editable ---------------------------------------------- #
INDEX=11  # Index for this job, used to differentiate runs

MODEL_NAME="Qwen/Qwen2.5-7B"   # initialized model. Use the last RL checkpoint
DATASET="general-knowledge/data/squad_val.json"  # evaluation dataset


OUTPUT_DIR="general-knowledge/results/continual_self_edits/run${INDEX}"
mkdir -p "${OUTPUT_DIR}"

# LoRA / tuning hyperparameters (matches: r α drop ep lr bs ga)
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
FINETUNE_EPOCHS=10
FINETUNE_LR=1e-3
BATCH_SIZE=1
GRAD_ACC=1

# Infrastructure layout
VLLM_SERVER_GPUS="0"       # GPU(s) for vLLM server (comma-sep)
PY_DRIVER_GPU="1"          # GPU on which the continual self-edit script runs
PORT=$((8001 + INDEX))     # vLLM HTTP port (unique per node)
ZMQ_PORT=$((5555 + INDEX)) # ZMQ port if driver spawns an inner server
SEED=$((42 + INDEX))

MAX_TOKENS=8192            # self-edit generation cap
TEMPERATURE=0            # self-edit sampling temperature
top_p=0.95                 # self-edit top-p

N_SEQUENCES=12              # number of sequence to average over
N_DATAPOINTS=8             # datapoints per sequence
# --------------------------------------------------------------------- #

# Safety mechanism parameters
SAFETY_MODE="lora_clip"     # Options: lora_clip, gate_only           ***didn't use gate_only in this experiment
FORGET_METRIC="em"        # Options: em, bits, kl

# Bits metric configuration
BITS_MODE="average"      # Options: total, average               **didn't use total in this experiment
BITS_MAX_TOTAL=50.0        # Max total bits increase (for bits_mode=total)
BITS_MAX_AVERAGE=0.08       # Max bits/token (for bits_mode=average)  

# Other metrics 
EM_MAX_DROP=0.07           # Max allowed EM drop
KL_MAX_BITS=0.7           # Max KL divergence


# LoRA clipping parameters
CLIP_BISECT_STEPS=5       # Binary search steps for lora_clip
CLIP_MIN_SCALE=0.1        # Minimum scale for lora_clip
# --------------------------------------------------------------------- #

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}

# -------- Launch Driver ---------------------------------------------- #
echo "Starting continual self-edits driver on GPU ${PY_DRIVER_GPU}"

START_TIME=$(date +%s)


python3 -u -m general-knowledge.src.continual.continual_self_edits \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --finetune_epochs ${FINETUNE_EPOCHS} \
    --finetune_lr ${FINETUNE_LR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --n_sequences ${N_SEQUENCES} \
    --n_datapoints ${N_DATAPOINTS} \
    --output_dir "${OUTPUT_DIR}" \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${top_p} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    --safety_mode ${SAFETY_MODE} \
    --forget_metric ${FORGET_METRIC} \
    --bits_mode ${BITS_MODE} \
    --em_max_drop ${EM_MAX_DROP} \
    --bits_max_total ${BITS_MAX_TOTAL} \
    --bits_max_average ${BITS_MAX_AVERAGE} \
    --kl_max_bits ${KL_MAX_BITS} \
    --clip_bisect_steps ${CLIP_BISECT_STEPS} \
    --clip_min_scale ${CLIP_MIN_SCALE}


STATUS=$?
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
if [ $STATUS -ne 0 ]; then
    echo "Python crashed with exit code $STATUS"
else
    echo "Job finished successfully."
fi

echo "Total runtime: $RUNTIME seconds (~$((RUNTIME / 60)) minutes)"
