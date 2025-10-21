#!/bin/bash
#BSUB -P rl_evtols
#BSUB -J query                     # Job name
#BSUB -o logs/%J_query.log         # Stdout log
#BSUB -e logs/%J_query.err         # Stderr log
#BSUB -q normal                    # Queue name (adjust if needed)
#BSUB -W 8:00                      # Max runtime
#BSUB -n 1                         # Total cores/slots
#BSUB -gpu "num=0"                # No GPU required for client

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- Static Config ---------------------------------------------- #
# SERVER_HOST="<TTT server IP>"  # set to TTT server IP
ZMQ_PORT=5555

OUTPUT_DIR="/scratch/projects/rl_evtols/general-knowledge/results/query_server"
#OUTPUT_DIR="general-knowledge/results/query_server"
mkdir -p "${OUTPUT_DIR}"

# -------- Experiment Grid -------------------------------------------- #
# Columns: exp_name dataset  k  evalT  r  α  drop  ep  lr  bs  ga  n_articles
EXPERIMENTS=(
    "rank_iter0 /scratch/projects/rl_evtols/general-knowledge/data/synthetic_data/train/iter0_train.json  5  3  32  64  0  10  1e-3  1  1 50"
    #"rank_iter0 general-knowledge/data/synthetic_data/train/iter0_train.json  5  3  32  64  0  10  1e-3  1  1 50"
    # "eval_baseline general-knowledge/data/synthetic_data/eval/base_val.json  1  1  32  64  0  10  1e-3  1  1 200"
)

CHAIN_OF_THOUGHT=0  # whether to use chain of thought when answering
SPLIT_NEWLINES=1  # whether to split newlines into separate training documents
REWARD_MODE="ttt"  # "ttt", "proxy", or "both" for reward mode

# -------- Loop & Launch ---------------------------------------------- #
for EXP in "${EXPERIMENTS[@]}"; do
    read -r EXP_NAME DATASET K_COMPLETIONS EVAL_TIMES \
            LORA_RANK LORA_ALPHA LORA_DROPOUT \
            FINETUNE_EPOCHS FINETUNE_LR \
            BATCH_SIZE GRAD_ACC N_ARTICLES <<< "${EXP}"

    TAG=$(basename "${DATASET%.json}")_k${K_COMPLETIONS}_$((RANDOM))
    LOG_FILE="logs/${LSB_JOBID}_query_${TAG}.log"

    SN_FLAG=""
    if [[ "${SPLIT_NEWLINES}" == "true" ]]; then
        SN_FLAG="--split_newlines"
    fi

    COT_FLAG=""
    if [[ "${CHAIN_OF_THOUGHT}" == "1" ]]; then
        COT_FLAG="--chain_of_thought"
    fi

    echo "Query-server run: ${TAG}"
    python3 -u -m general-knowledge.src.query.query_server \
        --exp_name ${EXP_NAME} \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --server_host "${SERVER_HOST}" \
        --zmq_port "${ZMQ_PORT}" \
        --n_articles "${N_ARTICLES}" \
        --k_completions "${K_COMPLETIONS}" \
        --eval_times "${EVAL_TIMES}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --finetune_epochs "${FINETUNE_EPOCHS}" \
        --finetune_lr "${FINETUNE_LR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACC}" \
        --reward_mode "${REWARD_MODE}" \
        ${SN_FLAG} \
        ${COT_FLAG} \
        >> "${LOG_FILE}" 2>&1
done

echo "Job finished."
