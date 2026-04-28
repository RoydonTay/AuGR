#!/bin/bash
# =============================================================================
# Avazu CTR training launcher
# Usage:
#   bash run_train_avazu_ctr_v1.sh <exp>
# Example:
#   bash run_train_avazu_ctr_v1.sh local_exp
# =============================================================================

export WANDB_START_METHOD="thread"
export WANDB_API_KEY=''  # set your Weights & Biases API key here or in the environment

set -euo pipefail

exp=${1:-local}
DATASET_ROOT='/home/work/chatbot-llms-3/roydon.tay/BARS-CTR/Avazu_x4' # set your dataset root path here

export PYTHONPATH='/home/work/chatbot-llms-3/roydon.tay/chatbot_rcmd' # set your own PYTHONPATH

# Inputs
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"${DATASET_ROOT}/train.csv"}
VALID_DATA_PATH=${VALID_DATA_PATH:-"${DATASET_ROOT}/valid.csv"}
TEST_DATA_PATH=${TEST_DATA_PATH:-"${DATASET_ROOT}/test.csv"}
AVAZU_VOCAB_PATH=${AVAZU_VOCAB_PATH:-"${DATASET_ROOT}/avazu_vocab.json"}
MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-"${DATASET_ROOT}/avazu_grouping_model_config_v1.json"}

# Hyperparameters (override by env)
BATCH_SIZE=${BATCH_SIZE:-10000}
NUM_EPOCHS=${NUM_EPOCHS:-20}
EVAL_STEPS=${EVAL_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
EMBEDDING_DIM=${EMBEDDING_DIM:-40}
DROPOUT=${DROPOUT:-0.2}
D_MODEL=${D_MODEL:-256}
LAMBDA_GEN=${LAMBDA_GEN:-0.1}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_VALID_SAMPLES=${MAX_VALID_SAMPLES:-0}
MAX_TEST_SAMPLES=${MAX_TEST_SAMPLES:-0}
MOE_LOAD_BALANCE=${MOE_LOAD_BALANCE:-0.01}
MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS:-4}
MOE_TOP_K=${MOE_TOP_K:-1}
MOE_FFN_DIM=${MOE_FFN_DIM:-0}

OUTPUT_DIR=${OUTPUT_DIR:-"/home/work/chatbot-llms-3/roydon.tay/chatbot_rcmd/app/unigcr/outputs/output_avazu_sc_moe_${exp}"}

echo "[Avazu CTR] exp=${exp}"
echo "[Avazu CTR] train=${TRAIN_DATA_PATH}"
echo "[Avazu CTR] valid=${VALID_DATA_PATH}"
echo "[Avazu CTR] test=${TEST_DATA_PATH}"
echo "[Avazu CTR] vocab=${AVAZU_VOCAB_PATH}"
echo "[Avazu CTR] model_config=${MODEL_CONFIG_PATH}"
echo "[Avazu CTR] output=${OUTPUT_DIR}"

if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
  echo "[ERROR] train file not found: ${TRAIN_DATA_PATH}"
  exit 1
fi

if [[ ! -f "${VALID_DATA_PATH}" ]]; then
  echo "[ERROR] valid file not found: ${VALID_DATA_PATH}"
  exit 1
fi

if [[ ! -f "${TEST_DATA_PATH}" ]]; then
  echo "[ERROR] test file not found: ${TEST_DATA_PATH}"
  exit 1
fi

if [[ ! -f "${AVAZU_VOCAB_PATH}" ]]; then
  echo "[ERROR] avazu vocab file not found: ${AVAZU_VOCAB_PATH}"
  exit 1
fi

cd /home/work/chatbot-llms-3/roydon.tay/chatbot_rcmd

python -m app.unigcr.train_avazu_unigcr \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --valid_data_path "${VALID_DATA_PATH}" \
  --avazu_vocab_path "${AVAZU_VOCAB_PATH}" \
  --model_config_path "${MODEL_CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --eval_steps "${EVAL_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --dropout "${DROPOUT}" \
  --d_model "${D_MODEL}" \
  --lambda_gen "${LAMBDA_GEN}" \
  --use_post_hstu_moe \
  --moe_num_experts ${MOE_NUM_EXPERTS} \
  --moe_load_balance ${MOE_LOAD_BALANCE} \
  --moe_top_k ${MOE_TOP_K} \
  --moe_ffn_dim ${MOE_FFN_DIM} \
  --num_workers "${NUM_WORKERS}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_valid_samples "${MAX_VALID_SAMPLES}" \
  --wandb_project "${WANDB_PROJECT:-avazu_unigcr}" \
  --wandb_run_name "${WANDB_RUN_NAME:-avazu_ctr_sc_moe_${exp}}" \
  --exp "${exp}" \
  --test_data_path "${TEST_DATA_PATH}" \
  --max_test_samples "${MAX_TEST_SAMPLES}" \
  --ctr_hidden_units 40 20 \
  --lambda_ctr 1.0 \
  --lambda_gen 0.1 \
  --hstu_num_heads 2 \
  --hstu_num_blocks 4 \
  --gen_loss_decay \
  --ctr_shallow_shortcut
  
