#!/bin/bash
# =============================================================================
# TaoBao CTR training launcher
# Usage:
#   bash run_train_taobao_ctr_v1.sh <exp> [dataset_root]
# Example:
#   bash run_train_taobao_ctr_v1.sh local_exp /data/BARS-CTR/TaoBao
# =============================================================================

set -euo pipefail

exp=${1:-local}
DATASET_ROOT=${2:-"/data/BARS-CTR/TaoBao"}

# Inputs
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"${DATASET_ROOT}/train.csv"}
VALID_DATA_PATH=${VALID_DATA_PATH:-"${DATASET_ROOT}/valid.csv"}
TEST_DATA_PATH=${TEST_DATA_PATH:-"${DATASET_ROOT}/test.csv"}
TAOBAO_VOCAB_PATH=${TAOBAO_VOCAB_PATH:-"${DATASET_ROOT}/taobao_vocab.json"}
MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-"${DATASET_ROOT}/taobao_grouping_model_config_v1.json"}

# Feature setup
LABEL_COL=${LABEL_COL:-"clk"}
TARGET_FEATURE_NAME=${TARGET_FEATURE_NAME:-"cate_id"}
CATEGORICAL_FEATURES=${CATEGORICAL_FEATURES:-"final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,adgroup_id,pid,price,brand,campaign_id,cate_id,customer"}
ITEM_FEATURES=${ITEM_FEATURES:-"adgroup_id,pid,price,brand,campaign_id,cate_id,customer"}
SEQUENCE_FEATURES=${SEQUENCE_FEATURES:-""}
SEQUENCE_SEPARATOR=${SEQUENCE_SEPARATOR:-"^"}
SEQUENCE_MAX_LEN=${SEQUENCE_MAX_LEN:-50}
SEQUENCE_POOLING_TYPE=${SEQUENCE_POOLING_TYPE:-"self_attention"}

# Hyperparameters (override by env)
BATCH_SIZE=${BATCH_SIZE:-10000}
NUM_EPOCHS=${NUM_EPOCHS:-20}
EVAL_STEPS=${EVAL_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
EMBEDDING_DIM=${EMBEDDING_DIM:-32}
D_MODEL=${D_MODEL:-256}
DROPOUT=${DROPOUT:-0.2}
LAMBDA_USER=${LAMBDA_USER:-0.1}
LAMBDA_ITEM=${LAMBDA_ITEM:-1.0}
LAMBDA_FUSED=${LAMBDA_FUSED:-1.0}
USE_FUSED_CTR_LOSS=${USE_FUSED_CTR_LOSS:-0}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_VALID_SAMPLES=${MAX_VALID_SAMPLES:-0}
MAX_TEST_SAMPLES=${MAX_TEST_SAMPLES:-0}
MAX_STEPS=${MAX_STEPS:--1}

OUTPUT_DIR=${OUTPUT_DIR:-"./app/unigcr/outputs/output_taobao_${exp}"}

echo "[TaoBao CTR] exp=${exp}"
echo "[TaoBao CTR] train=${TRAIN_DATA_PATH}"
echo "[TaoBao CTR] valid=${VALID_DATA_PATH}"
echo "[TaoBao CTR] test=${TEST_DATA_PATH}"
echo "[TaoBao CTR] vocab=${TAOBAO_VOCAB_PATH}"
echo "[TaoBao CTR] model_config=${MODEL_CONFIG_PATH}"
echo "[TaoBao CTR] output=${OUTPUT_DIR}"

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

if [[ ! -f "${TAOBAO_VOCAB_PATH}" ]]; then
  echo "[ERROR] taobao vocab file not found: ${TAOBAO_VOCAB_PATH}"
  exit 1
fi

FUSED_FLAG=""
if [[ "${USE_FUSED_CTR_LOSS}" == "1" ]]; then
  FUSED_FLAG="--use_fused_ctr_loss"
fi

python3 -m app.unigcr.train_taobao_ctr \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --valid_data_path "${VALID_DATA_PATH}" \
  --test_data_path "${TEST_DATA_PATH}" \
  --taobao_vocab_path "${TAOBAO_VOCAB_PATH}" \
  --model_config_path "${MODEL_CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --label_col "${LABEL_COL}" \
  --target_feature_name "${TARGET_FEATURE_NAME}" \
  --categorical_features "${CATEGORICAL_FEATURES}" \
  --item_features "${ITEM_FEATURES}" \
  --sequence_features "${SEQUENCE_FEATURES}" \
  --sequence_separator "${SEQUENCE_SEPARATOR}" \
  --sequence_max_len "${SEQUENCE_MAX_LEN}" \
  --sequence_pooling_type "${SEQUENCE_POOLING_TYPE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --d_model "${D_MODEL}" \
  --dropout "${DROPOUT}" \
  --lambda_user "${LAMBDA_USER}" \
  --lambda_item "${LAMBDA_ITEM}" \
  --lambda_fused "${LAMBDA_FUSED}" \
  --num_workers "${NUM_WORKERS}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_valid_samples "${MAX_VALID_SAMPLES}" \
  --max_test_samples "${MAX_TEST_SAMPLES}" \
  --wandb_project "${WANDB_PROJECT:-taobao_unigcr}" \
  --wandb_run_name "${WANDB_RUN_NAME:-taobao_ctr_${exp}}" \
  --exp "${exp}" \
  --ctr_hidden_units 40 20 \
  --hstu_num_heads 2 \
  --hstu_num_blocks 4 \
  ${FUSED_FLAG}
