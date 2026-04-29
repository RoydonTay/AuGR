#!/bin/bash
# =============================================================================
# TaoBao CTR training launcher
# Usage:
#   bash run_train_taobao_ctr_v1.sh <exp>
# Example:
#   bash run_train_taobao_ctr_v1.sh local_exp
# =============================================================================

export WANDB_START_METHOD="thread"
export WANDB_API_KEY=''  # set your Weights & Biases API key here or in the environment

set -euo pipefail

exp=${1:-local}
DATASET_ROOT=${2:-"/home/work/chatbot-llms-3/roydon.tay/BARS-CTR/TaobaoAd_x1"} # set your dataset root path here

# Inputs
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"${DATASET_ROOT}/train.csv"}
VALID_DATA_PATH=${VALID_DATA_PATH:-"${DATASET_ROOT}/test.csv"}
TEST_DATA_PATH=${TEST_DATA_PATH:-"${DATASET_ROOT}/test.csv"}
TAOBAO_VOCAB_PATH=${TAOBAO_VOCAB_PATH:-"${DATASET_ROOT}/taobao_vocab.json"}
MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-"${DATASET_ROOT}/taobao_grouping_model_config_v1.json"}

# Feature setup
LABEL_COL=${LABEL_COL:-"clk"}
TARGET_FEATURE_NAME=${TARGET_FEATURE_NAME:-"cate_id"}
CATEGORICAL_FEATURES=${CATEGORICAL_FEATURES:-"final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,adgroup_id,pid,price,brand,campaign_id,cate_id,customer,cms_segid,cms_group_id"}
ITEM_FEATURES=${ITEM_FEATURES:-"adgroup_id,pid,price,brand,campaign_id,cate_id,customer"}
SEQUENCE_FEATURES=${SEQUENCE_FEATURES:-"cate_his,btag_his,brand_his"}
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

OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/output_taobao_${exp}"}

echo "[Taobao CTR] train=${TRAIN_DATA_PATH}"
echo "[Taobao CTR] valid=${VALID_DATA_PATH}"
echo "[Taobao CTR] test=${TEST_DATA_PATH}"
echo "[Taobao CTR] vocab=${TAOBAO_VOCAB_PATH}"
echo "[Taobao CTR] model_config=${MODEL_CONFIG_PATH}"
echo "[Taobao CTR] output=${OUTPUT_DIR}"

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

cd ../../.. # change to repo root

python -m src.training.train_taobao_gen_ctr \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --valid_data_path "${VALID_DATA_PATH}" \
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
  --wandb_project "${WANDB_PROJECT:-taobao_unigcr}" \
  --wandb_run_name "${WANDB_RUN_NAME:-${exp}}" \
  --test_data_path "${TEST_DATA_PATH}" \
  --max_test_samples "${MAX_TEST_SAMPLES}" \
  --ctr_hidden_units 32 16 \
  --lambda_ctr 1.0 \
  --lambda_gen 0.1 \
  --hstu_num_heads 2 \
  --hstu_num_blocks 4 \
  --ctr_shallow_shortcut \
  --gen_loss_decay \
