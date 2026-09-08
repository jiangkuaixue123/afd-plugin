#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

CLIENT_HOST=${LM_EVAL_HOST:-127.0.0.1}
MODEL_NAME=${LM_EVAL_MODEL:-${SERVED_MODEL_NAME}}
TOKENIZER=${LM_EVAL_TOKENIZER:-${DSV4_MODEL}}
TOKENIZER_BACKEND=${LM_EVAL_TOKENIZER_BACKEND:-None}
OUTPUT_PATH=${LM_EVAL_OUTPUT_PATH:-${SCRIPT_DIR}/lm_eval_gsm8k_300}
TASKS=${LM_EVAL_TASKS:-gsm8k_offline}
NUM_CONCURRENT=${LM_EVAL_NUM_CONCURRENT:-64}
MAX_LENGTH=${LM_EVAL_MAX_LENGTH:-4096}
MAX_GEN_TOKS=${LM_EVAL_MAX_GEN_TOKS:-512}
LIMIT=${LM_EVAL_LIMIT:-300}
NUM_FEWSHOT=${LM_EVAL_NUM_FEWSHOT:-}
GSM8K_DATA_DIR=${LM_EVAL_GSM8K_DATA_DIR:-${REPO_ROOT}/gsm8k_offline_data}
LOG_SAMPLES=${LM_EVAL_LOG_SAMPLES:-1}
WAIT_TIMEOUT=${LM_EVAL_WAIT_TIMEOUT:-1800}
WAIT_INTERVAL=${LM_EVAL_WAIT_INTERVAL:-10}

BASE_URL="http://${CLIENT_HOST}:${PROXY_PORT}/v1/completions"
HEALTH_URL="http://${CLIENT_HOST}:${PROXY_PORT}/healthcheck"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval not found in PATH" >&2
  exit 1
fi

echo "Waiting for DSV4 PD proxy at ${HEALTH_URL} ..."
deadline=$((SECONDS + WAIT_TIMEOUT))
until curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${HEALTH_URL}" >&2
    exit 1
  fi
  sleep "${WAIT_INTERVAL}"
done

DATASET_FORMAT=arrow
TRAIN_FILE="${GSM8K_DATA_DIR}/gsm8k-train.arrow"
TEST_FILE="${GSM8K_DATA_DIR}/gsm8k-test.arrow"
if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
  DATASET_FORMAT=parquet
  TRAIN_FILE="${GSM8K_DATA_DIR}/gsm8k-train.parquet"
  TEST_FILE="${GSM8K_DATA_DIR}/gsm8k-test.parquet"
fi

if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
  echo "Missing GSM8K files under ${GSM8K_DATA_DIR}" >&2
  echo "Expected gsm8k-train/test.arrow or gsm8k-train/test.parquet" >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
TASK_DIR="${OUTPUT_PATH}/tasks"
mkdir -p "${TASK_DIR}"
TASK_FILE="${TASK_DIR}/gsm8k_offline.yaml"

cat > "${TASK_FILE}" <<EOF
tag:
  - math_word_problems
task: gsm8k_offline
dataset_path: ${DATASET_FORMAT}
dataset_name: null
output_type: generate_until
dataset_kwargs:
  data_files:
    train: ${TRAIN_FILE}
    test: ${TEST_FILE}
training_split: train
fewshot_split: train
test_split: test
doc_to_text: 'Q: {{question}}
  A(Please follow the summarized result at the end with the format of "The answer is xxx", where xx is the result.):'
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ','
      - '\$'
      - "(?s).*#### "
      - '\.$'
generation_kwargs:
  until:
    - "\n\nQ:"
    - "\nQ:"
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: '#### (\-?[0-9\.\,]+)'
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: '(-?[\$0-9.,]{2,})|(-?[0-9]+)'
      - function: "take_first"
metadata:
  version: 3.0
EOF

LM_EVAL_ARGS=(
  --model local-completions
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL},tokenizer=${TOKENIZER},tokenizer_backend=${TOKENIZER_BACKEND},tokenized_requests=False,trust_remote_code=True,num_concurrent=${NUM_CONCURRENT},max_length=${MAX_LENGTH},max_gen_toks=${MAX_GEN_TOKS}"
  --include_path "${TASK_DIR}"
  --tasks "${TASKS}"
  --output_path "${OUTPUT_PATH}"
  --limit "${LIMIT}"
)

if [[ -n "${NUM_FEWSHOT}" ]]; then
  LM_EVAL_ARGS+=(--num_fewshot "${NUM_FEWSHOT}")
fi

case "${LOG_SAMPLES,,}" in
  1 | true | yes | on)
    LM_EVAL_ARGS+=(--log_samples)
    ;;
esac

echo "Running DSV4 GSM8K: first ${LIMIT} samples, model=${MODEL_NAME}, base_url=${BASE_URL}, concurrency=${NUM_CONCURRENT}, output=${OUTPUT_PATH}"
lm_eval "${LM_EVAL_ARGS[@]}"
