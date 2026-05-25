#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${AFD_MODEL:-${AFD_GPU_E2E_MODEL:-}}"
CAPTURE_SIZE="${AFD_CAPTURE_SIZE:-64}"

if [[ -z "${MODEL}" ]]; then
  echo "Set AFD_MODEL to a local DeepSeekV2-Lite model path." >&2
  exit 2
fi

exec python "${SCRIPT_DIR}/run.py" \
  --model "${MODEL}" \
  --vllm-bin "${AFD_VLLM_BIN:-vllm}" \
  --num-attention-servers 2 \
  --num-ffn-servers 2 \
  --attention-gpus "${AFD_ATTENTION_GPUS:-0,1}" \
  --ffn-gpus "${AFD_FFN_GPUS:-2,3}" \
  --api-port-base "${AFD_API_PORT_BASE:-18400}" \
  --afd-port "${AFD_PORT:-6279}" \
  --max-tokens "${AFD_MAX_TOKENS:-8}" \
  --startup-timeout "${AFD_STARTUP_TIMEOUT:-900}" \
  --ffn-start-delay "${AFD_FFN_START_DELAY:-25}" \
  --cuda-graph-full-decode-only \
  --use-decode-bench-connector \
  --cudagraph-capture-size "${CAPTURE_SIZE}" \
  --num-requests "${AFD_NUM_REQUESTS:-128}" \
  --request-concurrency "${AFD_REQUEST_CONCURRENCY:-${AFD_NUM_REQUESTS:-128}}" \
  --enable-dbo \
  --dbo-decode-token-threshold "${AFD_DBO_DECODE_TOKEN_THRESHOLD:-1}" \
  --dbo-prefill-token-threshold "${AFD_DBO_PREFILL_TOKEN_THRESHOLD:-${CAPTURE_SIZE}}" \
  --common-vllm-arg=--trust-remote-code
