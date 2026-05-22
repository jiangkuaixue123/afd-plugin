#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${AFD_MODEL:-${AFD_GPU_E2E_MODEL:-}}"

if [[ -z "${MODEL}" ]]; then
  echo "Set AFD_MODEL to a local DeepSeekV2-Lite model path." >&2
  exit 2
fi

exec python "${SCRIPT_DIR}/run.py" \
  --model "${MODEL}" \
  --vllm-bin "${AFD_VLLM_BIN:-vllm}" \
  --num-attention-servers 1 \
  --num-ffn-servers 1 \
  --attention-gpus "${AFD_ATTENTION_GPUS:-0}" \
  --ffn-gpus "${AFD_FFN_GPUS:-1}" \
  --api-port-base "${AFD_API_PORT_BASE:-18000}" \
  --afd-port "${AFD_PORT:-6239}" \
  --max-tokens "${AFD_MAX_TOKENS:-8}" \
  --startup-timeout "${AFD_STARTUP_TIMEOUT:-900}" \
  --ffn-start-delay "${AFD_FFN_START_DELAY:-25}" \
  --common-vllm-arg=--trust-remote-code
