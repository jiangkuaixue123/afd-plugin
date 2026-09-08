#!/usr/bin/env bash

set -euo pipefail

PROXY_IP=${PROXY_IP:-127.0.0.1}
PROXY_PORT=${PROXY_PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-dsv4}

curl -fsS "http://${PROXY_IP}:${PROXY_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Who are you?\"}],
    \"max_tokens\": 32,
    \"temperature\": 0
  }"
