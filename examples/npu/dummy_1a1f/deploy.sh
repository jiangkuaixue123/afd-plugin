#!/bin/bash
# Deploy AFD 1A1F on NPU
set -e

MODEL=/home/admin/model-csi/models/modelhub_97542_deepseek-v2-lite-36500041_20260318110950/model

# cd to /tmp to avoid vllm namespace package conflict with code/vllm/ repo root
cd /tmp

# Start FFN side (card 1, headless)
echo "Starting FFN side on card 1..."
ASCEND_RT_VISIBLE_DEVICES=1 VLLM_USE_V1=1 nohup vllm serve "$MODEL" \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUFFNWorker \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --enforce-eager \
  --additional-config '{
    "afd": {
      "enabled": true,
      "role": "ffn",
      "connector": "npudummyconnector",
      "host": "127.0.0.1",
      "port": 1239,
      "num_attention_servers": 1,
      "num_ffn_servers": 1,
      "afd_server_rank": 0
    }
  }' > /tmp/afd_ffn.log 2>&1 &
FFN_PID=$!
echo "FFN PID: $FFN_PID"

# Wait for FFN to start
sleep 5

# Start Attention side (card 0, API port 8000)
echo "Starting Attention side on card 0..."
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --enforce-eager \
  --additional-config '{
    "afd": {
      "enabled": true,
      "role": "attention",
      "connector": "npudummyconnector",
      "host": "127.0.0.1",
      "port": 1239,
      "num_attention_servers": 1,
      "num_ffn_servers": 1,
      "afd_server_rank": 0,
      "extra_config": {
        "dummy_passthrough_without_peer": true
      }
    }
  }'
