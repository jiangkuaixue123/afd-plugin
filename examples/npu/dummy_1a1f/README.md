# NPU AFD 1A1F Dummy Connector

This example starts a local 1 Attention + 1 FFN AFD deployment on NPU with the
development-only `npudummyconnector`.

`npudummyconnector` is intended for validating the NPU AFD worker/model-runner
lifecycle. It is not a production cross-process connector; use the real NPU
connector when validating end-to-end communication.

Set the model path first:

```bash
export MODEL=/path/to/model
```

Start the FFN side first:

```bash
ASCEND_RT_VISIBLE_DEVICES=1 VLLM_USE_V1=1 \
vllm serve "$MODEL" \
  --headless \
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
  }'
```

Then start the Attention side:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_USE_V1=1 \
vllm serve "$MODEL" \
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
```

If the installed `vllm serve` does not support `--headless`, remove it from the
FFN command and assign the FFN side an unused API port.
