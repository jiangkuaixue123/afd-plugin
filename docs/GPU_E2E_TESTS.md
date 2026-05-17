# GPU E2E Tests

AFD GPU tests are opt-in because they start real multi-process vLLM servers and
load DeepSeekV2-Lite weights on every role.

## Pytest

```bash
AFD_GPU_E2E_MODEL=/home/jcz/models/DeepSeek-V2-Lite \
  uv run pytest -q -m gpu
```

Defaults:

- `AFD_GPU_E2E_GPUS=0,1,2,3`
- `AFD_GPU_E2E_VLLM_BIN=vllm`
- eager mode only; the runner always passes `--enforce-eager`
- FFN uses ordinary `vllm serve`; no `--headless`
- no `--disable-hybrid-kv-cache-manager`

The pytest cases currently cover:

- `test_deepseek_v2_eager_1a1f_end_to_end`
- `test_deepseek_v2_eager_2a2f_end_to_end`

## Manual Runner

The same command builder can be used directly.

`1A1F`:

```bash
python tests/e2e_deepseek_v2_afd.py \
  --model /home/jcz/models/DeepSeek-V2-Lite \
  --num-attention-servers 1 \
  --num-ffn-servers 1 \
  --attention-gpus 0 \
  --ffn-gpus 1 \
  --api-port-base 18000 \
  --afd-port 6239 \
  --common-vllm-arg=--trust-remote-code
```

`2A2F`:

```bash
python tests/e2e_deepseek_v2_afd.py \
  --model /home/jcz/models/DeepSeek-V2-Lite \
  --num-attention-servers 2 \
  --num-ffn-servers 2 \
  --attention-gpus 0,1 \
  --ffn-gpus 2,3 \
  --api-port-base 18100 \
  --afd-port 6249 \
  --common-vllm-arg=--trust-remote-code
```

`--ffn-headless` is available for deployment isolation experiments, but it is
not required for current FFN startup.
