# GPU E2E Tests

AFD GPU tests are opt-in because they start real data-parallel vLLM servers and
load DeepSeekV2-Lite weights on every role.

## Pytest

```bash
AFD_GPU_E2E_MODEL=/home/jcz/models/DeepSeek-V2-Lite \
  uv run pytest -q -m gpu
```

Defaults:

- `AFD_GPU_E2E_GPUS=0,1,2,3`
- `AFD_GPU_E2E_VLLM_BIN=vllm`
- eager tests pass `--enforce-eager`; CUDA graph tests opt into
  `cudagraph_mode=FULL_DECODE_ONLY`
- XAYF topologies use native vLLM DP: Attention runs with `DP=X, TP=1`, FFN
  runs with `DP=Y, TP=1`, and both roles pass `--enable-expert-parallel`
- graph tests pass `DecodeBenchConnector`, align `max-num-seqs`,
  `max-num-batched-tokens`, and CUDA graph capture size
- the DBO graph test exercises the two-stage ubatch CUDA graph path

The pytest cases currently cover:

- `test_deepseek_v2_eager_1a1f_end_to_end`
- `test_deepseek_v2_eager_2a2f_end_to_end`
- `test_deepseek_v2_full_decode_cudagraph_1a1f_end_to_end`
- `test_deepseek_v2_full_decode_cudagraph_2a2f_end_to_end`
- `test_deepseek_v2_full_decode_cudagraph_2a2f_dbo_replays_ubatch_graph`

Useful graph overrides:

- `AFD_GPU_E2E_GRAPH_CAPTURE_SIZE=64`
- `AFD_GPU_E2E_GRAPH_REQUESTS=64`
- `AFD_GPU_E2E_DBO_REQUESTS=128`
- `AFD_GPU_E2E_GRAPH_MAX_TOKENS=8`
- `AFD_GPU_E2E_DBO_DECODE_THRESHOLD=1`
- `AFD_GPU_E2E_DBO_PREFILL_THRESHOLD=64`

## Manual Runner

The same command builder can be used directly.

`1A1F`:

```bash
python tests/e2e/gpu/deepseek_v2_lite/runner.py \
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
python tests/e2e/gpu/deepseek_v2_lite/runner.py \
  --model /home/jcz/models/DeepSeek-V2-Lite \
  --num-attention-servers 2 \
  --num-ffn-servers 2 \
  --attention-gpus 0,1 \
  --ffn-gpus 2,3 \
  --api-port-base 18100 \
  --afd-port 6249 \
  --common-vllm-arg=--trust-remote-code
```

This starts one Attention `vllm serve` with
`CUDA_VISIBLE_DEVICES=0,1 --data-parallel-size 2 --tensor-parallel-size 1`
and one FFN `vllm serve` with
`CUDA_VISIBLE_DEVICES=2,3 --data-parallel-size 2 --tensor-parallel-size 1`.

`2A2F FULL_DECODE_ONLY`:

```bash
python tests/e2e/gpu/deepseek_v2_lite/runner.py \
  --model /home/jcz/models/DeepSeek-V2-Lite \
  --num-attention-servers 2 \
  --num-ffn-servers 2 \
  --attention-gpus 0,1 \
  --ffn-gpus 2,3 \
  --api-port-base 18300 \
  --afd-port 6269 \
  --cuda-graph-full-decode-only \
  --use-decode-bench-connector \
  --expect-ffn-cudagraph-replay \
  --cudagraph-capture-size 64 \
  --num-requests 64 \
  --request-concurrency 64 \
  --common-vllm-arg=--trust-remote-code
```

`2A2F FULL_DECODE_ONLY + DBO`:

```bash
python tests/e2e/gpu/deepseek_v2_lite/runner.py \
  --model /home/jcz/models/DeepSeek-V2-Lite \
  --num-attention-servers 2 \
  --num-ffn-servers 2 \
  --attention-gpus 0,1 \
  --ffn-gpus 2,3 \
  --api-port-base 18400 \
  --afd-port 6279 \
  --cuda-graph-full-decode-only \
  --use-decode-bench-connector \
  --enable-dbo \
  --dbo-decode-token-threshold 1 \
  --dbo-prefill-token-threshold 64 \
  --expect-ffn-cudagraph-replay \
  --expect-ffn-ubatch-cudagraph-replay \
  --cudagraph-capture-size 64 \
  --num-requests 128 \
  --request-concurrency 128 \
  --common-vllm-arg=--trust-remote-code
```

For capture size `64` on `2A2F`, the DBO graph test intentionally uses 128
concurrent requests. With only 64 concurrent requests each DP rank often sees
about 32 tokens, and vLLM avoids creating an empty second ubatch, so live replay
can stay on the single-stage `[0:[64,64]]` graph.

## Current Limitations

- GPU tests are opt-in and are not run by default on CPU-only development
  machines.
- Graph tests assert request success; they do not yet compare token output
  against an eager baseline.
- CUDA graph coverage is limited to `FULL_DECODE_ONLY`, TP=1, and current
  DeepSeekV2 wrapper paths.
