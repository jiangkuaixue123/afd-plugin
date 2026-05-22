# DeepSeekV2-Lite AFD Examples

These examples run DeepSeekV2-Lite with the AFD plugin through native
`vllm serve` plus explicit worker class paths. They assume vLLM `v0.19.1`, the
AFD plugin checkout on `PYTHONPATH`, and a local DeepSeekV2-Lite model path.

Set the model once:

```bash
export AFD_MODEL=/path/to/DeepSeek-V2-Lite
```

Optional knobs shared by both examples:

```bash
export AFD_VLLM_BIN=vllm
export AFD_API_PORT_BASE=18000
export AFD_PORT=6239
export AFD_MAX_TOKENS=8
```

## Eager 1A1F

```bash
bash example/deepseekv2-lite/eager_1a1f.sh
```

Defaults:

- Attention GPU: `0`
- FFN GPU: `1`
- vLLM mode: `--enforce-eager`

Override the GPUs with:

```bash
AFD_ATTENTION_GPUS=2 AFD_FFN_GPUS=3 bash example/deepseekv2-lite/eager_1a1f.sh
```

## DBO + Compile + FULL_DECODE_ONLY Graph 2A2F

```bash
bash example/deepseekv2-lite/dbo_compile_graph_full_decode_only_2a2f.sh
```

Defaults:

- Attention GPUs: `0,1`
- FFN GPUs: `2,3`
- AFD topology: `2A2F`
- vLLM DBO: enabled
- Compile config: `{"cudagraph_mode":"FULL_DECODE_ONLY"}`
- Capture size: `64`
- Completion requests: `128`

Override the main graph/DBO knobs with:

```bash
AFD_CAPTURE_SIZE=128 \
AFD_NUM_REQUESTS=256 \
AFD_DBO_DECODE_TOKEN_THRESHOLD=1 \
AFD_DBO_PREFILL_TOKEN_THRESHOLD=128 \
bash example/deepseekv2-lite/dbo_compile_graph_full_decode_only_2a2f.sh
```
