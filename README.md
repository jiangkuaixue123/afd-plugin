# afd-plugin

vLLM external plugin for Attention-FFN Disaggregation (AFD).

This repository is migrating the original in-tree AFD implementation into an
out-of-tree vLLM plugin. The current target runtime is vLLM `v0.19.1`.

## Current Status

Phase 1 established the CPU-safe plugin skeleton:

- Python package metadata for `vllm-afd-plugin`.
- `vllm.general_plugins` entry point: `afd = afd_plugin:register_afd`.
- Idempotent `register_afd()`. Current FFN daemon support applies a narrow,
  isolated EngineCore compat patch under `afd_plugin.compat.patches`.
- Plugin-owned `AFDConfig` parsed from `additional_config["afd"]`.
- Validation helpers for role, connector, topology, and worker class paths.
- No plugin-owned executable entrypoint package; both Attention and FFN sides
  are expected to enter through native `vllm serve` plus explicit class paths.
- Runtime class-path placeholders for:
  - `afd_plugin.runtime.AFDAttentionWorker`
  - `afd_plugin.runtime.AFDAttentionModelRunner`
  - `afd_plugin.runtime.AFDFFNWorker`
  - `afd_plugin.runtime.GPUFFNModelRunner`

Phase 2 added the Attention-side MVP:

- `AFDAttentionWorker` now injects `AFDAttentionModelRunner` while preserving
  the vLLM v1 GPU worker lifecycle.
- `AFDAttentionModelRunner` parses `additional_config["afd"]`, initializes the
  plugin-owned dummy connector, builds AFD metadata, and exposes it through
  `ForwardContext.additional_kwargs["afd_metadata"]`.
- A minimal plugin-owned model helper can read AFD metadata from the forward
  context without patching `vllm.forward_context`.

Phase 3 has started for the FFN side:

- `AFDFFNWorker` injects `GPUFFNModelRunner`, returns an empty KV cache spec,
  and starts a connector-driven FFN loop during worker initialization.
- `GPUFFNModelRunner` consumes Attention-side hidden states from the dummy
  connector, runs `model.compute_ffn_output()` when available, and returns
  outputs to the Attention side.
- The dummy connector now supports an in-process Attention -> FFN -> Attention
  round trip for CPU-safe smoke tests.

Phase 4 has started the P2P connector migration:

- `p2pconnector` is registered in the plugin connector factory.
- A CPU-safe topology helper defines the Phase 4 rank mapping:
  FFN ranks first, Attention ranks second, with each FFN owning one or more
  consecutive Attention ranks.
- The P2P connector lazily initializes vLLM/PyTorch distributed state, PyNCCL
  subgroups, DP metadata send/recv, and hidden-state send/recv paths.
- `extra_config["afd_size"]` remains supported for compatibility with the
  original AFD branch, while canonical plugin config continues to use
  `num_attention_servers` and `num_ffn_servers`.
- A DeepSeekV2 AFD E2E wrapper is registered lazily for server-side smoke
  testing. This wrapper loads full model weights on both Attention and FFN
  sides, and only splits the forward path across the connector.

Phase 4 is still in progress: FFN serving now works through ordinary
`vllm serve` and does not require `--headless` or
`--disable-hybrid-kv-cache-manager`. Scheduler-driven FFN execution still fails
fast, and role-based weight pruning, ubatching/DBO, and CUDA graph support
remain deferred. AFD runtimes currently fail fast unless `--enforce-eager` is
used. Opt-in GPU E2E coverage exists for eager `1A1F` and `2A2F` DeepSeekV2
P2P runs.

Example config shape:

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "dummy",
    "num_afd_stages": 3,
    "num_attention_servers": 1,
    "num_ffn_servers": 1
  }
}
```

P2P config shape:

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "p2pconnector",
    "host": "127.0.0.1",
    "port": 1239,
    "num_attention_servers": 2,
    "num_ffn_servers": 1,
    "afd_server_rank": 0
  }
}
```

## Development

```bash
uv run pytest
uv run ruff check .
```

Opt-in GPU E2E tests:

```bash
AFD_GPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite uv run pytest -q -m gpu
```

The GPU tests default to `AFD_GPU_E2E_GPUS=0,1,2,3`, run eager `1A1F` and
`2A2F`, and shell out to `tests/e2e_deepseek_v2_afd.py`. XAYF tests use native
vLLM data parallelism: Attention runs with `DP=X, TP=1`, FFN runs with
`DP=Y, TP=1`, and both roles enable expert parallelism.
