# afd-plugin

vLLM external plugin for Attention-FFN Disaggregation (AFD).

This repository is migrating the original in-tree AFD implementation into an
out-of-tree vLLM plugin. The current target runtime is vLLM `v0.19.1`.

## Current Status

Phase 1 established the CPU-safe plugin skeleton:

- Python package metadata for `vllm-afd-plugin`.
- `vllm.general_plugins` entry point: `afd = afd_plugin:register_afd`.
- Idempotent `register_afd()` with no monkey patches.
- Plugin-owned `AFDConfig` parsed from `additional_config["afd"]`.
- Validation helpers for role, connector, topology, and worker class paths.
- No plugin-owned executable entrypoint package; both Attention and FFN sides
  are expected to enter through native `vllm serve` plus explicit class paths.
- Runtime class-path placeholders for:
  - `afd_plugin.runtime.AFDAttentionWorker`
  - `afd_plugin.runtime.AFDAttentionModelRunner`
  - `afd_plugin.runtime.AFDFFNWorker`
  - `afd_plugin.runtime.GPUFFNModelRunner`

Phase 2 has started for the Attention side:

- `AFDAttentionWorker` now injects `AFDAttentionModelRunner` while preserving
  the vLLM v1 GPU worker lifecycle.
- `AFDAttentionModelRunner` parses `additional_config["afd"]`, initializes the
  plugin-owned dummy connector, builds AFD metadata, and exposes it through
  `ForwardContext.additional_kwargs["afd_metadata"]`.
- A minimal plugin-owned model helper can read AFD metadata from the forward
  context without patching `vllm.forward_context`.

The FFN runtime class paths remain Phase 1 placeholders until Phase 3.

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

## Development

```bash
uv run pytest
uv run ruff check .
```
