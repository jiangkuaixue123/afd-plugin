# afd-plugin

vLLM external plugin for Attention-FFN Disaggregation (AFD).

This repository is migrating the original in-tree AFD implementation into an
out-of-tree vLLM plugin. The current target runtime is vLLM `v0.19.1`.

## Phase 1 Status

Phase 1 establishes the CPU-safe plugin skeleton:

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

The runtime classes are intentionally import/resolve-only placeholders in Phase
1. Real Attention and FFN execution is scheduled for Phase 2 and Phase 3.

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
