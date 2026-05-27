# Phase 4 P2P Connector Design

Phase 4 uses the real P2P backend while keeping the Attention and FFN runtime
adapters intact. The earlier in-process dummy connector has been removed.

## Scope

- Register `p2pconnector` through the plugin-owned connector factory.
- Keep package import CPU-safe by delaying torch, CUDA, PyNCCL, and vLLM runtime
  imports until connector initialization or send/recv calls.
- Use `additional_config["afd"]` as the only config channel.
- Support the original branch's topology rule: `num_attention_servers` must be
  greater than or equal to `num_ffn_servers`, and must be an integer multiple of
  it.
- Preserve `extra_config["afd_size"]` compatibility for original-style values
  such as `4A2F`.
- Provide a DeepSeekV2 E2E model wrapper that loads full weights on both roles
  and splits only the forward path. This covers first server-side eager `1A1F`
  and `2A2F` smoke testing, not the final memory-efficient model implementation.
- Keep P2P independent of graph-cache policy. Phase 5/6 now layer two-way DBO
  metadata and `FULL_DECODE_ONLY` CUDA graph replay on top of this connector.

## Rank Mapping

The AFD P2P world places FFN ranks first, followed by Attention ranks:

```text
[F0, F1, ..., A0, A1, ...]
```

Each FFN rank owns one subgroup. For `4A2F`, the groups are:

```text
F0 -> A0, A1
F1 -> A2, A3
```

Only the first `min(num_attention_servers, num_ffn_servers)` Attention ranks
send DP metadata to FFN ranks. Hidden states are exchanged inside each
FFN-owned subgroup.

## Current Gaps

- Role-based weight pruning is not implemented; both sides load full
  DeepSeekV2 weights.
- GPU-gated multi-process tests exist for eager `1A1F` / `2A2F`,
  `FULL_DECODE_ONLY` `1A1F` / `2A2F`, and `FULL_DECODE_ONLY` `2A2F` with DBO
  ubatch replay. They are opt-in through `AFD_GPU_E2E_MODEL`.
- CUDA graph metadata flags and FFN receive-buffer preallocation are supported
  by the P2P connector; graph cache ownership remains in `GPUFFNModelRunner`.
- Two-way DBO metadata is supported for current AFD graph/eager paths; only
  `num_ubatches=2` is currently allowed.
- More flexible non-divisible A/F routing is deferred until topology hardening.
