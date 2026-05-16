# Phase 4 P2P Connector Design

Phase 4 replaces the Phase 3 dummy connector transport with a real P2P backend
while keeping the Attention and FFN runtime adapters intact.

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
  and splits only the forward path. This is for first server-side 1A1F smoke
  testing, not the final memory-efficient model implementation.
- Continue to fail fast for AFD + ubatching/DBO; that belongs to Phase 5.

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
- GPU-gated multi-process round-trip tests are still pending.
- CUDA graph capture is not supported by this connector version; server runs
  must pass `--enforce-eager`.
- Ubatching/DBO support remains Phase 5.
- More flexible non-divisible A/F routing is deferred until topology hardening.
