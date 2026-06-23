# DeepSeek-V2-Lite AFD Examples

End-to-end launch scripts for running DeepSeek-V2-Lite with the AFD
(Attention-FFN Disaggregation) plugin on vLLM `v0.19.1`.

## Prerequisites

- 4 GPUs (A/H-class, tested against L20X).
- vLLM `v0.19.1` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- DeepSeek-V2-Lite weights on disk. All scripts default to
  `/path/model_weights/DeepSeek-V2-Lite`; override with
  `MODEL_PATH=...` when launching.
- A free TCP port `6269` on `127.0.0.1` for the AFD p2p connector, and
  ports `18301`/`18302`/`18305` for the vLLM HTTP servers.

## Directory layout

```
.
├── benchmark.sh                          # online serving benchmark client
├── prefill_decode_disaggregation/        # prefill_decode_disaggregation, 2P1A1F topology
│   ├── 2p1a1f_eager_dbo.sh
│   └── 2p1a1f_graph_dbo.sh
└── prefill_decode_multiplexing/          # prefill_decode_multiplexing, 2A2F topology
    ├── 2a2f_eager_dbo_dp1tp2.sh
    ├── 2a2f_eager_dbo_dp2tp1.sh
    ├── 2a2f_graph_dbo_dp1tp2.sh
    └── 2a2f_graph_dbo_dp2tp1.sh
```

## File name convention

`<topology>_<mode>_<dp>tp<tp>.sh`

| Token        | Meaning                                                       |
|--------------|---------------------------------------------------------------|
| `NpNaNf`     | N prefill producers + N attention workers + N FFN workers     |
| `NaNaf`      | N attention workers + N FFN workers     |
| `eager`      | `--enforce-eager`, CUDA graph disabled                        |
| `graph`      | `FULL_DECODE_ONLY` CUDA graph                |
| `dbo`        | Dual Batch Overlap enabled                       |
| `dpNtpM`     | `--data-parallel-size N --tensor-parallel-size M`             |

So `2a2f_graph_dbo_dp1tp2.sh` = 2 attention + 2 FFN, CUDA graph on, DBO on,
DP=1, TP=2.

## Topologies

### 1. Prefill/Decode Disaggregation — `2p1a1f`

4 processes, one GPU each (`CUDA_VISIBLE_DEVICES=0,1,2,3`):

| GPU | Role                            | Worker class                  | Port  |
|-----|---------------------------------|-------------------------------|-------|
| 0   | Prefill producer #1             | default vLLM worker           | 18301 |
| 1   | Prefill producer #2             | default vLLM worker           | 18302 |
| 2   | Decode attention                | `AFDAttentionWorker`          | 18305 |
| 3   | Decode FFN                      | `AFDFFNWorker`                | 18305 |

KV cache is produced on GPUs 0/1 and shipped to the decode side through
`LMCacheConnectorV1` (`kv_role=kv_producer` → `kv_consumer`). Within the
decode tier, attention and FFN are further split across GPUs 2 and 3 via
the AFD p2p connector.

### 2. Prefill/Decode Multiplexing — `2a2f`

2 processes, two GPUs each:

| GPUs   | Role              | Worker class           | Port  |
|--------|-------------------|------------------------|-------|
| 0, 1   | Attention    | `AFDAttentionWorker`   | 18305 |
| 2, 3   | FFN          | `AFDFFNWorker`         | 18305 |

The four variants cover the TP/DP cross product:

| File                            | DP | TP |
|---------------------------------|----|----|
| `2a2f_*_dp1tp2.sh`              | 1  | 2  |
| `2a2f_*_dp2tp1.sh`              | 2  | 1  |

## Running

Pick a script and execute it from the repository root. Each script
backgrounds its workers and writes per-worker logs (`afd_prefill.log`,
`afd_prefill1.log`, `attn.log`, `ffn.log`) in the current directory.

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash example/deepseekv2-lite/prefill_decode_multiplexing/2a2f_graph_dbo_dp1tp2.sh
```

Wait for `attn.log` and `ffn.log` to print the
`Application startup complete` line on port `18305` before sending
traffic.

### Running the benchmark

Once the serving stack is up, run:

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export RESULT_DIR=/path/results
export RESULT_FILENAME=2a2f_graph_dbo_dp1tp2.json
bash example/deepseekv2-lite/benchmark.sh
```

It fires 1024 random requests (1024 input tokens / 128 output tokens) at
unlimited request rate with `--max-concurrency 32` against `127.0.0.1:18305`,
and dumps the JSON result to `$RESULT_DIR/$RESULT_FILENAME`.

## Common AFD configuration

Every AFD worker is wired through `--additional-config` with the same
shape; only `role` and `afd_size` differ between attention and FFN:

```jsonc
{
  "afd": {
    "enabled": true,
    "role": "attention",            // or "ffn"
    "connector": "p2pconnector",
    "host": "127.0.0.1",
    "port": 6269,
    "num_attention_servers": 1,      // 2 in 2A2F
    "num_ffn_servers": 1,            // 2 in 2A2F
    "extra_config": {
      "afd_size": "1A1F"             // "2A2F" in 2A2F
    }
  }
}
```

DBO (Dual Batch Overlap) is turned on for all examples with
`--dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12`.

### Switching eager → graph

Graph mode replaces `--enforce-eager` with:

```
--max-cudagraph-capture-size 64
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
                       "cudagraph_capture_sizes":[64]}'
```
