# DeepSeek-V4-Flash 1P1D + P-side CamAsync AFD

Prefill-only AFD 与 EP16/DP4TP8-EP32 性能实验的拓扑、启动命令、OOM 记录和
benchmark 流程见 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)。

This directory deploys DeepSeek-V4-Flash on two 16-NPU Ascend 910C nodes.
The P node additionally splits Attention and FFN with `CAMAsyncAFDConnector`.

## Topology

| Node | Role | Devices | Parallelism |
| --- | --- | --- | --- |
| P | Attention | `0-7` | DP2 x TP4 |
| P | FFN | `8-15` | TP1 x EP8 |
| D | Full decode | `0-15` | DP16 x TP1 + EP |

P/D KV transfer uses `MooncakeHybridConnector`. The P-side CamAsync path uses
exactly two AFD-managed MoE ubatches with `async_moe_split=token`, so tokens
are balanced between the two stages. Do not add vLLM native DBO flags: native
DBO is incompatible with CamAsync.

The D topology and DSV4 runtime parameters follow the vllm-ascend 910C 1P1D
example. The P topology is changed from its full-model DP4TP4 layout to the
requested AFD Attention DP2TP4 + FFN EP8 layout. FlashComm1 is configurable;
the launchers enable it for TP Attention/full-model roles and disable it for
the TP1 FFN role.

## Prerequisites

- Matching vLLM v0.26, vllm-ascend v0.26, and the DSV4-capable AFD plugin on
  both nodes.
- CANN 9.0.1 and the CamAsync operator packages installed on the P node.
- `MooncakeHybridConnector` dependencies available on both nodes.
- The same DSV4 Flash W8A8 checkpoint path on both nodes.
- A network interface usable by HCCL, Gloo, and Mooncake.

The defaults use `${REPO_ROOT}/afd-plugin26` and the model path already used by
`scripts/start_dsv4.sh`. Override `PYTHON_AFD_PLUGIN_PATH` and `DSV4_MODEL` when
the deployment paths differ.

## Start

Set the same network variables on both nodes:

```bash
export P_NODE_IP=<P-node-IP>
export D_NODE_IP=<D-node-IP>
export NIC_NAME=<HCCL-network-interface>
export DSV4_MODEL=/path/to/DeepSeek-V4-Flash-w8a8-mtp/model
```

Start D on the D node:

```bash
bash scripts/dsv4-flash/run_decode.sh
```

Start the two P roles on the P node:

```bash
PREFILL_ENABLE_KV_CONNECTOR=1 bash scripts/dsv4-flash/run_prefill.sh
```

After both API backends are healthy, start the proxy on either node (normally
the P node):

```bash
bash scripts/dsv4-flash/run_proxy.sh
```

Verify:

```bash
curl http://${P_NODE_IP}:7100/health
curl http://${D_NODE_IP}:7100/health
PROXY_IP=${P_NODE_IP} bash scripts/dsv4-flash/curl_test.sh
```

## Prefill benchmark

The benchmark dataset contains the shuffled union of `formal_0`, `formal_1`,
and `formal_2`. It uses native vLLM custom-dataset fields (`prompt` and
`output_tokens`) and does not require a Python loader wrapper.

The fixed AFD/EP16/EP32 matrix is driven by `run_bench_sweep.sh`. For example,
after bringing up `afd_dp6tp4` with chunk size 8192:

```bash
DSV4_BENCH_HOST=${P_NODE_IP} \
bash scripts/dsv4-flash/run_bench_sweep.sh \
  --topology afd_dp6tp4 \
  --chunk-size 8192 \
  --repeats 3
```

See [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) before running the matrix. It
defines the two-node AFD/EP32 layouts, single-node EP16 layout, fixed settings,
result validation, and OOM recording rules.

## Useful overrides

```bash
MAX_MODEL_LEN=131072 \
PREFILL_MAX_NUM_BATCHED_TOKENS=8192 \
DECODE_MAX_NUM_BATCHED_TOKENS=120 \
PREFILL_MAX_NUM_SEQS=16 \
DECODE_MAX_NUM_SEQS=60 \
PREFILL_GPU_MEMORY_UTILIZATION=0.85 \
AFD_ASYNC_MOE_LAYOUT_LOG=1 \
bash scripts/dsv4-flash/run_prefill.sh
```

`PREFILL_MAX_NUM_BATCHED_TOKENS` must be identical on the Attention and FFN
roles. The shared `common_env.sh` enforces that by using the same variable in
both launchers.

## Stop

Run this on each node where scripts were started:

```bash
bash scripts/dsv4-flash/stop_local.sh
```

Logs and pid files are under `scripts/dsv4-flash/logs/`.
