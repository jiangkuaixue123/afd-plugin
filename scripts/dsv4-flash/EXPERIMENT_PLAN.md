# DSV4 Flash Prefill：AFD 与大 EP 性能对比实验

## 1. 实验目标

比较 DeepSeek-V4-Flash prefill-only 场景下 AFD 与普通大 EP 的：

- achieved QPS；
- input token throughput；
- TTFT mean、P25、P50、P90、P95、P99；
- E2EL（输出长度为 1，因此应接近 TTFT）；
- 成功率、失败数和每 NPU 归一化吞吐。

本实验不启动 decode、PD proxy 或 KV connector。所有服务显式关闭
prefix cache，benchmark 直接请求 prefill 主节点的 Attention/full-model
HTTP API。

## 2. 固定实验矩阵

### 2.1 拓扑

| 名称 | 拓扑 | NPU die | 节点布局 | 当前计划 |
| --- | --- | ---: | --- | --- |
| `afd_dp6tp4` | Attention DP6TP4；FFN EP8 | 32 | node0: A16；node1: A8 + F8 | 正式实验 |
| `afd_dp3tp8` | Attention DP3TP8；FFN EP8 | 32 | node0: A16；node1: A8 + F8 | 正式实验 |
| `ep16` | full model DP4TP4/EP16 | 16 | 单台 910C，node0 运行全部 16 die | 正式实验 |
| `ep32` | full model DP4TP8/EP32 | 32 | node0: 16；node1: 16 | 暂缓，不启动 |

`ep16` 不跨机。`ep32`、`afd_dp6tp4` 和 `afd_dp3tp8` 使用两台
16-die 910C。

### 2.2 Chunk size 与 offered load

- `afd_dp6tp4`、`afd_dp3tp8`：16384、32768、65536；
- `ep16`：4096、8192、16384、32768、65536；
- `ep32`：本阶段暂缓，不产生正式结果；
- `--request-rate`：4、6、8 req/s；
- 每个测量点正式运行 3 次；
- AFD：`2 topology × 3 chunk × 3 rate × 3 repeat = 54` 份；
- EP16：`1 topology × 5 chunk × 3 rate × 3 repeat = 45` 份；
- 当前正式矩阵共 `11 topology/chunk`、99 份 detailed result。

原始需求中的 `32786` 按 2 的幂修正为 `32768`。

> 2026-08-29 执行变更：AFD 不再运行 `chunk=4096` 和 `chunk=8192`；
> 两档已产生的 partial results 和日志仅作为探索性证据保留，不补齐、不标
> OOM，也不纳入正式矩阵统计。EP16 将 `chunk=4096` 加入正式矩阵，复用
> 已通过校验的结果并在恢复实验后补齐缺失 repeat/rate。EP32 暂缓，只有
> 收到新的明确指令后才启动。
>
> 2026-08-29 offered-load 变更：正式实验只保留 `4、6、8 req/s`。变更前
> 已完整落盘的 `12、16、20 req/s` result 和日志作为探索性证据保留，
> 不覆盖、不补齐，也不纳入正式矩阵覆盖率或性能统计。

### 2.3 当前执行状态

实验已于 `2026-08-29` 完成，正式矩阵 `99/99` 个测量点均已有最终判定：

- `36` 份 detailed result 通过全部字段和数组校验；
- `63` 个点由 `7` 个 topology/chunk 的 `capacity_undeployable` 状态及
  服务日志解释；
- `invalid=0`，`unaccounted=0`；
- 六台 itask 已清场，无相关服务、benchmark、sweep、EngineCore 或 NPU
  进程残留。

完整结论、性能汇总和证据包校验值见
`bench_results/dsv4-flash/final_report/EXPERIMENT_REPORT.md`。所有已完整落盘
的非正式 result 和日志继续保留；其中 `RPS=12/16/20` 以及 AFD
`chunk=4096/8192` 仅作为探索性证据，不纳入正式覆盖率或性能结论。

## 3. 固定配置

所有拓扑保持以下设置一致：

```text
MAX_MODEL_LEN=65536
PREFILL_MAX_NUM_SEQS=16（每 DP rank）
BLOCK_SIZE=128
output_tokens=1
temperature=0
prefix cache=false
KV connector=false
chunked prefill=true
enforce eager=true
seed=1024
```

AFD 额外固定：

```text
connector=CAMAsyncAFDConnector
AFD_HOST=P_NODE_IP
AFD_PORT=1239
num_attention_ranks=24
num_ffn_ranks=8
async_moe_ubatching=true
async_moe_num_ubatches=2
async_moe_split=token
enable_force_load_balance=false
```

跨机 DP 固定使用：

```text
DP coordinator=P_NODE_IP
PREFILL_DP_RPC_PORT=12321
node0 提供唯一 HTTP API
node1 以 --headless 加入
```

每次正式实验记录模型路径、vLLM/vllm-ascend/AFD commit、CANN 版本、
两节点 IP、NIC、启动命令和服务日志。实验期间不得改变模型、软件版本、
绑核、HCCL、FlashComm1 或显存利用率配置。

## 4. 数据集

使用：

```text
moonconv-wildchat-v4-flash-prefill/workloads/formal_0_1_2_vllm_bench.jsonl
```

冻结属性：

| 属性 | 值 |
| --- | ---: |
| 请求数 | 1536 |
| formal_0/formal_1/formal_2 | 各 512 |
| 总输入 token | 15,803,063 |
| 平均输入长度 | 10,288.45 |
| 最大输入长度 | 63,778 |
| 每请求输出 token | 1 |
| SHA-256 | `1ebccbd149bc8f28568d3d5eced3911d2a9473fb015bdb829b898a26abf63d08` |

benchmark 使用 `--disable-shuffle --no-oversample --num-prompts -1`，确保
所有实验使用相同请求集合和顺序。

## 5. 公共环境变量

两节点使用相同的主节点、次节点和网卡配置：

```bash
export P_NODE_IP=<node0-ip>
export P_SECONDARY_NODE_IP=<node1-ip>
export NIC_NAME=<hccl-gloo-interface>
export DSV4_MODEL=/path/to/DeepSeek-V4-Flash-w8a8-mtp/model
export AFD_HOST=${P_NODE_IP}
export AFD_PORT=1239
export PREFILL_DP_RPC_PORT=12321
export PREFILL_ENABLE_KV_CONNECTOR=0
export MAX_MODEL_LEN=65536
```

`HCCL_IF_IP` 不手工复用主节点地址：启动脚本会根据
`PREFILL_NODE_ID` 分别设置为 node0/node1 的本地 IP。AFD rendezvous 和
DP coordinator 始终填写 node0 的 `P_NODE_IP`。

## 6. 服务启动

以下命令中的 `<chunk>` 按 2.2 节对应拓扑的 chunk 列表替换。每次改变
chunk 或拓扑必须先在所有相关节点执行 `stop_local.sh`，再重新启动服务。

### 6.1 AFD DP6TP4 + EP8

先在 node1 启动 FFN EP8 和 Attention local DP2/headless：

```bash
PREFILL_TOPOLOGY=afd_dp6tp4 \
PREFILL_NODE_ID=1 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill.sh
```

再在 node0 启动 Attention local DP4、DP coordinator 和 API：

```bash
PREFILL_TOPOLOGY=afd_dp6tp4 \
PREFILL_NODE_ID=0 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill.sh
```

全局 Attention rank 为 DP6TP4=24；node0 的 DP start rank 为 0，node1
为 4。

### 6.2 AFD DP3TP8 + EP8

先在 node1 启动 FFN EP8 和 Attention local DP1/headless：

```bash
PREFILL_TOPOLOGY=afd_dp3tp8 \
PREFILL_NODE_ID=1 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill.sh
```

再在 node0 启动 Attention local DP2、DP coordinator 和 API：

```bash
PREFILL_TOPOLOGY=afd_dp3tp8 \
PREFILL_NODE_ID=0 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill.sh
```

全局 Attention rank 为 DP3TP8=24；node0 的 DP start rank 为 0，node1
为 2。

### 6.3 大 EP16：单机 DP4TP4/EP16

只在 node0 执行：

```bash
PREFILL_TOPOLOGY=ep16 \
PREFILL_NODE_ID=0 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill_full.sh
```

脚本拒绝以 `PREFILL_NODE_ID=1` 启动 EP16，避免误配成跨机实验。

### 6.4 大 EP32：两机 DP4TP8/EP32（预留，本阶段不执行）

以下命令仅供后续恢复 EP32 时使用；当前不得启动。

先在 node1 启动 local DP2/headless，DP start rank=2：

```bash
PREFILL_TOPOLOGY=ep32 \
PREFILL_NODE_ID=1 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill_full.sh
```

再在 node0 启动 local DP2、DP coordinator 和 API：

```bash
PREFILL_TOPOLOGY=ep32 \
PREFILL_NODE_ID=0 \
PREFILL_MAX_NUM_BATCHED_TOKENS=<chunk> \
bash scripts/dsv4-flash/run_prefill_full.sh
```

### 6.5 健康检查与停止

```bash
curl -fsS http://${P_NODE_IP}:7100/health
```

停止时在参与实验的每台节点执行：

```bash
bash scripts/dsv4-flash/stop_local.sh
```

## 7. OOM/无法部署处理

按 chunk 从小到大启动。若服务在模型初始化、KV cache 初始化或 warmup
阶段出现明确的 NPU OOM，且主 API 无法健康，则该 topology/chunk 标记为
`oom_undeployable`，停止两节点残留进程并跳过对应的 3 个 request rate。

示例：

```bash
bash scripts/dsv4-flash/run_bench_sweep.sh \
  --topology afd_dp6tp4 \
  --chunk-size 65536 \
  --record-oom "node1 prefill_attention log: NPU out of memory during KV cache initialization"
```

记录写入：

```text
bench_results/dsv4-flash/<topology>/chunk_<chunk>/deployment_status.json
```

只有日志明确显示 OOM 时记录 `oom_undeployable`。网络超时、rank 未加入、
端口冲突或配置错误不得归类为 OOM，应修复后重跑。若服务能够部署和
warmup，但只在某个高 request rate 下 OOM，则记录该测量点失败，不将整个
chunk 标记为无法部署。

## 8. Benchmark 执行

服务健康后，在 node0 运行完整三档 offered request rate。正式实验：

```bash
DSV4_BENCH_HOST=${P_NODE_IP} \
DSV4_BENCH_PORT=7100 \
bash scripts/dsv4-flash/run_bench_sweep.sh \
  --topology <topology> \
  --chunk-size <chunk> \
  --rates 4,6,8 \
  --repeats 3
```

冒烟阶段先运行：

```bash
DSV4_BENCH_HOST=${P_NODE_IP} \
bash scripts/dsv4-flash/run_bench_sweep.sh \
  --topology <topology> \
  --chunk-size 16384 \
  --rates 4,8 \
  --repeats 1
```

默认不传 `--max-concurrency`，保持 open-loop offered load，避免客户端
并发上限把高 TTFT 场景错误限流。每次 benchmark 执行 16 个 warmup 请求，
然后运行全部 1536 个正式请求并启用 `--save-detailed`。

结果目录：

```text
bench_results/dsv4-flash/
  <topology>/
    chunk_<chunk>/
      deployment_status.json
      rps_<offered-rps>/
        repeat_<n>/result.json
```

`run_bench.sh` 默认拒绝覆盖已有结果。只有明确需要重跑并替换时才设置：

```bash
export DSV4_BENCH_OVERWRITE=1
```

## 9. 单点验收标准

每份正式结果必须满足：

- `completed=1536`；
- `failed=0`；
- `total_input_tokens=15803063`；
- `total_output_tokens=1536`；
- detailed 数组包含 1536 个 input length 和 TTFT；
- metadata 中 topology、chunk、offered RPS、repeat、dataset SHA-256 正确；
- 服务端日志未启用 prefix cache、KV connector 或 speculative decode。

不满足时保留结果和日志，但标记为失败测量点，不进入性能均值。

## 10. 汇总与对比方法

每个测量点对三次重复计算均值、标准差和变异系数。输出以下曲线：

1. offered QPS → achieved QPS；
2. offered QPS → input token/s；
3. offered QPS → P50/P90/P99 TTFT；
4. chunk size → 最大稳定 QPS/吞吐；
5. 每 NPU achieved QPS 和 input token/s。

饱和点定义为以下任一条件首次出现：

- achieved QPS < offered QPS × 95%；
- P99 TTFT 相邻档位明显陡升；
- 出现请求失败或运行时 OOM。

当前阶段比较 `afd_dp6tp4`、`afd_dp3tp8` 与 `ep16`。跨方案直接比较只使用
共同 chunk：16384、32768、65536；EP16 的 4096、8192 仅用于自身 chunk
趋势分析。由于 AFD 使用 32 die、EP16 使用 16 die，报告必须同时展示
原始系统吞吐和每 NPU 归一化吞吐；
不得把资源规模不同的原始 QPS 直接解释为架构效率。EP32 暂不纳入结果，
后续若恢复，可用于同为 32 die 的等资源主比较和 EP16→EP32 扩展性分析。

TPOT 和 ITL 在输出长度为 1 时没有统计意义，不作为本实验结论指标。
