# DSV4 Flash AFD DP6TP4 + EP8 Profiling 方案

## 1. 目标与固定配置

在 DeepSeek-V4-Flash prefill-only 场景下，采集接近满载并出现请求排队时的
AFD Attention 侧和 FFN 侧 NPU profiling。

| 项目 | 配置 |
| --- | --- |
| Topology | Attention DP6TP4；FFN EP8 |
| `max-num-batched-tokens` | 65536（64K） |
| Offered load | 8 req/s，open-loop |
| Dataset | `formal_0_1_2_vllm_bench.jsonl` |
| Attention profiling | 10 个有效 runner step |
| FFN profiling | 20 个有效 FFN transaction |
| Shape 记录 | Attention、FFN 两侧均开启 |
| Stack 记录 | 两侧均关闭 |
| Prefix cache | 关闭 |
| KV connector | 关闭 |
| Execution | eager |

Profiling 运行只用于性能定位，不覆盖或替代正式 benchmark 结果。如果
`chunk=65536` 的服务不能正常部署和通过健康检查，应先处理或记录部署失败，
不得在初始化异常的服务上采集 profiling。

## 2. Profiler 机制

两侧使用不同的 profiling 入口：

- Attention 使用原有的 vLLM/vllm-ascend service profiler，通过
  `PREFILL_ATTN_PROFILER_CONFIG` 配置，并通过 Attention HTTP API 的
  `/start_profile`、`/stop_profile` 动态控制采集窗口；
- FFN 使用 AFD 插件自带的 NPU profiler，通过
  `AFD_NPU_FFN_PROFILER_*` 环境变量配置 schedule。

FFN 运行在 connector-driven daemon loop 中，不能使用 vLLM 的
`--profiler-config`、`PREFILL_FFN_PROFILER_CONFIG` 或 FFN HTTP
`/start_profile`、`/stop_profile` 完成这次采集。

两侧 step 均不等价于 request：

- Attention step 对应一次 Attention `execute_model()` 调度迭代；
- FFN step 对应一次 connector-driven FFN transaction；
- 一次 step 可能承载多个请求或 chunk。

FFN 的 AFD profiler 源码固定使用：

```python
record_shapes=True
```

因此 FFN 不需要、也没有 `AFD_NPU_FFN_PROFILER_RECORD_SHAPES` 环境变量；
只要启用 AFD FFN profiler，就会记录算子输入 shape。FFN profiler 同时固定
使用 Level2。Attention 则在 `PREFILL_ATTN_PROFILER_CONFIG` 中显式设置
`torch_profiler_record_shapes=true`。

注意：当前仓库参考版 `vllm-ascend` 的 `TorchNPUProfilerWrapper` 没有显式
把 `torch_profiler_record_shapes` 传给 `torch_npu.profiler.profile()`。本次通过
AFD 插件兼容补丁
`afd_plugin/compat/patches/npu/service_profiler_record_shapes.py` 补齐参数透传，
不直接修改 `vllm-ascend`；正式采集后仍必须检查每个 rank 的
`profiler_info_*.json` 均为 `record_shapes=true`，并确认
`operator_details.csv` 的 `Input Shapes` 存在非空数据。FFN 的 AFD profiler
已在源码中固定 `record_shapes=True`，不受此问题影响。

## 3. 采样窗口

Attention 在 RPS=8 workload 已经出现持续排队后通过 HTTP 动态开始，并抓取
10 个有效 worker step。

FFN profiler 必须在服务启动前通过环境变量启用，使用以下 schedule：

| FFN 阶段 | 数值 | 说明 |
| --- | ---: | --- |
| `SKIP_FIRST` | 0 | 不额外使用第二层跳过 |
| `WAIT` | 600 | 前 600 个 FFN transaction 不记录 |
| `WARMUP` | 1 | profiler warmup，数据丢弃 |
| `ACTIVE` | 20 | 实际写入 trace 的有效窗口 |
| `REPEAT` | 1 | 只生成一轮 trace |

正式采集前应先用一次无 profiler 的 RPS=8 运行确认大约第 600 个 FFN
transaction 时 workload 尚未结束且系统已经持续排队。如果采样点不满足这
两个条件，应调整 `WAIT`，不要通过增加 `ACTIVE` 来等待排队。

完整数据集包含 1536 个请求，RPS=8 时理论发包时间约为 192 秒，以上短窗口
无需覆盖整轮 benchmark。

## 4. Attention 侧配置

在启动 Attention 服务前设置原有的 service profiler 配置：

```bash
export PREFILL_ATTN_PROFILER_CONFIG='{
  "profiler":"torch",
  "torch_profiler_dir":"/tmp/profile/afd_dp6tp4_c65536_rps8/attention",
  "torch_profiler_with_stack":false,
  "torch_profiler_with_memory":false,
  "torch_profiler_record_shapes":true,
  "ignore_frontend":true,
  "delay_iterations":0,
  "max_iterations":10
}'
```

`run_prefill_attention.sh` 会将该变量作为 `--profiler-config` 传给 vLLM。
Attention 侧不要设置 `AFD_NPU_ATTENTION_PROFILER_ENABLE=1`，避免两套
profiler 同时工作。

Attention 服务跨 node0 和 node1。两个节点启动 Attention 前都需要提供
profiler config，并使用各自独立的输出目录，避免跨节点收集后文件重名：

```bash
# 将 JSON 中的 torch_profiler_dir 分别设置为：
# node0: /tmp/profile/afd_dp6tp4_c65536_rps8/attention_node0
# node1: /tmp/profile/afd_dp6tp4_c65536_rps8/attention_node1
```

运行 benchmark 并确认 `num_requests_waiting > 0` 持续约 10 秒后触发：

```bash
curl -X POST "http://${P_NODE_IP}:7100/start_profile"
```

达到 10 个有效 worker step 后会自动停止记录。benchmark 结束前仍显式调用
一次 stop，确保 profiler 状态清理并完成输出：

```bash
curl -X POST "http://${P_NODE_IP}:7100/stop_profile"
```

## 5. FFN 侧配置

FFN EP8 只在 node1 启动。在启动 node1 FFN 服务前设置：

```bash
export AFD_NPU_FFN_PROFILER_ENABLE=1
export AFD_NPU_FFN_PROFILER_DIR=/tmp/profile/afd_dp6tp4_c65536_rps8/ffn_node1
export AFD_NPU_FFN_PROFILER_SKIP_FIRST=0
export AFD_NPU_FFN_PROFILER_WAIT=600
export AFD_NPU_FFN_PROFILER_WARMUP=1
export AFD_NPU_FFN_PROFILER_ACTIVE=20
export AFD_NPU_FFN_PROFILER_REPEAT=1
export AFD_NPU_FFN_PROFILER_WITH_STACK=0
```

FFN profiler 由 `AFDNPUFFNModelRunner` 创建，并在 connector-driven loop 的
每次 transaction 前推进 schedule。达到 20 个 active transaction 后会触发
trace handler 写盘，不需要 FFN HTTP profiling 接口。

FFN EP8 涉及 dispatch/combine 和专家负载差异，默认保留全部 8 个 FFN rank
的 trace。当前实现没有 rank allowlist。

## 6. 服务启动

为了减少 profiler 对近满载系统的叠加扰动，Attention 和 FFN 优先分两轮采集：

1. 第一轮只设置 Attention service profiler 配置，FFN profiler 保持关闭；
2. 第二轮只设置 FFN profiler 环境变量，Attention profiler 保持关闭；
3. 两轮均使用完全相同的 topology、chunk、数据集和 RPS。

每轮先在 node1 启动 FFN EP8 和 Attention local DP2/headless：

```bash
PREFILL_TOPOLOGY=afd_dp6tp4 \
PREFILL_NODE_ID=1 \
PREFILL_MAX_NUM_BATCHED_TOKENS=65536 \
bash scripts/dsv4-flash/run_prefill.sh
```

再在 node0 启动 Attention local DP4、DP coordinator 和 API：

```bash
PREFILL_TOPOLOGY=afd_dp6tp4 \
PREFILL_NODE_ID=0 \
PREFILL_MAX_NUM_BATCHED_TOKENS=65536 \
bash scripts/dsv4-flash/run_prefill.sh
```

启动 benchmark 前确认服务健康：

```bash
curl -fsS "http://${P_NODE_IP}:7100/health"
```

## 7. Benchmark

在 node0 运行一轮独立的 profiling benchmark：

```bash
DSV4_BENCH_HOST="${P_NODE_IP}" \
bash scripts/dsv4-flash/run_bench.sh \
  --topology afd_dp6tp4 \
  --chunk-size 65536 \
  --request-rate 8 \
  --repeat 1
```

不要设置 `--max-concurrency`，保持 open-loop offered load。运行期间观察：

```bash
curl -s "http://${P_NODE_IP}:7100/metrics" |
  grep -E 'num_requests_(running|waiting)'
```

需要确认 profiler 的 active 窗口内 `num_requests_waiting > 0`。如果无法从
metrics 直接对应 runner step，应结合服务日志时间戳和生成的 trace 时间戳
核对。

## 8. 推荐执行顺序

1. 无 profiler 预跑 RPS=8，确认第 600 个 FFN transaction 附近 workload 尚未
   结束且系统已持续排队。
2. 清理两节点残留服务和旧 profiling 输出。
3. 第一轮仅配置 Attention service profiler，排队稳定后通过 HTTP 抓取 10
   个有效 step。
4. 等待 trace handler 写盘，记录文件清单和目录大小后停止两节点服务。
5. 清理进程并重新启动完全相同的服务配置。
6. 第二轮仅开启 FFN profiler，抓取 20 个有效 transaction。
7. 等待 FFN trace 写盘后停止服务。
8. 保存两轮服务日志、benchmark 日志、环境变量和 trace 文件清单。

## 9. 结果检查

每轮完成后执行：

```bash
find /tmp/profile/afd_dp6tp4_c65536_rps8 -maxdepth 4 -type f -print
du -sh /tmp/profile/afd_dp6tp4_c65536_rps8/*
```

同时确认：

- profiling active 窗口内请求等待队列持续非零；
- benchmark 没有请求失败；
- Attention trace 覆盖 10 个有效 runner step；
- FFN trace 覆盖 20 个有效 transaction；
- Attention 和 FFN trace 均包含算子输入 shape；
- 两侧均未开启 stack；
- node0、node1 输出目录没有混写或混入旧 trace；
- profiling 轮次的数据只作为辅助证据，不并入正式性能结果。

## 10. 产物归档

远端保留完整原始 trace 和离线解析目录。Attention 的最终上传包采用带 timeline
的解析格式，保留每个 rank 的 `profiler_info_*.json`、
`profiler_metadata.json`，以及 `ASCEND_PROFILER_OUTPUT` 下的
`trace_view.json`、算子、kernel、通信和 step 等 CSV/JSON；只排除体积较大且
可由远端原始 trace 重新生成的 profiler DB、原始 `PROF_*` 目录、`FRAMEWORK`
目录和解析日志。FFN 完整压缩包体积可控，保留全部 8 个 rank 的完整数据。
归档后应逐包确认 `trace_view.json` 数量与 rank 数一致，执行 `gzip -t` 并记录
SHA256。
