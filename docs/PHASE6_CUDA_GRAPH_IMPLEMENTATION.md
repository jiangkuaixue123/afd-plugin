# Phase 6 CUDA Graph 开发实施文档

本文档定义 AFD plugin Phase 6 的 CUDA graph 支持方案。目标版本仍固定为
vLLM `v0.19.1`，不修改 `../vllm` 源码树；所有行为通过本插件 runtime、
connector、model wrapper、显式 class path 或受控 compat shim 提供。

## 背景

当前 AFD runtime 已经完成 eager P2P 闭环，并在代码中预留了部分 CUDA graph
信号：

- `AFDAttentionModelRunner` 和 `GPUFFNModelRunner` 目前都会调用
  `fail_if_cuda_graph_enabled()`，要求启动时传 `--enforce-eager`。
- `AFDAttentionModelRunner` 已经有 `_is_warmup`、`_afd_pending_metadata` 和
  `_dummy_run()` provider 机制，但还没有把 `is_graph_capturing` 传到 connector。
- `P2PAFDConnector.update_state_from_dp_metadata()` / `send_dp_metadata_list()`
  / `recv_dp_metadata_list()` 已经携带 `is_graph_capturing` 和 `is_warmup`。
- `GPUFFNModelRunner.capture_model()` 当前是空实现，FFN 侧还没有 graph cache。
- `AFDUBatchWrapper` 对 `CUDAGraphMode.FULL` 仍 fail fast。

原始 AFD commit 在 in-tree `GPUModelRunner` 中把 AFD metadata 注入 normal
`execute_model()` 和 `_dummy_run()`；在 `_warmup_and_capture()` 中用
`_is_warmup` 区分 warmup 与正式 capture；FFN 侧按 DP metadata key 管理 graph。
迁移到 external plugin 时不能照搬大段 vLLM runner，因此 Phase 6 需要以最小覆盖
点实现同等语义。

## 开发策略

Phase 6 的第一原则是：原始 AFD commit 中已经验证过的 CUDA graph 逻辑，能复用就
尽量复用。本插件只做 external plugin 化所必需的解耦：

- Attention 侧复用原始 `GPUModelRunner` 中 normal run / `_dummy_run()` /
  `_warmup_and_capture()` 对 AFD metadata 的处理方式，只把 `vllm_config.afd_config`
  改成 plugin-owned `additional_config["afd"]` 解析结果。
- FFN 侧优先迁移原始 `GPUFFNModelRunner` 的 `_make_graph_key()`、`_cuda_graphs`、
  `_graph_memory_pool`、`_dummy_run()`、`capture_model()` 和 replay fallback 逻辑。
- connector 侧沿用原始 `send_dp_metadata_list(dp_metadata_list,
  is_graph_capturing=..., is_warmup=...)` 信号，只把实现落在
  `afd_plugin.connectors`。
- 只有原始实现强绑定 in-tree vLLM 改动时，才在插件内抽 helper 或加受控 compat
  shim。

Phase 6 只支持 vLLM `FULL_DECODE_ONLY` CUDA graph。其他 CUDA graph mode，包括
`FULL`、`PIECEWISE`、`FULL_AND_PIECEWISE` 以及 ubatching full graph，都应 fail fast。

## 目标

- normal run、warmup、capture、replay 都向 FFN 侧发送正确的 DP/AFD metadata。
- Attention 侧只放行 `CUDAGraphMode.FULL_DECODE_ONLY`，并复用 vLLM 原生 full decode
  graph capture/replay 调度。
- FFN 侧复用原始 AFD 的 graph-keyed capture/replay 逻辑，按 DP metadata shape
  维护 CUDA graph cache。
- 保持 package import CPU-safe，CUDA-heavy import 继续延迟到 runtime。
- 增加 GPU-gated correctness tests 和 trace/runbook，失败时能清楚区分 metadata、
  graph key、connector shape 和 replay 问题。

## 非目标

- Phase 6 第一版不解决 role-based weight pruning。
- 第一版不支持 `CUDAGraphMode.PIECEWISE`。
- 第一版不支持 `CUDAGraphMode.FULL` 或 `FULL_AND_PIECEWISE`。
- 第一版不支持 AFD + ubatching + CUDA graph 的组合。
- 第一版不支持 TP/PP 下 FFN graph cache，除非 eager 基线已覆盖并有独立验证。

## 支持矩阵

| 场景 | Phase 6.1 | 后续 | 备注 |
| --- | --- | --- | --- |
| Attention eager | 支持 | 支持 | 当前基线 |
| Attention `FULL_DECODE_ONLY` | 支持 | 支持 | 唯一放行的 CUDA graph mode |
| Attention `PIECEWISE` | fail fast | 待定 | 当前不支持 |
| Attention `FULL` / `FULL_AND_PIECEWISE` | fail fast | 待定 | 当前不支持 |
| FFN eager | 支持 | 支持 | 当前基线 |
| FFN graph-keyed capture/replay | 支持 | 支持 | 优先复用原始 FFN runner 逻辑 |
| AFD + ubatching + eager | 支持 Phase 5 结果 | 支持 | 仅两路 ubatch |
| AFD + ubatching + CUDA graph | fail fast | 待设计 | 需要独立 metadata/graph key |

## 核心设计

### 1. 配置与校验

把当前 `fail_if_cuda_graph_enabled()` 改为更细的 mode 校验：

- 当 `model_config.enforce_eager=True` 时保持当前 eager 路径。
- 当 AFD role 为 Attention 时，只允许 `enforce_eager=True` 或
  `compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY`。
- 当 AFD role 为 Attention 且 runtime mode 可能进入 `PIECEWISE`、`FULL` 或
  `FULL_AND_PIECEWISE` 时 fail fast，错误信息指出 Phase 6 只支持
  `FULL_DECODE_ONLY`。
- 当 AFD role 为 FFN 时，允许启动时不传 `--enforce-eager`，但只启用插件自己的
  graph-keyed capture/replay cache。
- 当 `parallel_config.use_ubatching=True` 且 CUDA graph enabled 时继续 fail fast。

建议新增 helper：

```python
validate_cuda_graph_mode(vllm_config, *, role: str) -> AFDCUDAGraphPolicy
```

`AFDCUDAGraphPolicy` 应至少包含：

- `enabled`
- `mode_name`
- `allow_attention_full_decode_only`
- `enable_ffn_graph_cache`
- `allow_cuda_graph_with_ubatching`

CPU-safe tests 覆盖不同 config 组合，不需要 import torch/vLLM CUDA。

### 2. Attention Metadata 发送

Attention 侧需要覆盖三条路径：

- normal `execute_model()`：当前 `_model_forward()` 已能安装 metadata，但
  `_send_dp_metadata()` 需要接收 `is_graph_capturing`。
- CUDA graph warmup `_dummy_run(..., cudagraph_runtime_mode=NONE)`：需要发送
  `is_warmup=True`，让 FFN 侧只做 eager warmup 或预分配，不登记正式 graph。
- CUDA graph capture `_dummy_run(..., is_graph_capturing=True)`：需要发送
  `is_graph_capturing=True`，让 FFN 侧按同一个 DP metadata key capture 或确认已有
  graph。

实施点：

- 覆盖 `AFDAttentionModelRunner._warmup_and_capture()`，在 warmup 循环中设置
  `self._is_warmup=True`，正式 capture 前恢复为 `False`。
- 覆盖或包裹 `_dummy_run()`，把 `is_graph_capturing` 保存为短生命周期字段，例如
  `self._afd_is_graph_capturing`。
- `_send_dp_metadata()` 同时向 connector 传
  `is_graph_capturing=self._afd_is_graph_capturing` 和
  `is_warmup=self._is_warmup`。
- `_build_afd_metadata()` 使用 padded ubatch slices 时，metadata token lens 必须与
  graph batch descriptor 一致；模型实际通信 shape 使用 connector 的
  `_tensor_metadata_list` 校验。

### 3. Connector 状态机

DP metadata 发送的 payload 已经是：

```text
(dp_metadata_list, is_graph_capturing, is_warmup)
```

Phase 6 需要明确 FFN 侧状态语义：

- `is_warmup=True`：FFN 执行 eager warmup，不登记 graph。
- `is_graph_capturing=True`：FFN 对当前 graph key 执行 capture，第一版按原始
  `_dummy_run()` 逻辑 capture `_ffn_forward()`。
- 两者都为 `False` 且 graph cache 命中：FFN replay graph。
- 两者都为 `False` 且 graph cache miss：默认 eager fallback，并 trace warning；
  测试阶段可通过 env 开启 strict miss fail fast。

建议新增 `AFDGraphRunMode`：

```text
EAGER
WARMUP
CAPTURE
REPLAY
```

connector 只负责传递状态和预分配 shape buffer，不负责管理 graph cache。connector
通信是否能稳定参与 capture 由 GPU E2E 验证决定；如不稳定，再把 FFN graph 收窄为
compute-only fallback。

### 4. FFN Graph Cache

FFN 侧优先复用原始 AFD `GPUFFNModelRunner` 的结构，而不是重新设计一套 graph
runtime。建议直接迁移并插件化以下字段和方法：

```python
self.use_cuda_graph = not self.model_config.enforce_eager
self._cuda_graphs: dict[tuple, dict] = {}
self._graph_memory_pool = None

@staticmethod
def _make_graph_key(dp_metadata_list: dict) -> tuple: ...

def _dummy_run(cudagraph_runtime_mode, dp_metadata_list, is_attn_graph_capturing):
    ...

def capture_model(dp_metadata_list=None, is_warmup=False, is_attn_graph_capturing=True):
    ...
```

graph key 第一版沿用原始实现：

```text
tuple((stage_idx, tuple(meta.num_tokens_across_dp_cpu.tolist()))
      for stage_idx, meta in sorted(dp_metadata_list.items()))
```

只有当 GPU 验证发现 ratio、dtype、hidden size 或 TP/EP 会造成 key collision 时，才
扩展 key。扩展 key 时要加 backward-compatible tests。

运行时顺序：

```text
recv_attn_output
  -> graph replay 或 _ffn_forward eager fallback
  -> send_ffn_output
```

capture 顺序：

```text
Attention 发送 is_graph_capturing=True
  -> FFN recv dp_metadata_list
  -> GPUFFNModelRunner.capture_model(dp_metadata_list, is_attn_graph_capturing=True)
  -> _dummy_run(CUDAGraphMode.FULL, dp_metadata_list, ...)
  -> _ffn_forward(dp_metadata_list, is_graph_capturing=True)
  -> store graph in _cuda_graphs[graph_key]
```

这部分先尽量贴近原始实现。实施时必须用 GPU trace 验证 capture/replay 是否存在
不可重放的 connector 副作用；如果发现 NCCL send/recv 被 capture 后不能稳定 replay，
再把 FFN graph 收窄为 compute-only，而不是一开始就偏离原始代码。

### 5. Attention `FULL_DECODE_ONLY`

当前 DeepSeekV2 wrapper 在 `forward_with_afd()` 中逐层执行：

```text
attention compute -> send_attn_output -> recv_ffn_output -> next layer
```

Phase 6 只允许 `FULL_DECODE_ONLY`，因为 decode-only full graph 的 shape 集合更小，
更接近原始 AFD 验证路径，也更容易和 FFN graph key 对齐。

实施要求：

- normal prefill / mixed prefill-decode 仍可落到 eager。
- uniform decode 命中 `FULL_DECODE_ONLY` capture size 时走 graph。
- `_warmup_and_capture()` warmup 阶段向 FFN 发送 `is_warmup=True`。
- 正式 capture 阶段向 FFN 发送 `is_graph_capturing=True`。
- replay 阶段向 FFN 发送普通 DP metadata，让 FFN 用 graph key replay。

如果后续要支持 `FULL`、`FULL_AND_PIECEWISE` 或 `PIECEWISE`，必须单独开设计文档，
不要混入 Phase 6.1。

## 实施步骤

### Step 1：策略与校验

- 新增 `afd_plugin/runtime/cuda_graph.py`，放 CPU-safe policy、mode helper 和 FFN
  graph key helper。
- 替换 runner init 中的 `fail_if_cuda_graph_enabled()`，但保留旧函数作为兼容别名
  或测试入口。
- 添加 CPU tests：eager 允许、`FULL_DECODE_ONLY` 允许、`PIECEWISE` 拒绝、
  `FULL` / `FULL_AND_PIECEWISE` 拒绝、ubatching + graph 拒绝。

### Step 2：Attention dummy/capture metadata

- 在 `AFDAttentionModelRunner` 中保存 `_afd_is_graph_capturing`。
- 覆盖 `_warmup_and_capture()`，设置 `_is_warmup`。
- 修改 `_send_dp_metadata()` 传递 `is_graph_capturing`。
- 添加 CPU fake connector tests，验证 normal、warmup、capture 三种 flag。

### Step 3：FFN graph cache

- 迁移原始 `GPUFFNModelRunner` 的 graph cache 字段和 `_make_graph_key()`。
- 迁移原始 `_dummy_run()` / `capture_model()` / `_capture_graphs()` 的主要控制流。
- `execute_model()` 中按 graph key replay；miss 时沿用原始 eager fallback，并增加
  trace。
- 添加 CPU-level key tests；GPU unit test 用 tiny torch module 验证 capture/replay
  输出一致。

### Step 4：P2P buffer 与 shape

- 确认 `P2PAFDConnector.update_state_from_dp_metadata()` 预分配的
  `_recv_attn_buffers` 与 FFN graph replay 期望 shape 一致。
- 对 ratio > 1 的 FFN output split 增加 graph replay 后 shape 校验。
- trace 中输出 graph key、run mode、stage/layer、tensor shape。

### Step 5：GPU E2E 验证

新增 opt-in GPU cases：

- `1A1F` DeepSeekV2 eager baseline。
- `1A1F` DeepSeekV2 `FULL_DECODE_ONLY` + FFN graph cache。
- `2A2F` DeepSeekV2 `FULL_DECODE_ONLY` + FFN graph cache。
- 两路 ubatch eager 回归，确认 Phase 6 没破坏 Phase 5。

验收标准：

- graph 模式输出与 eager baseline token 序列一致。
- trace 中能看到 warmup、capture、replay 的 DP metadata key 一致。
- graph replay 期间没有新的 FFN graph capture。
- 没有要求 `--enforce-eager`；但非 `FULL_DECODE_ONLY` graph 组合有清楚的
  fail-fast 错误。

## 测试计划

CPU tests：

- `tests/test_cuda_graph_policy.py`
- `tests/test_attention_runtime.py` 增加 metadata flag assertions。
- `tests/test_ffn_runtime.py` 增加 graph key 和 miss policy。
- `tests/test_p2p_connector.py` 增加 `(metadata, is_graph_capturing, is_warmup)`
  payload compatibility。

GPU tests：

- `tests/test_gpu_e2e_deepseek_v2.py` 增加 graph marker case。
- 使用 `AFD_GPU_E2E_MODEL` opt-in。
- 默认先跑 `1A1F`；`2A2F` 作为第二层验证。
- 远程 L20X 验证按 AGENTS.md 流程使用临时分支，验证后删除本地和远程临时分支。

## 风险与待验证点

- vLLM `v0.19.1` 的 `CUDAGraphMode.FULL_DECODE_ONLY` 解析和当前 checkout 可能有
  patch-level 差异，实施前需要再次固定目标 commit/tag。
- 原始 FFN graph capture 逻辑是否能在 external plugin 的 P2P connector 下稳定
  replay，需要用 trace 和 GPU run 确认。
- FFN `compute_ffn_output()` 内部如果包含 TP/EP collective，graph cache 可能需要
  额外 communicator 支持；第一版限制 TP=1/PP=1 更稳。
- DP padding、ubatch padded token lens 与 connector tensor shape 必须完全一致，否则
  graph key 会命中但 replay buffer shape 错。
- CUDA graph memory profiling 可能低估 FFN graph cache，需要把 FFN graph memory
  计入 worker 可用显存预算或在 runbook 中要求保守 `gpu_memory_utilization`。

## 完成定义

- AFD 不再全局要求 `--enforce-eager`。
- Attention `FULL_DECODE_ONLY` + FFN graph cache 在 `1A1F` 和 `2A2F` GPU E2E 通过。
- Unsupported graph modes / ubatching graph 组合 fail fast 且错误信息明确。
- CPU smoke tests 在无 CUDA、无 vLLM wheel 的本地开发环境干净通过或干净 skip。
- 文档更新 README current status、GPU runbook 和已知限制。
