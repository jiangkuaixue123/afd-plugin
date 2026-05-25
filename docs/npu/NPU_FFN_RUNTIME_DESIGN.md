# NPU FFN Worker / ModelRunner 架构设计

本文档专门设计 NPU 版 AFD FFN runtime 的类关系和职责边界。重点是：

- NPU 版 `FFNWorker` / `FFNModelRunner` 如何继承 vLLM-Ascend；
- 如何复用当前 GPU FFN runtime 中已经沉淀的 AFD daemon loop 语义；
- 如何让 GPU 和 NPU FFN runtime 长期共存，而不是维护两套漂移的 AFD 实现。

本文档只覆盖 FFN 侧。Attention 侧见
`docs/npu/NPU_ATTENTION_RUNTIME_DESIGN.md`。

## 设计结论

推荐新增独立 NPU class path，不让 NPU FFN 直接继承当前插件的 GPU FFN 类：

```text
GPU:
  afd_plugin.v1.worker.AFDFFNWorker
    -> vllm.v1.worker.gpu_worker.Worker

  afd_plugin.v1.worker.GPUFFNModelRunner
    -> vllm.v1.worker.lora_model_runner_mixin.LoRAModelRunnerMixin

NPU:
  afd_plugin.v1.worker.ascend.AFDNPUFFNWorker
    -> vllm_ascend.worker.worker.NPUWorker

  afd_plugin.v1.worker.ascend.AFDNPUFFNModelRunner
    -> vllm_ascend.worker.model_runner_v1.NPUModelRunner
    -> vllm.v1.worker.gpu_model_runner.GPUModelRunner
```

核心原则和 Attention 侧一致：**不复用 GPU 的 worker/model runner 继承链，但复用
GPU 已经沉淀出来的 AFD runtime 语义**。

NPU 参考实现里 `NPUFFNModelRunner` 是
`class NPUFFNModelRunner(NPUModelRunner, GPUFFNModelRunner)`。这个事实说明 FFN
控制流和 GPU 版非常接近，但在 external plugin 里不建议直接继承当前
`GPUFFNModelRunner`。更好的长期形态是把可复用的 FFN daemon/metadata/graph-key
语义抽到 `AFDFFNRuntimeCoordinator` 或同等 helper，然后 GPU/NPU runner 分别继承
各自平台基类。

第一版为了快速打通 NPU runtime dummy 闭环，NPU runner 中可以保留少量重复 glue，
但这些重复应被标记为待抽取，不能长期分叉。

为了降低 NPU Worker / ModelRunner 的开发成本，Phase 1 先引入
`npudummyconnector`。它只负责让 FFN daemon loop 能收到 Attention 侧 metadata，并
完成本地 passthrough 或最小 `compute_ffn_output` 调用；不依赖真实 A2E/E2A、HCCL
或 CAM op。等 NPU Attention/FFN 生命周期稳定后，再接入 `camp2pconnector`。

## 复杂度判断

FFN Worker 相对简单，主要是生命周期 adapter：

```text
AFDNPUFFNWorker:
  低复杂度
  - 继承 NPUWorker
  - 创建 AFDNPUFFNModelRunner
  - 返回空 KV cache spec
  - 启停 FFN daemon loop
  - scheduler-driven execute_model fail fast
```

FFN ModelRunner 是主要复杂度所在：

```text
AFDNPUFFNModelRunner:
  中高复杂度
  - 继承 NPUModelRunner
  - Phase 1 创建 npudummyconnector，后续切换到 camp2pconnector
  - 接收 dp_metadata_list
  - per-layer / per-stage recv Attention output
  - 建立 Ascend forward context
  - 调 model.compute_ffn_output(...)
  - send FFN output
  - 处理 AFDRecvOutput payload
  - 后续再处理 ACL graph / ubatching
```

Worker 不应承载 connector payload、AFD metadata 或 MoE 计算语义。

## Worker 设计

### 继承关系

```python
class AFDNPUFFNWorker(NPUWorker):
    afd_expected_role = "ffn"
```

必须继承 `vllm_ascend.worker.worker.NPUWorker`，这样才能复用：

- vllm-ascend 的 patch 注册和 op 注册；
- Ascend config 初始化；
- NPU device/distributed 初始化；
- workspace manager；
- vllm-ascend worker 的 load model、profile、shutdown、sleep mode 等生命周期。

### `__init__`

`__init__` 只做轻量兼容入口，不承载 AFD 业务：

```python
class AFDNPUFFNWorker(_NPUWorker):
    def __init__(self, *args, **kwargs):
        ensure_ascend_runtime_available()
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)
        self._ffn_thread = None
        self._ffn_shutdown_event = None
        self._ffn_loop_error = None
```

如果必须 patch vLLM-Ascend，统一通过 `afd_plugin.compat.ascend` 中的幂等 helper
完成。`NPUWorker` 自身需要的 `adapt_patch()` 仍由 vllm-ascend 负责。

### `init_device`

推荐第一版不要调用 `super().init_device()` 后再替换 runner。原因和 Attention 侧
相同：`NPUWorker.init_device()` 会创建原生 `NPUModelRunner`，初始化成本高，也可能
提前建立不适合 AFD FFN 的状态。

推荐结构：

```python
def init_device(self):
    assert_compatible_afd_stack(
        self.vllm_config,
        caller="AFDNPUFFNWorker.init_device",
        expected_role="ffn",
    )
    fail_if_unsupported_npu_afd_features(self.vllm_config)

    if self.use_v2_model_runner:
        raise RuntimeError("AFD NPU FFN supports only vllm-ascend MRv1 first")

    self.device = self._init_device()
    init_ascend_workspace_for_afd(self.device, num_ubatches=1)
    self.model_runner = AFDNPUFFNModelRunner(self.vllm_config, self.device)
```

`init_ascend_workspace_for_afd()` 应收敛到 `afd_plugin.compat.ascend.runtime`，避免
NPU worker 模块散落 vllm-ascend 强绑定 import。

### KV cache 和 EngineCore

FFN 侧不需要 KV cache，NPU FFN worker 应和 GPU FFN worker 一样返回空 spec：

```python
def get_kv_cache_spec(self) -> dict:
    return {}
```

`initialize_from_config(...)` 不分配 KV cache，而是作为启动 FFN runtime 的 hook：

```python
def initialize_from_config(self, kv_cache_config):
    self.model_runner.initialize_kv_cache(kv_cache_config)
    self.model_runner.initialize_afd_connector()
    self.start_ffn_server_loop()
```

如果 vLLM / vLLM-Ascend 的 EngineCore 对空 KV cache 有额外假设，相关兼容逻辑仍应
放在 `afd_plugin.compat.patches` 或 `afd_plugin.compat.ascend`，不进入 worker 业务逻辑。

### daemon loop

FFN daemon loop 由 Worker 管理线程生命周期，ModelRunner 负责实际执行：

```text
AFDNPUFFNWorker.initialize_from_config(...)
  -> model_runner.initialize_afd_connector()
  -> start_ffn_server_loop()

loop:
  torch.npu.set_device(...)
  while not shutdown:
    dp_metadata_list, is_graph_capturing, is_warmup =
      connector.recv_dp_metadata_list(...)

    if graph path is supported and capture/warmup:
      model_runner.capture_model(...)
    else:
      model_runner.execute_ffn_step(...)

    torch.npu.synchronize()
```

第一版建议给 ModelRunner 一个语义化方法：

```python
execute_ffn_step(
    dp_metadata_list=...,
    is_graph_capturing=False,
    is_warmup=False,
)
```

`execute_model(...)` 可以保留为兼容 vLLM/vllm-ascend 的方法，但只接受 connector
driven 调用；如果没有 `dp_metadata_list`，应 fail fast。

### 不覆盖的 Worker 方法

除 FFN daemon 必需入口外，尽量继承 `NPUWorker` 原生行为：

- `load_model()`
- `determine_available_memory()`，空 KV 路径下理论上不应进入；
- `sample_tokens()`
- LoRA / profile / sleep mode / shutdown 的大部分平台逻辑

普通 scheduler request 不应驱动 FFN：

```python
def execute_model(self, scheduler_output):
    raise RuntimeError(
        "AFD NPU FFN workers are connector-driven; scheduler-driven "
        "execute_model() is not supported."
    )
```

## ModelRunner 设计

### 继承关系

```python
class AFDNPUFFNModelRunner(NPUModelRunner):
    afd_expected_role = "ffn"
```

必须继承 `vllm_ascend.worker.model_runner_v1.NPUModelRunner`，原因是 FFN 侧虽然不走
普通 request，但仍需要 vllm-ascend 的平台状态：

- model load 和 Ascend memory helper；
- `parallel_config`、`scheduler_config`、`max_num_tokens`、`uniform_decode_query_len`；
- Ascend fused MoE / token dispatcher 所需 forward context；
- ACL graph 相关 dispatcher 和 graph pool；
- NPU profiler 和设备同步语义。

不建议继承当前插件的 `GPUFFNModelRunner`，因为该类绑定 CUDA graph、CUDA forward
context 和 GPU connector payload 假设。

### `__init__`

推荐顺序：

```python
class AFDNPUFFNModelRunner(_NPUModelRunner):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self.afd_config = parse_afd_config(vllm_config, expected_role="ffn")
        fail_if_unsupported_npu_afd_features(vllm_config)

        self.afd_state = AFDFFNRuntimeCoordinator(
            vllm_config=vllm_config,
            afd_config=self.afd_config,
            device=device,
            role="ffn",
        )
        self.connector = self.afd_state.create_connector()
        self.model_memory_usage = 0
        self.num_layers = resolve_num_hidden_layers(self.model_config)
```

connector 可以在 `__init__` 中创建，但初始化建议由 worker 的
`initialize_from_config()` 触发，和 GPU FFN worker 保持一致。若 vllm-ascend 某些
状态要求提前拿到 `attn_size` / `ffn_size`，可以在 `initialize_afd_connector()` 后
回填到 runner。

### 空 KV / 兼容接口

NPU FFN runner 应提供和 GPU FFN runner 一致的兼容接口：

```python
def get_kv_cache_spec(self) -> dict:
    return {}

def initialize_kv_cache(self, kv_cache_config):
    return None

def profile_run(self):
    return None

def sample_tokens(...):
    raise RuntimeError("FFN runners do not sample tokens")
```

LoRA、pooling、draft tokens、tensorized save 等接口也可以先保持 no-op 或 fail fast，
与 GPU FFN runner 对齐。

### connector-driven 执行

推荐执行入口：

```python
def execute_ffn_step(
    self,
    *,
    dp_metadata_list: dict[int, Any],
    is_graph_capturing: bool = False,
    is_warmup: bool = False,
) -> None:
    if dp_metadata_list is None:
        raise RuntimeError("AFD NPU FFN requires dp_metadata_list")
    return self._ffn_forward(dp_metadata_list=dp_metadata_list)
```

兼容方法：

```python
def execute_model(self, scheduler_output=None, intermediate_tensors=None, *,
                  dp_metadata_list=None, is_graph_capturing=False,
                  is_warmup=False):
    if dp_metadata_list is None:
        raise RuntimeError("AFD NPU FFN is connector-driven")
    return self.execute_ffn_step(...)
```

### `_ffn_forward`

NPU FFN forward 的控制流和 GPU FFN runner 相近：

```text
for layer_idx in layers:
  for stage_idx / ubatch_idx:
    connector_data = connector.create_recv_metadata(...)
    recv_output = connector.recv_attn_output(metadata=connector_data, ...)
    connector.update_metadata(connector_data, recv_output)

    with Ascend forward context:
      rank_ffn_output = model.compute_ffn_output(
        hidden_states=recv_output.hidden_states,
        layer_idx=layer_idx,
        group_list=recv_output.group_list,
        topk_weights=recv_output.topk_weights,
        topk_ids=recv_output.topk_ids,
        router_logits=recv_output.router_logits,
        row_idx=recv_output.row_idx,
        x_active_mask=recv_output.x_active_mask,
        cam_p2p_ep_name=recv_output.cam_p2p_ep_name,
      )

    connector.send_ffn_output(rank_ffn_output, connector_data, ...)
```

Phase 1 应只支持 `npudummyconnector` 和单流 dummy/passthrough 闭环；接入
`camp2pconnector` 后仍保持单流通信：

- 不支持 FFN comm stream/event；
- 不支持 `quant_mode != 0`，`dynamic_scales` 不透传；
- 不支持 `compute_gate_on_attention=true`；
- 不支持权重加载裁剪；
- 通信 backend 细节只在 connector 内部处理。

### Ascend forward context

FFN 侧不是普通 `NPUModelRunner.execute_model()`，因此需要自己建立最小 Ascend
forward context，供 Ascend fused MoE / model wrapper 读取：

```python
with set_ascend_forward_context(
    attn_metadata=None,
    vllm_config=self.vllm_config,
    batch_descriptor=None,
    aclgraph_runtime_mode=aclgraph_runtime_mode,
    model_instance=self.model,
    num_tokens=num_tokens,
    num_tokens_across_dp=num_tokens_across_dp,
):
    install_afd_metadata_on_forward_context(...)
    output = self.model.compute_ffn_output(...)
```

AFD metadata 的 canonical 存储仍建议走
`forward_context.additional_kwargs["afd_metadata"]`。如果 vllm-ascend patch 或模型
必须读取 `forward_context.afd_metadata`，由 `afd_plugin.compat.ascend` 提供受控
mirror helper，不在 runner 中散落 `setattr`。

### AFDRecvOutput payload

`camp2pconnector.recv_attn_output(...)` 返回 `AFDRecvOutput` 风格对象。NPU FFN
runner 应把 payload 当作 connector/model contract，而不是拆散到 Worker：

- `hidden_states`：必需；
- `topk_weights` / `topk_ids`：当前 `compute_gate_on_attention=false` 下通常为空；
- `group_list`、`router_logits`、`row_idx`、`x_active_mask`：按模型 wrapper 需要透传；
- `dynamic_scales`：`quant_mode` 当前不移植，第一版不启用；
- `cam_p2p_ep_name`：`camp2pconnector` / Ascend MoE backend 可用；
- `handle` / `atten_batch_size` 等 connector data：由 `connector.update_metadata(...)`
  写回 metadata 后供 `send_ffn_output(...)` 使用。

### DP metadata 和非等 A/F

NPU FFN runner 应复用 GPU FFN 的 graph key / DP metadata 思路：

```text
dp_metadata_list
  -> make key: ((stage_idx, tuple(num_tokens_across_dp_cpu)), ...)
  -> connector.update_state_from_dp_metadata(...)
  -> create_recv_metadata(...)
```

`camp2pconnector` 支持非等 A/F 路由。A > F 且整除时，FFN 侧需要把多个 Attention rank
的 token 数合并为本 FFN rank 的 `batch_size`。这部分应封装在 connector 或
coordinator helper 中，runner 只调用 `create_recv_metadata(...)`，避免把 topology
计算散落到 model runner。

## graph / warmup / capture

第一版建议以 eager 单流通信闭环为主，ACL graph 后续单独支持。

在 ACL graph 未正式支持前：

- 如果 `use_aclgraph` 为 true 且会 capture AFD 通信副作用，应 fail fast；
- 不复用 GPU `cuda_graph.py` 的 CUDA graph policy；
- 不把 connector recv/send capture 进 ACL graph；
- 可以保留 `dp_metadata_key` 生成 helper，供后续 graph cache 使用。

后续支持 ACL graph 时，NPU FFN runner 应复用 vllm-ascend graph 机制：

```text
Attention sends dp_metadata_list with is_warmup / is_graph_capturing
  -> Worker daemon loop receives flags
  -> ModelRunner.capture_model(dp_metadata_list, is_warmup, ...)
  -> warmup: eager _ffn_forward
  -> capture: NPUGraph + _ffn_forward
  -> normal: replay if graph exists, otherwise eager fallback
```

graph key 应只由实际 shape / DP metadata 决定。`quant_mode` 当前不移植，不参与
graph/hash。

## GPU/NPU 共存原则

### 命名共存

GPU 现有公开 class path 保持不变：

```text
afd_plugin.v1.worker.AFDFFNWorker
afd_plugin.v1.worker.GPUFFNModelRunner
```

NPU 新增 class path：

```text
afd_plugin.v1.worker.ascend.AFDNPUFFNWorker
afd_plugin.v1.worker.ascend.AFDNPUFFNModelRunner
```

### import 共存

`afd_plugin.v1.worker.__init__` 不应 import NPU 类，避免没有 vllm-ascend /
torch-npu 的环境 import 失败。NPU 类只在显式导入
`afd_plugin.v1.worker.ascend` 时解析。

NPU 模块内部使用 lazy import：

```python
_NPUWorker, _NPUWorker_IMPORT_ERROR = optional_class(
    "vllm_ascend.worker.worker",
    "NPUWorker",
)
```

### 逻辑共存

不要让 NPU 类继承当前插件的 GPU FFN 类。共享层应该是纯 AFD FFN helper：

```text
                         +--------------------------+
                         | AFD FFN coordinator      |
                         | config / connector / DP  |
                         | graph key / loop state   |
                         +------------+-------------+
                                      |
          +---------------------------+---------------------------+
          |                                                       |
+---------v-----------+                              +------------v------------+
| GPU FFN runner      |                              | NPU FFN runner          |
| -> plugin GPU path  |                              | -> NPUModelRunner       |
+---------------------+                              +-------------------------+
```

共享 coordinator 应覆盖：

- `AFDConfig` 解析和 role validation；
- connector 创建、初始化和关闭；
- `dp_metadata_list` key 构造；
- `update_state_from_dp_metadata()`；
- FFN daemon loop 中的 warmup/capture/normal 标记解释；
- 空 KV cache 兼容接口；
- `AFDRecvOutput` 到通用 compute payload 的整理。

平台 runner 只保留平台 glue：

```text
GPU runner:
  CUDA forward context
  CUDA graph
  GPU connector payload

NPU runner:
  Ascend forward context
  ACL graph
  AFDRecvOutput / camp2p payload
```

## 第一版支持边界

第一版 NPU FFN runtime 建议支持：

- vLLM `v0.19.1`；
- vLLM-Ascend `v0.19.1rc1`；
- `--additional-config '{"afd": ...}'`；
- `vllm serve` + `--worker-cls`；
- `npudummyconnector`，用于 NPU dummy run 和 FFN daemon loop 调试；
- 后续接入 `camp2pconnector`；
- connector-driven FFN daemon loop；
- 空 KV cache；
- 单流 eager 通信闭环；
- 完整权重加载。

第一版明确不支持：

- `vllm fserver`；
- `compute_gate_on_attention=true`；
- `quant_mode != 0`；
- Attention/FFN 通信多流；
- 权重加载裁剪；
- vllm-ascend model runner v2；
- scheduler-driven FFN request；
- 把通信 backend 暴露到 runtime 层。

## 后续实现步骤

1. 新建 `afd_plugin.v1.worker.ascend` 包，先提供 CPU-safe import skeleton。
2. 新建或扩展 `afd_plugin.compat.ascend`，提供 NPUWorker/NPUModelRunner lazy
   import、workspace helper、forward context mirror helper 和受控 patch 入口。
3. 实现 `AFDNPUFFNWorker(NPUWorker)`，覆盖 `init_device()`、空 KV cache、
   `initialize_from_config()`、daemon loop、scheduler-driven `execute_model()` fail fast
   和 shutdown。
4. 实现 `AFDNPUFFNModelRunner(NPUModelRunner)`，先支持 eager 单流
   `execute_ffn_step()`。
5. 实现 `npudummyconnector` 的 FFN 侧方法，至少覆盖 `recv_dp_metadata_list()`、
   `recv_attn_output()`、`send_ffn_output()` 和关闭逻辑。
6. 用 `npudummyconnector` 跑通 NPU FFN daemon loop，验证
   `recv -> compute_ffn_output/passthrough -> send` 的最小闭环。
7. 接入 `camp2pconnector` 的 `create_recv_metadata()`、`recv_attn_output()`、
   `update_metadata()`、`send_ffn_output()`。
8. 从 GPU `GPUFFNModelRunner` 抽出设备无关 `AFDFFNRuntimeCoordinator`，减少
   GPU/NPU 重复逻辑。
9. 增加 Ascend-gated import、validation、daemon loop 和最小 multi-process 通信测试。
10. connector 闭环稳定后，再单独设计 ACL graph、ubatching 和性能 profiling。
