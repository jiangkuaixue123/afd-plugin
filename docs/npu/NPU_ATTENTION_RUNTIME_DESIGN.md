# NPU Attention Worker / ModelRunner 架构设计

本文档专门设计 NPU 版 AFD Attention runtime 的类关系和职责边界。重点是：

- NPU 版 `AttentionWorker` / `AttentionModelRunner` 如何继承 vLLM-Ascend；
- 如何复用当前 GPU AFD runtime 中已经沉淀的配置、connector、metadata 能力；
- 如何避免 GPU 和 NPU 两套 worker/model runner 互相污染。

本文档只覆盖 Attention 侧。FFN 侧可以沿用同样的分层原则，但需要单独设计 daemon
loop、FFN graph 和 `AFDRecvOutput` payload。

## 设计结论

推荐新增独立 NPU class path，而不是让现有 GPU 类同时兼容 Ascend：

```text
GPU:
  afd_plugin.v1.worker.AFDAttentionWorker
    -> vllm.v1.worker.gpu_worker.Worker

  afd_plugin.v1.worker.AFDAttentionModelRunner
    -> vllm.v1.worker.gpu_model_runner.GPUModelRunner

NPU:
  afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker
    -> vllm_ascend.worker.worker.NPUWorker

  afd_plugin.v1.worker.ascend.AFDNPUAttentionModelRunner
    -> vllm_ascend.worker.model_runner_v1.NPUModelRunner
    -> vllm.v1.worker.gpu_model_runner.GPUModelRunner
```

核心原则是：**不复用 GPU 的 worker/model runner 继承链，但复用 GPU 已经沉淀出来的
AFD runtime 语义**。

GPU 和 NPU 共享 AFD 的配置解析、validation、connector factory、metadata 构造和
DP metadata 发送策略，但不共享 vLLM/vLLM-Ascend 的 worker/model runner 继承链。
共享逻辑优先放到普通 helper/coordinator 中，而不是通过跨设备多重继承解决。

第一版为了打通功能，可以在 NPU runner 中保留少量重复的 AFD glue；但这些重复必须
被标记为待抽取，后续 `camp2pconnector` 通信闭环稳定后，应收敛到共享
`AFDAttentionRuntimeCoordinator` 或同等 helper，避免 GPU/NPU 两份 AFD 语义长期漂移。

## 为什么不能直接复用 GPU 类

当前 GPU 版 `AFDAttentionWorker` 的做法是：

```text
AFDAttentionWorker.init_device()
  -> super().init_device()
  -> native GPUModelRunner 被创建
  -> 替换为 AFDAttentionModelRunner
```

这对 CUDA 路径是可接受的，因为 worker 和 model runner 都来自 vLLM 原生 GPU 栈。

NPU 侧不适合这样继承 GPU worker：

- `vllm_ascend.worker.worker.NPUWorker` 不是 `GPUWorker` 的简单子类，而是直接继承
  `WorkerBase`，并在 `__init__` / `_init_device` / `init_device` 中注册 Ascend patch、
  custom op、Ascend config、NPU distributed environment、workspace manager 等；
- `vllm_ascend.worker.model_runner_v1.NPUModelRunner` 虽然继承 vLLM
  `GPUModelRunner`，但它在 `__init__` 和 `execute_model()` 里维护 Ascend sampler、
  attention backend、ACL graph、PCP/DCP、Ascend forward context 和 NPU buffer；
- 如果 NPU runtime 继承当前 GPU `AFDAttentionWorker`，就会绕开
  `NPUWorker` 生命周期，导致 vllm-ascend 的平台初始化不完整；
- 如果 NPU runner 继承当前 GPU `AFDAttentionModelRunner`，则会复用 CUDA graph、
  CUDA ubatch wrapper 和 GPU forward context 假设，和 Ascend 路径冲突。

因此，NPU 版应该继承 vllm-ascend 的 NPU 类，只复用 AFD 业务 helper。

## 推荐目录与 class path

建议把 Ascend/NPU runtime 放到独立子包，避免顶层 worker import 时引入
`vllm_ascend` 或 `torch_npu`：

```text
afd_plugin/
  compat/
    ascend/
      __init__.py
      imports.py
      patches.py
      runtime.py
  v1/
    worker/
      attention_worker.py          # 当前 GPU 版
      attention_model_runner.py    # 当前 GPU 版
      attention_runtime.py         # 设备无关 AFD helper，可后续抽取
      ascend/
        __init__.py
        attention_worker.py
        attention_model_runner.py
```

外部启动使用显式 class path：

```bash
vllm serve ... \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --additional-config '{"afd": {"enabled": true, "role": "attention", "connector": "camp2pconnector"}}'
```

不引入 `vllm fserver`，也不使用 `vllm.afd_connectors` entry point。NPU 路径只注册
并验证真实 connector `camp2pconnector`；已移除早期开发调试用 dummy connector。

## Worker 设计

### 继承关系

```python
class AFDNPUAttentionWorker(NPUWorker):
    afd_expected_role = "attention"
```

`AFDNPUAttentionWorker` 必须继承 `vllm_ascend.worker.worker.NPUWorker`，这样才能复用：

- vllm-ascend 的 patch 注册和 op 注册；
- Ascend config 初始化；
- NPU device/distributed 初始化；
- NPU worker 的 profiling、KV cache、PP/SP、sleep mode 等生命周期。

### `__init__`

`__init__` 只做轻量 AFD validation 前置检查，主要生命周期交给 `NPUWorker.__init__`。
如果需要对 vLLM-Ascend 做 AFD-specific patch，必须通过
`afd_plugin.compat.ascend` 中的幂等 helper 完成。

推荐结构：

```python
class AFDNPUAttentionWorker(_NPUWorker):
    def __init__(self, *args, **kwargs):
        ensure_ascend_runtime_available()
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)
```

这里的 `apply_afd_ascend_patches_if_needed()` 只能做 AFD-specific patch；vllm-ascend
自身已有的 `adapt_patch()` 仍由 `NPUWorker` 负责。

### `init_device`

推荐第一版不要调用 `super().init_device()` 后再替换 runner，因为
`NPUWorker.init_device()` 会先创建原生 `NPUModelRunner`，这个 runner 初始化成本高，
也可能提前建立不适合 AFD 的状态。

更好的做法是复用 `NPUWorker` 的 device 初始化步骤，但直接创建 AFD runner：

```python
def init_device(self):
    assert_compatible_afd_stack(
        self.vllm_config,
        caller="AFDNPUAttentionWorker.init_device",
        expected_role="attention",
    )
    fail_if_unsupported_npu_afd_features(self.vllm_config)

    if self.use_v2_model_runner:
        raise RuntimeError("AFD NPU Attention supports only vllm-ascend MRv1 first")

    self.device = self._init_device()
    init_ascend_workspace_for_afd(self.device, num_ubatches=1)
    self.model_runner = AFDNPUAttentionModelRunner(self.vllm_config, self.device)
```

`init_ascend_workspace_for_afd()` 应封装在 `afd_plugin.compat.ascend.runtime` 中，
内部可以调用 vllm-ascend 的 `init_workspace_manager`。这样 vllm-ascend 强绑定逻辑
集中在 compat 层。

### 不覆盖的 worker 方法

第一版应尽量继承 `NPUWorker` 的以下行为，不做 AFD override：

- `load_model()`
- `determine_available_memory()`
- `initialize_from_config()`
- `compile_or_warm_up_model()`
- `execute_model()`
- `sample_tokens()`
- LoRA / KV cache / profile / sleep mode 相关方法

Attention 侧 AFD 只需要在 model runner forward 前注入 metadata；worker 不应重新实现
调度、pipeline parallel 或 NPU memory profiling。

## ModelRunner 设计

### 继承关系

```python
class AFDNPUAttentionModelRunner(NPUModelRunner):
    afd_expected_role = "attention"
```

必须继承 `vllm_ascend.worker.model_runner_v1.NPUModelRunner`，原因是 NPU
`execute_model()` 已经负责：

- scheduler output 到 NPU input batch 的转换；
- Ascend attention metadata 和 attention backend；
- `set_ascend_forward_context(...)`；
- ACL graph dispatch；
- PCP/DCP、sequence parallel、spec decode、sampler 等平台行为。

AFD runner 只在这些平台行为之上增加 AFD metadata 和 connector 控制面。

### `__init__`

推荐顺序：

```python
class AFDNPUAttentionModelRunner(_NPUModelRunner):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self.afd_config = parse_afd_config(vllm_config, expected_role="attention")
        fail_if_unsupported_npu_afd_features(vllm_config)
        self.afd_state = AFDAttentionRuntimeCoordinator(
            vllm_config=vllm_config,
            afd_config=self.afd_config,
            device=device,
            role="attention",
        )
        self.afd_connector = self.afd_state.create_connector()
        self.afd_connector.init_afd_connector()
```

`AFDAttentionRuntimeCoordinator` 不是必须立即存在，但建议后续从当前 GPU
`AFDAttentionModelRunner` 中抽出设备无关逻辑，避免 GPU/NPU 两边复制：

- AFD config parse / validation；
- DP rank 派生；
- connector 创建和初始化；
- `AFDMetadata` 构造；
- DP metadata list 构造；
- `send_dp_metadata_list()` 和 `update_state_from_dp_metadata()`；
- transaction id。

### `_model_forward`

vllm-ascend 的 `NPUModelRunner.execute_model()` 会在调用 `_model_forward()` 前建立
`set_ascend_forward_context(...)`。因此 NPU AFD runner 的最小插入点应是
`_model_forward()`：

```python
def _model_forward(self, *args, **kwargs):
    forward_context = get_forward_context()
    self.afd_state.install_metadata_on_forward_context(forward_context)
    return super()._model_forward(*args, **kwargs)
```

`install_metadata_on_forward_context()` 的规则：

- canonical 存储仍优先使用 `forward_context.additional_kwargs["afd_metadata"]`；
- 如果 NPU model wrapper 或 vllm-ascend patch 必须读取 `forward_context.afd_metadata`，
  由 `afd_plugin.compat.ascend` 提供受控 mirror helper；
- 不在 connector 或 model runner 中散落 `setattr(forward_context, ...)`；
- 不启用 `afd_comm_stream` / `afd_comm_event` 相关通信多流。

### DP metadata

NPU runner 应复用当前 GPU 侧的 DP metadata 语义：

```text
forward_context.dp_metadata
  -> dp_metadata_list
  -> connector.update_state_from_dp_metadata(...)
  -> connector.send_dp_metadata_list(...)
```

如果 `forward_context.dp_metadata` 不存在且 DP size 为 1，可以使用 GPU 侧已有的
fallback 逻辑构造 `AFDDPMetadata`。如果 DP size 大于 1 且缺失 DP metadata，必须
fail fast。

### graph / warmup / capture

第一版建议只保证 eager / 单流通信闭环。ACL graph 支持单独阶段处理。

在 ACL graph 支持前：

- 如果 vLLM-Ascend graph mode 会导致 AFD metadata 只在 capture 时发送一次，应该
  fail fast；
- 不把 connector 通信副作用 capture 进 ACL graph；
- 不复用 GPU `cuda_graph.py` 的策略类作为 NPU graph 策略。

后续支持 ACL graph 时，NPU runner 应复用 vllm-ascend 的 graph dispatch，只补充
AFD metadata 的 warmup/capture/normal 标记：

```text
NPUModelRunner graph dispatch
  -> AFDNPUAttentionModelRunner marks AFD warmup/capture state
  -> connector sends DP metadata with is_warmup / is_graph_capturing
  -> FFN side decides warmup, capture, or normal execution
```

## GPU/NPU 共存原则

### 命名共存

GPU 现有公开 class path 保持不变：

```text
afd_plugin.v1.worker.AFDAttentionWorker
afd_plugin.v1.worker.AFDAttentionModelRunner
```

NPU 新增 class path，不覆盖 GPU 名称：

```text
afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker
afd_plugin.v1.worker.ascend.AFDNPUAttentionModelRunner
```

这样用户启动脚本能通过 class path 明确选择设备 runtime。

### import 共存

`afd_plugin.v1.worker.__init__` 不应 import NPU 类，避免没有 `vllm_ascend` /
`torch_npu` 的 GPU 或 CPU 环境 import 失败。NPU 类只在显式导入
`afd_plugin.v1.worker.ascend` 时解析。

NPU 模块内部也应使用 lazy import / optional import：

```python
_NPUWorker, _NPUWorker_IMPORT_ERROR = optional_class(
    "vllm_ascend.worker.worker",
    "NPUWorker",
)
```

如果缺少 vllm-ascend，只有用户显式使用 NPU class path 时才报清晰错误。

### 逻辑共存

不要让 GPU 类继承 NPU 类，也不要让 NPU 类继承 GPU AFD 类。共享层应该是纯 AFD
helper，而不是平台 runner：

```text
                       +-----------------------------+
                       | AFD attention helper/state  |
                       | config / connector / meta   |
                       +--------------+--------------+
                                      |
          +---------------------------+---------------------------+
          |                                                       |
+---------v-----------+                              +------------v------------+
| GPU AFD runner      |                              | NPU AFD runner          |
| -> GPUModelRunner   |                              | -> NPUModelRunner       |
+---------------------+                              +-------------------------+
```

长期维护目标是让 runner 只保留平台 glue：

```text
GPU runner:
  从 vLLM GPU forward context 取信息
  调用共享 AFD coordinator
  调用 super()._model_forward()

NPU runner:
  从 Ascend forward context 取信息
  调用同一个共享 AFD coordinator
  调用 super()._model_forward()
```

共享 coordinator 应覆盖：

- `AFDConfig` 解析和 role validation；
- connector 创建、初始化和关闭；
- `AFDMetadata` 构造；
- DP metadata list 构造；
- `update_state_from_dp_metadata()` 和 `send_dp_metadata_list()`；
- warmup / capture / normal 的 AFD 状态标记；
- forward context 写入策略。

这些逻辑不应长期停留在 GPU/NPU 两个 runner 的重复代码中。

### patch 共存

如果必须 patch vLLM-Ascend，patch 只允许进入 `afd_plugin.compat.ascend`：

- patch 必须幂等；
- patch 必须检查 vLLM-Ascend 版本；
- patch 必须只处理 AFD-specific 行为；
- runtime、connector、model runner 不能各自散落 monkey patch。

## 第一版支持边界

第一版 NPU Attention runtime 建议支持：

- vLLM `v0.19.1`；
- vLLM-Ascend `v0.19.1rc1`；
- `--additional-config '{"afd": ...}'`；
- `camp2pconnector`；
- 单流通信；
- 完整权重加载；
- eager 路径或经验证不会 capture 通信副作用的最小 graph 路径。

第一版明确不支持：

- `vllm fserver`；
- `compute_gate_on_attention=true`；
- `quant_mode != 0`；
- Attention/FFN 通信多流；
- 权重加载裁剪；
- vllm-ascend model runner v2；
- 把通信 backend 暴露到 runtime 层。

## 后续实现步骤

1. 新建 `afd_plugin.compat.ascend`，提供 vllm-ascend lazy import、版本检查、
   workspace 初始化 helper 和受控 patch 入口。
2. 新建 `afd_plugin.v1.worker.ascend` 包，先提供 CPU-safe import skeleton。
3. 实现 `AFDNPUAttentionWorker(NPUWorker)`，直接创建
   `AFDNPUAttentionModelRunner`，不先创建原生 `NPUModelRunner` 再替换。
4. 实现 `AFDNPUAttentionModelRunner(NPUModelRunner)`，只覆盖 `__init__`、
   `_model_forward` 和必要的 AFD metadata helper。
5. 接入 `camp2pconnector`，跑通 Attention 侧 connector 创建、`AFDMetadata`
   构造、`dp_metadata_list` 发送和 forward context 注入。
6. 使用 `camp2pconnector` 验证 worker/model runner 生命周期、class path 和真实
   A2E/E2A 通信。
7. 第一版允许 NPU runner 内少量重复 AFD glue，用于快速收敛真实通信闭环。
8. 从 GPU `AFDAttentionModelRunner` 抽出设备无关
   `AFDAttentionRuntimeCoordinator`，逐步减少两边重复逻辑。
9. 增加 Ascend-gated import、validation 和最小 multi-process 通信测试。
