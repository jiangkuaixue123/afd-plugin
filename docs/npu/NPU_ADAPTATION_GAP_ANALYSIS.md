# NPU AFD 适配差异盘点与决策底稿

本文档记录把 Ascend/NPU 版本 AFD 归一迁移到 `afd-plugin` 时已经观察到的主要
差异点。目标是先把事实和决策面摊开，方便后续逐项解释、取舍和重排阶段。

本文档不定义最终 API，也不代表实现方案已经确定。后续设计讨论应优先更新本文档
中的判断，再进入具体代码迁移。

## 参考基线

- 当前插件仓库：`/Users/jiangkuaixue/code/afd-plugin`
- 当前插件目标 vLLM：`v0.19.1`
- NPU 适配目标 vLLM-Ascend：`v0.19.1rc1`
- NPU AFD 0.13 vLLM 主仓参考：`../vllm` 分支 `afd_v0.13.0_jcz_dev`
- NPU AFD 0.13 vLLM-Ascend 参考：`../vllm-ascend` 分支
  `upstream/afd_v0.13.0_release`
- vLLM 0.13 基线：`../vllm` 的 `upstream/releases/v0.13.0`
- vLLM-Ascend 0.13 基线：`../vllm-ascend` 的 `origin/releases/v0.13.0`

## 已决策事项

- **版本基线已决策**：`afd-plugin` 只支持 vLLM `v0.19.1`；NPU 适配只支持
  vLLM-Ascend `v0.19.1rc1`。`0.13.0` 分支仅作为原始 NPU AFD 行为参考，不作为
  本插件的运行兼容目标。
- **配置入口已决策**：AFD 配置只通过 vLLM 原生
  `--additional-config '{"afd": ...}'` 传入。不新增也不保留 `--afd-config` 作为
  插件入口。
- **NPU connector 范围已决策**：当前文档只分析和迁移 `camp2pconnector`。
- **connector 注册机制已决策**：沿用当前 `afd-plugin` 的 plugin-owned
  `AFDConnectorFactory.register_connector(...)` 机制。`camp2pconnector` 迁入插件后
  由插件内静态注册，不使用 `vllm.afd_connectors` entry point，也不依赖
  vllm-ascend 桥接注册。
- **Ascend extension 构建边界已决策**：`afd-plugin` 负责构建迁移所需的 Ascend
  extension/custom op，不依赖 vllm-ascend 继续提供 AFD 专用 extension。
- **patch 策略已决策**：如果 NPU 迁移确实需要对 vLLM-Ascend 做 AFD-specific
  patch，统一收敛到 `afd_plugin.compat.ascend`。patch 必须幂等、版本受保护，并且
  不散落在 runtime、connector 或 model 模块里。
- **CLI 启动方式已决策**：NPU 侧和当前插件保持一致，使用 `vllm serve` +
  显式 class path / `--worker-cls` 接入；不使用也不恢复 `vllm fserver`。
- **`compute_gate_on_attention` 已决策**：当前 NPU 迁移不迁移该能力，暂不支持。
  如果配置开启，应 fail fast。
- **权重加载裁剪已决策**：当前 NPU 迁移暂不支持按 Attention/FFN role 裁剪加载
  MoE/gate 权重，第一版保持完整权重加载；后续作为独立能力再支持。
- **`quant_mode` 已决策**：当前 NPU 迁移不移植 `quant_mode`。`camp2pconnector`
  参考实现中该字段实际写死为 `0`；如果用户显式配置非 `0`，应 fail fast。
- **AFD 通信多流已决策**：当前不支持 event/stream 相关的 Attention 通信多流和
  FFN 通信多流。若配置启用 `is_attn_multistream`、`is_ffn_multistream`、
  `is_multistream` 或依赖 `afd_comm_stream` / `afd_comm_event` 的通信多流路径，应
  fail fast。
- **配置字段 schema 已决策**：第一版采用通用字段 + `extra_config` 全兼容方案。
  NPU 旧字段优先放入 `additional_config["afd"]["extra_config"]`，先降低打通
  `camp2pconnector` 的迁移成本；功能稳定后再考虑收紧 schema。

## 总体结论

当前 `afd-plugin` 是一个面向 vLLM `v0.19.1` 的 external plugin，运行时以
GPU/CUDA/NCCL P2P 为主，配置通过 `additional_config["afd"]` 承载，尽量通过
`--worker-cls`、ModelRegistry 和少量 compat shim 接入。

Ascend/NPU AFD 的行为参考来自 vLLM/vLLM-Ascend `0.13.0` 的组合：AFD contract 在
vLLM 主仓内，NPU connector 和模型/runtime 改动散布在 `vllm_ascend` 的 platform、
worker、ops、patch 和自定义算子体系中。后续迁移时需要把这些行为重新适配到
vLLM `0.19.1` / vLLM-Ascend `0.19.1rc1`，而不是让插件支持 `0.13.0` 运行时。它
`camp2pconnector` 的通信 backend 差异已经收敛在 connector 内部，外部 runtime 和
统一 connector contract 不应感知 HCCL/GLOO/torch-npu/_C_ascend 等实现细节。NPU
迁移仍需要处理 A2E/E2A、NPU stream/event、ACL graph、量化、专家选择和权重裁剪等
设备特定语义。其中 event/stream 相关通信多流和权重加载裁剪当前不进入第一版迁移范围。

## 差异清单

| 编号 | 差异面 | 当前 `afd-plugin` | NPU AFD 0.13 参考 | 决策/迁移含义 |
| --- | --- | --- | --- | --- |
| 1 | vLLM 版本基线 | 绑定 `v0.19.1`，compat helper 也按这个版本判断 | 参考实现基于 vLLM/vLLM-Ascend `0.13.0` | **已决策**：插件只支持 vLLM `0.19.1`；NPU 只支持 vLLM-Ascend `0.19.1rc1`；`0.13.0` 仅作行为参考 |
| 2 | 配置入口 | 只使用 `--additional-config '{"afd": ...}'` | 参考实现使用 `--afd-config` 并写入 `vllm_config.afd_config` | **已决策**：只使用 `--additional-config '{"afd": ...}'`；不新增 `--afd-config` |
| 3 | 配置字段 | 支持 `enabled`、`connector`、`role`、`host`、`port`、server 数量和 `extra_config` | 还使用 `compute_gate_on_attention`、`quant_mode`、`multistream_info`、core num 等字段 | **已决策**：第一版采用通用字段 + `extra_config` 全兼容；`compute_gate_on_attention`、`quant_mode != 0`、通信多流等不支持项开启时 fail fast |
| 4 | connector 名称 | 只注册 `p2pconnector` | 当前只看 NPU `camp2pconnector` | **已决策**：当前文档只分析和迁移 `camp2pconnector` |
| 5 | connector 注册机制 | plugin-owned `AFDConnectorFactory` 静态注册，并通过 lazy import 创建 connector class | vLLM 0.13 AFD 支持 `vllm.afd_connectors` entry point；vllm-ascend 使用该 entry point | **已决策**：沿用当前插件注册机制；`camp2pconnector` 迁入插件并由 `AFDConnectorFactory.register_connector(...)` 静态注册，不走 `vllm.afd_connectors` entry point |
| 6 | connector contract | 抽象方法主要是 `send_attn_output` / `recv_ffn_output` / `recv_attn_output` / `send_ffn_output` | `camp2pconnector` 还需要 metadata 构造/更新、DP metadata 收发和状态更新 helper；`compute_moe` / `select_experts` 只属于当前不支持的 gate-on-attention/MoE backend 路径 | 当前统一 connector contract 先不纳入 `compute_moe` / `select_experts` |
| 7 | recv 返回结构 | 多数路径返回 `(hidden_states, metadata)` | `camp2pconnector` 需要 `AFDRecvOutput` 风格对象，承载 topk、dynamic scales、group list、handle、CAMP2P payload | 建议先统一返回对象，再适配 GPU/NPU connector |
| 8 | connector metadata | 当前 `AFDConnectorMetadata` 只保留 layer/stage/seq_lens/recv handles | 0.13 metadata 包含 dtype/device/num_ubatches/connector_data/topk/row_idx/scale/expert token nums 等 | 需要决定通用 metadata 扩字段，还是使用 `connector_data` 承载设备细节 |
| 9 | ForwardContext | 优先写入 `forward_context.additional_kwargs["afd_metadata"]` | `camp2pconnector` 路径依赖 `forward_context.afd_metadata`、connector-specific data、`afd_comm_stream`、`afd_comm_event` | NPU 侧需要 Ascend-aware forward context shim；`afd_comm_stream` / `afd_comm_event` 相关通信多流当前不支持 |
| 10 | Attention worker | `AFDAttentionWorker` 继承 vLLM `GPUWorker` 并注入 `AFDAttentionModelRunner` | vllm-ascend 使用 `NPUWorker` / `NPUModelRunner` | NPU 需要独立 worker/model runner class path，不能直接复用 GPU 类 |
| 11 | FFN worker/model runner | `AFDFFNWorker` 继承 `GPUWorker`，启动 connector-driven FFN loop；`GPUFFNModelRunner` 负责 recv -> per-layer FFN -> send | `NPUWorker` 在 FFN role 创建 `NPUFFNModelRunner`；该类继承 `NPUModelRunner, GPUFFNModelRunner`，并覆盖 graph、recv payload 和 `_ffn_forward` 等核心设备路径 | **差异中等，不是两套完全不同设计**；FFN daemon loop、DP metadata 驱动、per-layer/per-ubatch 执行可抽公共语义，设备基类、graph 和 connector payload 仍需 NPU 适配；FFN 通信多流当前不支持 |
| 12 | graph runtime | 当前是 CUDA graph policy 和 `FULL_DECODE_ONLY` 等路径 | NPU 侧是 ACL graph/NPU graph，散布在 `acl_graph.py`、`NPUModelRunner`、`NPUFFNModelRunner` | Graph 不能直接共用实现，只能共用 metadata/control-plane 语义 |
| 13 | ubatching/DBO | 当前有 plugin-owned `AFDUBatchWrapper` 和 ubatch DP metadata 构造 | NPU 侧新增 `npu_ubatch_wrapper.py`，并向 `AFDMetadata` 填充 input/position/attn/dp list | 当前 `AFDMetadata` 缺少 0.13 需要的 list 字段 |
| 14 | topology/rank mapping | P2P world 固定 FFN ranks 在前，Attention ranks 在后；要求 A >= F 且整除 | `camp2pconnector` 使用 FFN ranks 在前、Attention ranks 在后，支持非等 A/F 路由 | 当前只抽取 `camp2pconnector` 所需 topology |
| 15 | 通信 backend | GPU P2P 使用 NCCL/PyNccl/StatelessProcessGroup | `camp2pconnector` 内部使用 HCCL/GLOO process group、`torch.distributed`、`torch_npu`、`_C_ascend` 和 CAM expert select op | **已决策**：backend 差异收敛在 connector 内部，外部 runtime 和统一 contract 不感知具体通信 backend |
| 16 | 自定义算子 | 当前插件纯 Python，CPU-safe import | NPU 分支新增 `csrc/a2e`、`csrc/e2a`，依赖 CANN、torch-npu、`umdk_cam_op_lib`、`cam_ge_operator` | **已决策**：`afd-plugin` 负责构建 Ascend extension/custom op |
| 17 | 模型层 | 当前 DeepSeek wrapper 偏 GPU e2e/smoke，未完整做 role weight pruning | NPU patch DeepSeek V2/V3、MTP、fused MoE、experts selector、token dispatcher | 模型层是最大迁移面，需要决定统一 wrapper 还是 GPU/NPU 分别维护 |
| 18 | `compute_gate_on_attention` | 当前插件配置和模型 wrapper 基本未完整覆盖 | NPU 侧大量依赖该字段，影响 gate 创建、权重加载、topk 计算位置和 connector payload | **已决策**：当前不迁移，暂不支持；配置开启时 fail fast |
| 19 | 权重加载裁剪 | 当前注释明确双方加载完整 DeepSeekV2 权重 | NPU patch 按 Attention/FFN role 跳过 MoE/gate 权重，并做 gate 权重重映射 | **已决策**：当前暂不支持权重加载裁剪，第一版保持完整权重加载；后续作为独立能力支持 |
| 20 | patch 策略 | 插件有少量 `afd_plugin.compat.patches`，范围较窄 | vllm-ascend 本身是 platform/worker patch 体系，AFD 也散在其中 | **已决策**：如必须 patch vLLM-Ascend，AFD-specific patch 统一放入 `afd_plugin.compat.ascend`，并保持幂等和版本保护 |
| 21 | CLI/启动方式 | 当前设计优先 `vllm serve` + `--worker-cls`，FFN 建议 `--headless`，不保留 `fserver` | 0.13 GPU 有 `vllm fserver`；NPU 侧参考代码仍假设 `vllm_config.afd_config` 已存在 | **已决策**：NPU 和当前插件保持一致，使用 `vllm serve` + 显式 class path / `--worker-cls`；不使用 `vllm fserver` |
| 22 | 测试矩阵 | 当前有 CPU/unit 和 GPU-gated DeepSeek Lite e2e | NPU 侧有 A2E/E2A 多卡 op 测试，但端到端依赖 Ascend 环境 | 需要新增 Ascend import skip、NPU op、connector multi-process、NPU e2e 分层 |
| 23 | AFD 通信多流 | 当前 GPU 路径已有 CUDA stream/ubatch wrapper 相关实现 | NPU 参考实现包含 Attention/FFN 通信多流、NPU stream/event 和 per-ubatch event | **已决策**：当前不支持 Attention 通信多流和 FFN 通信多流；相关配置开启时 fail fast |

## 配置差异展开

### 当前插件配置

当前插件的 canonical 形态保留通用字段，同时通过 `extra_config` 承载 NPU 旧字段和
connector-specific 字段。第一版采用这个宽松 schema，后续在 NPU 路径打通后再考虑
收紧。

```json
{
  "afd": {
    "enabled": true,
    "connector": "camp2pconnector",
    "role": "attention",
    "host": "127.0.0.1",
    "port": 1239,
    "num_afd_stages": 2,
    "num_attention_servers": 2,
    "num_ffn_servers": 2,
    "extra_config": {
      "afd_size": "2A2F",
      "compute_gate_on_attention": false,
      "multistream_info": {
        "attn_enable": "False",
        "ffn_enable": "False"
      }
    }
  }
}
```

并保留部分原始字段 alias，例如 `afd_connector -> connector`、
`afd_role -> role`、`afd_host -> host`。

配置入口已经决策为只使用 `--additional-config`。NPU 迁移过程中可以在
`additional_config["afd"]` 内兼容原始字段名，但不新增独立 `--afd-config` CLI
参数，也不要求把配置写回 vLLM 原生 `VllmConfig.afd_config` 字段。

第一版不引入 `connector_config` 之类的更细粒度 namespace。`extra_config` 中允许
保留原始 NPU 字段，plugin validation 只对已明确不支持的功能做 fail fast，例如
`compute_gate_on_attention=true`、`quant_mode != 0` 和通信多流开启。

### NPU 侧额外字段

NPU connector 和模型 patch 还读取：

- `compute_gate_on_attention`
- `quant_mode`
- `multistream_info`
- `is_attn_multistream`
- `is_ffn_multistream`
- `attn_core_num`
- `ffn_core_num`
- 部分旧代码中的 `is_multistream` 和 `multistream_info["core_num"]`

`compute_gate_on_attention` 已决策为当前不支持：配置解析可以识别该字段，但若用户
开启，应在 validation 或 runtime 初始化时 fail fast。

`quant_mode` 已决策为当前不移植：`camp2pconnector` 参考实现中 `quant_mode` 只是
metadata 占位并写死为 `0`，真正按该字段切换 int8/dynamic quant 的路径属于当前不看的
connector。配置解析可以识别该字段，但若用户显式设置为非 `0`，应 fail fast；该字段
不参与 graph/hash。

通信多流也已决策为当前不支持：`is_attn_multistream`、`is_ffn_multistream`、
`is_multistream` 或 `multistream_info` 中的 enable 字段一旦开启，应 fail fast。
`attn_core_num`、`ffn_core_num` 和 `core_num` 只在通信多流开启时有意义，第一版可
作为已知但未使用的字段处理。

其余字段有些影响计算图，有些只影响通信/性能。后续需要区分：

- 是否参与 hash/graph key；
- 是否属于通用 AFD 语义；
- 是否应该从宽松 `extra_config` 迁出到更严格的 connector-specific namespace；
- 是否需要兼容旧字段名。

## connector 差异展开

### GPU P2P connector

当前插件的 `p2pconnector` 主要承担：

- 建立 AFD process group；
- 建立 Attention <-> FFN subgroup；
- 发送/接收 DP metadata list；
- 发送 Attention hidden states；
- 接收 FFN output；
- 支持 CUDA graph 相关 control flags 和 receive buffer 预分配。

### NPU connector 范围

当前 NPU 迁移范围已经收敛为只分析和迁移 `camp2pconnector`：

- `camp2pconnector`：使用 `_C_ascend.a2e/e2a`，支持 CAM 风格的 expert
  select、A2E/E2A、DP metadata list、非等 A/F 路由。

`camp2pconnector` 中涉及的 multistream/event/stream 行为也不进入当前支持范围。

当前插件需要补齐的统一接口包括：

- `configure_metadata(metadata, **kwargs)`
- `create_recv_metadata(**kwargs)`
- `update_metadata(metadata, recv_output)`
- `send_dp_metadata_list(...)`
- `recv_dp_metadata_list(...)`
- `update_state_from_dp_metadata(...)`

`compute_moe(...)` 和 `select_experts(...)` 在 NPU 参考代码中存在，但当前只迁移
`camp2pconnector` 且 `compute_gate_on_attention` 已决策暂不支持。因此这两个方法先
不进入统一 connector contract；后续如果重新打开 gate-on-attention 或把 Ascend MoE
backend 抽到 connector capability，再单独设计。

## runtime 差异展开

### Attention 侧

当前插件的 Attention 路径是 GPU-specific：

```text
AFDAttentionWorker -> AFDAttentionModelRunner -> AFDConnectorFactory
```

NPU 侧需要适配：

```text
NPUWorker -> NPUModelRunner -> set_ascend_forward_context -> NPU connector
```

主要差异是设备 API、forward context、ACL graph dispatch、NPU stream/event 和
vllm-ascend 的既有 patch 生命周期。当前只迁移单流通信路径，不支持依赖
`afd_comm_stream` / `afd_comm_event` 的 Attention 通信多流。

### FFN 侧

当前插件 FFN 侧已经采用 connector-driven loop，这和 NPU 设计方向接近。对照
`../vllm` 的 `GPUFFNModelRunner` 和 `../vllm-ascend` 的 `NPUFFNModelRunner` 后，
结论是：两者差异没有 Attention 路径那么大，核心 FFN 控制流可以归一，但不能把
CUDA 版 runner 原样搬给 NPU。

相同或可归一的部分：

- FFN 侧都是 daemon/loop 语义，不由普通 scheduler output 驱动；
- 都围绕 `dp_metadata_list` 或 connector 内部 DP metadata 决定 stage/ubatch；
- 都是 recv Attention hidden states -> 设置 forward context -> 调
  `model.compute_ffn_output(...)` -> send FFN output；
- graph key 本质上都可以从 `dp_metadata_list` 派生；
- `profile_run`、空 KV cache、LoRA/sample/tensorize 等兼容接口基本一致。

主要差异集中在设备和 payload 适配：

- `NPUFFNModelRunner` 继承 `NPUModelRunner`，依赖 vllm-ascend 初始化出的
  `parallel_config`、`scheduler_config`、`max_num_tokens`、`uniform_decode_query_len`、
  `cudagraph_dispatcher`、`use_aclgraph` 等状态；这部分不应复制 CUDA runner 的
  `__init__`；
- ACL graph/NPU graph cache，而不是 CUDA graph cache；
- 根据 Attention 侧发送的 DP metadata 做 warmup/capture/normal 执行；
- connector `recv_attn_output` 返回 `AFDRecvOutput` 风格对象，额外携带
  `dynamic_scales`、`group_list`、`topk_weights`、`topk_ids`、`router_logits`、
  `row_idx`、`x_active_mask`、`cam_p2p_ep_name` 等 MoE/量化/CAM payload；
- `_run_ffn_computation` 的签名比 GPU 版宽，需要把上述 payload 透传给
  Ascend MoE backend；
- FFN multistream stream/event；该能力当前暂不迁移；
- profiler 环境变量控制。

因此更合理的迁移判断是：可以抽一个设备无关的 FFN runner contract/helper，统一
`dp_metadata_list` graph key、空 KV cache、compat method、daemon loop 调度和
recv/compute/send 的控制流；NPU 侧保留基于 `NPUModelRunner` 的 subclass，覆盖
设备初始化、ACL graph、Ascend forward context 和 `AFDRecvOutput` payload 处理。

## 模型层差异展开

当前插件 DeepSeek wrapper 主要完成 Attention/FFN forward 切分。权重加载裁剪已经
决策为当前暂不支持，因此第一版 Attention/FFN 双方保持完整模型权重加载。

NPU 侧模型 patch 更完整，至少涉及：

- `DeepseekV2MoE` / `DeepseekV3` 的 AFD forward；
- `compute_gate_on_attention` 下 gate 权重位置和 topk 计算位置；该能力当前暂不迁移；
- FFN role 下跳过 Attention-only 逻辑；
- Attention role 下跳过 MoE 权重；该能力当前暂不支持，后续单独支持；
- FFN role 下跳过 gate 权重；该能力当前暂不支持，后续单独支持；
- shared experts / mix placement；
- Ascend fused MoE、token dispatcher、experts selector；
- MTP/spec decode 与 AFD role 的互斥或降级行为。

这里需要先确定模型归属：

- 统一由 `afd_plugin.model_executor.models` 提供跨设备 wrapper；
- 或者 GPU wrapper 留在插件，NPU wrapper 依赖/复用 `vllm_ascend.patch.worker`
  中已有实现；
- 或者拆成通用 AFD model mixin + device-specific MoE backend。

当前 NPU 迁移不支持 `compute_gate_on_attention`，因此第一版模型/MoE 迁移应避开
gate-on-attention 路径，并在配置开启时给出明确错误。

## patch 与依赖边界

当前插件的约束是“不修改 vLLM 源码树”，并将必要 patch 放在
`afd_plugin.compat.patches`。NPU 侧现有实现依赖 vllm-ascend 的 platform/worker patch
体系，而 vllm-ascend 本身就是 vLLM platform plugin。

后续还需要明确的边界：

1. `afd-plugin` 是否直接依赖 `vllm_ascend` 作为 NPU backend provider。
2. `camp2pconnector` 已决策迁入 `afd_plugin.connectors.ascend` 并沿用插件内
   `AFDConnectorFactory` 静态注册；不通过 `vllm_ascend.distributed` 桥接注册。
3. 如果确实存在必须 patch vLLM-Ascend 的 AFD-specific 行为，统一收敛到
   `afd_plugin.compat.ascend`。这些 patch 不能散落在 runtime、connector 或 model
   模块里，必须幂等、版本受保护，并有测试或文档说明。

## 初步阶段建议

这不是最终路线，只是为了后续讨论提供分层：

1. **NPU Phase 0：兼容性盘点**
   明确 `0.13.0` NPU AFD 参考实现和目标 vLLM `0.19.1` / vLLM-Ascend
   `0.19.1rc1` 的接口差异，补充 connector/mode/模型差异矩阵。

2. **NPU Phase 1：配置与注册归一**
   扩展 `AFDConfig`，允许 `camp2pconnector` 和它需要的 NPU-specific 字段通过
   `extra_config` 宽松传入；将 `camp2pconnector` 迁入插件，并沿用
   `AFDConnectorFactory.register_connector(...)` 做插件内静态注册。

3. **NPU Phase 2：NPU Attention/FFN runtime 骨架**
   新增 NPU worker/model runner class path，复用 vllm-ascend `v0.19.1rc1` 的
   worker 生命周期。FFN runner 可以先抽取/复用 GPU 侧的设备无关 contract 与兼容
   helper，但 NPU 实现仍以 `NPUModelRunner` 生命周期为基类。此阶段只要求
   import/classpath、配置解析、role validation 和生命周期入口能跑通，不要求完整
   connector contract 和真实通信闭环。

4. **NPU Phase 3：connector contract 归一**
   引入 `AFDRecvOutput`、`connector_data`、metadata 构造/更新和 DP metadata 收发等
   接口，先让 GPU connector 适配新 contract，再让 NPU runtime 骨架接入统一
   contract。`compute_moe` / `select_experts` 当前不进入统一 contract。

5. **NPU Phase 4：Ascend connector 接入**
   以 CPU-safe import 为前提，接入 `camp2pconnector` 的最小 wrapper，缺少
   torch-npu 或 custom op 时干净 skip。

6. **NPU Phase 5：模型/MoE 语义**
   迁移不依赖 `compute_gate_on_attention` 的 Ascend fused MoE 和 token dispatcher
   相关逻辑；`compute_gate_on_attention` 和权重加载裁剪当前明确不支持，后续单独
   设计。

7. **NPU Phase 6：ACL graph、ubatching 和性能**
   在 connector 和模型语义稳定后，再处理 ACL graph、NPU e2e 和性能 profiling。
   Attention/FFN 通信多流当前明确不支持，若后续重新开启，应单独设计。

## 待决策问题

- `AFDRecvOutput` 是否成为所有 connector 的统一返回值？
- `ForwardContext.afd_metadata` 是否需要重新 patch，还是继续统一走
  `additional_kwargs["afd_metadata"]`？
- DeepSeek/Step3/NPU MoE 逻辑由插件统一维护，还是迁入
  `afd_plugin.compat.ascend` 中受控 patch？
- NPU e2e 测试使用远程 L20X 机器，还是需要新增 Ascend 远程验证流程？
