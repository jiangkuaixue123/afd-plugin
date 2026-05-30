# NPU AFD UBatch 设计文档

本文档记录在 `afd-plugin` 中为 vLLM-Ascend `v0.19.1rc1` 支持 NPU
Attention-FFN Disaggregation ubatch 的设计。目标是把原本散落在
`jcz_afd_v13_dev2` / 0.13 AFD 分支里的 NPU ubatch 行为，迁移为
out-of-tree plugin 代码，不修改 `../vllm` 或 `../vllm-ascend`。

## 背景结论

本轮对三个参考点的结论如下：

- vLLM-Ascend `origin/releases/v0.13.0` 基本没有真实 NPU ubatch 数据流。
  该分支只保留 DBO 配置校验、未使用的 `allow_microbatching` 参数，以及
  workspace 固定 `num_ubatches = 1`。
- `jcz_afd_v13_dev2` 是当前最有价值的 NPU AFD ubatch 行为参考。它在
  Attention 侧切分输入和 `AttentionMetadata`，在 FFN 侧通过
  `dp_metadata_list` 同步每个 ubatch 的 stage/token 形状，并按
  `ubatch_idx` 收发。
- vLLM-Ascend `v0.19.1rc1` 已有部分上游 ubatch 接口形状，例如
  `maybe_create_ubatch_slices`、`UBatchSlices` 和 runner 参数，但 Ascend
  当前显式禁用 DBO/ubatch，runner 实际不会进入 ubatch 路径，也没有 AFD
  metadata 或 NPU ubatch wrapper。

因此，`v0.19.1rc1` 不是从零开始，但缺少完整执行闭环。可复用的是上游
ubatch slice API 和 0.13 AFD 分支的算法思路；必须在插件内重写的是 NPU
runner glue、metadata split、NPU ubatch wrapper、AFD metadata / DP metadata
多 ubatch 协议和 graph 策略。

## 目标和非目标

### 第一阶段目标

- 只支持 vLLM `v0.19.1` + vLLM-Ascend `v0.19.1rc1`。
- 只支持 vLLM-Ascend model runner v1。
- 通过 `--additional-config '{"afd": ...}'` 启用 AFD NPU ubatch，不依赖
  vLLM-Ascend 原生 `--enable-dbo` 是否被平台层重置。
- 支持 eager 路径下的 NPU AFD ubatch 闭环：
  Attention runner 生成 ubatch slices 和 per-ubatch metadata，Attention/FFN
  connector 按 `ubatch_idx` 传输，FFN runner 按 ubatch 执行并返回。
- 保持 plugin import CPU-safe。`vllm_ascend`、`torch_npu` 和 NPU custom op
  只在显式使用 NPU class path 或 NPU connector 时导入。

### 第一阶段非目标

- 不支持 ACL graph / NPU graph capture 下的 ubatch。
- 不支持 Attention/FFN 通信多流。
- 不支持 `compute_gate_on_attention=true`。
- 不支持 `quant_mode != 0`。
- 不支持按 Attention/FFN role 裁剪权重加载。
- 不支持 vLLM-Ascend model runner v2。

这些能力如果用户显式开启，应在 validation 或 runner 初始化阶段 fail fast。

## 总体数据流

```text
SchedulerOutput
  -> AFDNPUAttentionModelRunner._determine_batch_execution_and_padding()
       使用 coordinate_batch_across_dp 得到 should_ubatch / DP padding
  -> maybe_create_ubatch_slices(...)
       得到 ubatch_slices / ubatch_slices_padded
  -> _build_attention_metadata(..., ubatch_slices=...)
       全 batch AscendCommonAttentionMetadata
       -> split_ascend_attention_metadata_for_ubatches(...)
       -> list[dict[layer_name, per_ubatch_attention_metadata]]
  -> build parent AFDMetadata + dp_metadata_list
  -> install AFD metadata on forward context
  -> AFDNPUUBatchWrapper
       为每个 ubatch 创建 child forward context
       child context 持有 attn_metadata[i] / ubatch_idx / num_ubatches
       child context 持有 ubatch DP metadata / AFD metadata
  -> model forward per ubatch
  -> concat outputs
  -> connector sends per-layer/per-ubatch Attention output to FFN
  -> FFN runner receives dp_metadata_list and loops ubatch_idx
```

## 核心设计点

### 1. 配置入口和 platform reset

vLLM-Ascend `v0.19.1rc1` 的 `platform.py` 会把原生 `enable_dbo` 重置为
`False`，并把 `ubatch_size` 重置为 `0`。因此第一阶段不应依赖原生
`--enable-dbo` CLI 作为 AFD NPU ubatch 的唯一入口。

建议在 `additional_config["afd"]` 中承载 AFD NPU ubatch 开关，例如：

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "camp2pconnector",
    "extra_config": {
      "npu_ubatching": true,
      "num_ubatches": 2
    }
  }
}
```

迁移早期可以兼容已有字段名，但 canonical 入口仍应是
`additional_config["afd"]`。如果后续确认需要复用 vLLM 原生
`parallel_config.use_ubatching` / `num_ubatches`，也应由 plugin validation
统一写入或校验，不应依赖 vLLM-Ascend 平台层默认保留该配置。

### 2. Workspace 初始化

当前 NPU worker 的 workspace 初始化仍按 `num_ubatches = 1`。开启 AFD NPU
ubatch 后，Attention 和 FFN 两侧 worker 都应使用配置中的 `num_ubatches`
初始化 Ascend workspace。

设计要求：

- `AFDNPUAttentionWorker.init_device()` 和 `AFDNPUFFNWorker.init_device()`
  读取同一份 AFD ubatch 配置。
- `init_ascend_workspace_for_afd(device, num_ubatches=...)` 不应固定为 `1`。
- 如果底层 vLLM-Ascend workspace manager 暂不支持多 ubatch，则在 runner
  初始化阶段 fail fast，而不是运行到 forward 才报低层错误。

### 3. `_determine_batch_execution_and_padding`

NPU runner 的 `_determine_batch_execution_and_padding()` 需要和 GPU v0.19.1
逻辑对齐。关键区别是当前 vLLM-Ascend `v0.19.1rc1` 仍通过
`_sync_metadata_across_dp(...)` 做 DP padding，并固定返回
`should_ubatch = False`。

目标逻辑：

- 复用 Ascend 现有 cudagraph / batch descriptor / sequence parallel padding
  行为。
- 当 AFD NPU ubatch 开启时，使用
  `vllm.v1.worker.dp_utils.coordinate_batch_across_dp(...)` 协调 DP ranks。
- `coordinate_batch_across_dp(...)` 的输入应包含：
  `num_tokens_unpadded`、`allow_microbatching`、`parallel_config`、
  `num_tokens_padded`、`uniform_decode`、`num_scheduled_tokens_per_request` 和
  当前 graph mode。
- 从返回值中取得 `should_ubatch`、`num_tokens_across_dp` 和同步后的 graph
  mode。
- 如果 `num_tokens_across_dp` 非空，仍按当前 Ascend runner 逻辑重新 dispatch
  batch descriptor，确保 DP padding 后的 token 数一致。

第一阶段 graph 不支持时，`force_eager` 或 AFD ubatch validation 应保证 graph mode
为 `NONE`。但 `_determine_batch_execution_and_padding()` 的 shape 仍应保留
graph 参数，避免后续支持 ACL graph 时再大改函数签名。

### 4. `execute_model` 和 `_dummy_run`

`execute_model()` 和 `_dummy_run()` 都需要真正计算 ubatch slices。

`execute_model()` 要求：

- `_determine_batch_execution_and_padding()` 返回真实 `should_ubatch`。
- 调用 `maybe_create_ubatch_slices(...)` 得到
  `ubatch_slices` 和 `ubatch_slices_padded`。
- eager 路径使用 `ubatch_slices`。
- 后续 graph 支持时，full graph 的 attention metadata 可以使用
  `ubatch_slices_padded`，但第一阶段应直接禁用 graph。
- `_build_attention_metadata(..., ubatch_slices=...)` 必须根据 slices 返回
  per-ubatch metadata list。
- 当前 batch 的 `dp_metadata_list` 应在 Attention side 发送给 FFN side。

`_dummy_run()` 要求：

- profile / warmup / dummy run 的 token shape 也要通过相同的 ubatch slice
  逻辑。
- dummy run 生成的 `dp_metadata_list` 应带有 `is_warmup` /
  `is_graph_capturing` 状态，供 FFN 侧决定 warmup、capture 或 normal execute。
- 第一阶段如果 graph capture 未支持，应在 dummy run graph capture 分支 fail
  fast，避免把 connector 通信副作用 capture 进 NPU graph。

### 5. AttentionMetadata split

需要新增 plugin-owned 的 metadata split helper，按 vLLM-Ascend
`v0.19.1rc1` 的 `AscendCommonAttentionMetadata` 字段逐项构造
per-ubatch metadata。

参考 `jcz_afd_v13_dev2` 的算法，但不能照搬。0.19.1rc1 需要重点覆盖：

- `query_start_loc`
- `query_start_loc_cpu`
- `seq_lens`
- `seq_lens_cpu`
- `num_computed_tokens_cpu`
- `num_reqs`
- `num_actual_tokens`
- `num_input_tokens`
- `max_query_len`
- `max_seq_len`
- `block_table_tensor`
- `slot_mapping`
- `positions`
- `decode_token_per_req`
- `actual_seq_lengths_q`
- `attn_state`
- `graph_pad_size`
- `causal`
- `prefill_context_parallel_metadata`
- `kvcomp_metadata`

切分时需要处理跨 ubatch 的 request：

- 如果 token slice 从 request 中间开始，需要平移
  `query_start_loc` / `query_start_loc_cpu`。
- 如果 token slice 在 request 中间结束，需要收缩最后一个 request 的
  query len 和 `seq_lens`。
- `block_table_tensor` 按 request slice 切。
- `slot_mapping` 和 `positions` 按 token slice 切。
- `actual_seq_lengths_q` 不能沿用 0.13 的简化构造，必须按 0.19.1rc1 当前
  attention backend 对该字段的语义重新验证。

第一阶段建议对复杂场景先 fail fast：

- PCP / DCP / context parallel active；
- spec decode active；
- GDN 或其他尚未验证 metadata builder；
- encoder-decoder / cross attention；
- graph padded metadata。

这些场景可以在单测覆盖后逐步放开。

### 6. AFD metadata 和 DP metadata

AFD NPU ubatch 需要两个层次的 metadata：

1. parent batch metadata：描述本次 AFD transaction、总 stage 数和每个 stage 的
   token/request 范围。
2. child ubatch metadata：每个 ubatch forward context 中可直接读取的 stage
   metadata。

`AFDMetadata` 至少需要表达：

- `afd_tokens_start_loc`
- `afd_reqs_start_loc`
- `afd_tokens_lens`
- `afd_tokens_unpadded_lens`
- `afd_stage_idx`
- `num_of_stages`
- `ubatch_idx`
- `transaction_id`
- `afd_connector`

`dp_metadata_list` 应按 `ubatch_idx` 建立：

```text
dp_metadata_list = {
  0: DPMetadata for ubatch 0,
  1: DPMetadata for ubatch 1,
  ...
}
```

DP size 为 1 时，可以使用 plugin-owned `AFDDPMetadata` fallback。DP size 大于
1 时，应优先使用 vLLM 原生 DP metadata 结构或 `coordinate_batch_across_dp(...)`
同步后的 token 信息。代码中直接访问 vLLM/vLLM-Ascend 的字段，不使用
`getattr` / `hasattr` 静默兼容。

### 7. ForwardContext 注入策略

插件内 canonical 存储仍优先使用：

```text
forward_context.additional_kwargs["afd_metadata"]
```

但 NPU 参考实现和部分 Ascend model/connector hook 会读取：

```text
forward_context.afd_metadata
forward_context.ubatch_idx
forward_context.num_ubatches
```

设计要求：

- 统一由 `afd_plugin.compat.ascend` 提供受控 mirror helper。
- runner 和 connector 不应散落 `setattr(forward_context, ...)`。
- NPU ubatch wrapper 创建 child context 时必须设置：
  `attn_metadata[i]`、child `dp_metadata`、`ubatch_idx`、`num_ubatches`、
  `num_tokens`、`batch_descriptor` 和 AFD metadata。
- 第一阶段不设置或不依赖 `afd_comm_stream` / `afd_comm_event` 的通信多流语义。
  如果底层为了 API 兼容必须存在字段，应填入 no-op 或受控单流对象，并明确不支持
  multistream。

### 8. NPU UBatchWrapper

需要新增 plugin-owned NPU ubatch wrapper，例如：

```text
afd_plugin.v1.worker.ascend.ubatch_wrapper.AFDNPUUBatchWrapper
```

职责：

- 接收 parent forward context 中的 `ubatch_slices`、per-ubatch
  `attn_metadata`、parent AFD metadata 和 inputs。
- 为每个 ubatch 切分 `input_ids`、`positions`、`inputs_embeds`、
  `intermediate_tensors`。
- 为每个 ubatch 创建 child forward context。
- child context 中安装 `attn_metadata[i]`、`dp_metadata_list[i]`、
  `ubatch_idx`、`num_ubatches` 和 AFD metadata。
- eager 路径下逐 ubatch 执行或按 vLLM ubatch context 线程模型执行。
- 按 ubatch 顺序 `torch.cat` 输出，不尝试把 `AttentionMetadata` 合回 parent。

第一阶段不迁移 0.13 分支中的 NPU graph replay、全局 graph params dict、
`time.sleep(...)` 等不稳定逻辑。graph 后续单独设计。

### 9. FFN 侧协议

Attention 侧发送 `dp_metadata_list` 后，FFN 侧必须以相同 `ubatch_idx` 为协议
边界。

FFN runner 目标循环：

```text
for ubatch_idx in range(num_ubatches):
  hidden_states, metadata = connector.recv_attn_output(ubatch_idx=ubatch_idx)
  output = ffn_forward(hidden_states, metadata, ubatch_idx=ubatch_idx)
  connector.send_ffn_output(output, metadata, ubatch_idx=ubatch_idx)
```

要求：

- connector 的 HCCL/GLOO group、handle、payload 和 recv/send queue 都不能跨
  `ubatch_idx` 串用。
- `camp2pconnector` 的 metadata 应能从 `dp_metadata_list[ubatch_idx]` 推导该
  ubatch 的 max token shape。
- FFN worker 的 daemon loop 通过
  `recv_dp_metadata_list()` 接收 `is_warmup` / `is_graph_capturing`，并按状态
  决定 warmup、capture 或 normal execute。
- 第一阶段 graph disabled 时，`is_graph_capturing=true` 应 fail fast 或被明确
  拒绝。

## 与现有插件代码的关系

当前插件已有以下基础：

- `AFDNPUAttentionWorker`
- `AFDNPUAttentionModelRunner`
- `AFDNPUFFNWorker`
- `AFDNPUFFNModelRunner`
- `npudummyconnector`
- `camp2pconnector` placeholder / 部分 Ascend op 迁移
- GPU 侧 `AFDUBatchWrapper` 和 ubatch metadata helper

NPU ubatch 不应直接继承 GPU `AFDUBatchWrapper`，因为 GPU wrapper 依赖 CUDA
stream、CUDA graph 和 GPU forward context 假设。可以复用的只是设备无关 helper
思路，例如：

- clone parent `AFDMetadata` 生成 child metadata；
- `build_ubatch_dp_metadata_list(...)` 的 DP=1 fallback 思路；
- per-ubatch `additional_kwargs["afd_metadata"]` 注入方式。

长期建议抽出设备无关 coordinator，但第一阶段可以在 NPU runner / wrapper 内保留
少量重复 glue，用于先打通功能闭环。

## 实现阶段

### Phase U1：配置和 validation

- 增加 AFD NPU ubatch 配置解析。
- 校验 vLLM `v0.19.1`、vLLM-Ascend `v0.19.1rc1`、MRv1。
- 开启 ubatch 时禁用 graph、multistream、`compute_gate_on_attention`、
  `quant_mode != 0`、MRv2。
- worker workspace 初始化读取 `num_ubatches`。

### Phase U2：runner 生成 ubatch slices

- 在 `AFDNPUAttentionModelRunner` 中实现 GPU-like
  `_determine_batch_execution_and_padding()`。
- 使用 `coordinate_batch_across_dp(...)` 生成真实 `should_ubatch`。
- 在 `execute_model()` 和 `_dummy_run()` 中调用
  `maybe_create_ubatch_slices(...)`。
- 为 DP=1 和 DP>1 都生成可发送给 FFN 的 `dp_metadata_list`。

### Phase U3：AttentionMetadata split

- 新增 `split_ascend_attention_metadata_for_ubatches(...)`。
- 按 0.19.1rc1 字段逐项构造 child `AscendCommonAttentionMetadata`。
- 单测覆盖 decode-only、prefill-only、跨 request 边界切分、DP=1 fallback。
- 对 PCP/spec/GDN/encoder-decoder 等未验证场景 fail fast。

### Phase U4：AFDNPUUBatchWrapper

- 新增 NPU-specific ubatch wrapper。
- wrapper 为每个 ubatch 构造 child forward context。
- eager 路径跑通 per-ubatch model forward 和输出 concat。
- 不支持 graph capture。

### Phase U5：FFN 和 connector 闭环

- FFN worker daemon loop 接收 `dp_metadata_list`。
- FFN runner 按 `ubatch_idx` 循环收发。
- `npudummyconnector` 先跑通 metadata/control plane。
- `camp2pconnector` 再接入真实 HCCL/GLOO/custom op payload。

### Phase U6：NPU 环境验证

- CPU unit：配置、metadata split、AFD metadata clone、DP metadata list。
- NPU smoke：`npudummyconnector` Attention/FFN 双进程。
- NPU integration：`camp2pconnector` 多进程，验证 per-ubatch 收发顺序。
- 远程 L20X 验证后删除临时测试分支。

## 风险和待确认点

- 0.19.1rc1 的 `AscendCommonAttentionMetadata` 字段比 0.13 更多，
  `split_attn_metadata` 不能机械照搬。
- `actual_seq_lengths_q`、PCP/DCP、spec decode、GDN metadata builder 的 ubatch
  语义需要逐项验证。
- DP>1 时 `coordinate_batch_across_dp(...)` 和 Ascend 现有
  `_sync_metadata_across_dp(...)` 的职责边界需要在代码中收敛，避免双重同步。
- FFN side 的 graph/warmup/capture 状态来自 Attention side 的
  `dp_metadata_list`，第一阶段 graph disabled 时必须显式拒绝 capture 状态。
- `camp2pconnector` 多 ubatch 下的 group endpoint、handle 生命周期和 payload
  shape 必须按 `ubatch_idx` 隔离。
- 如果后续支持 ACL graph，需要单独设计 per-ubatch graph params、capture 更新和
  replay 策略，不应混入第一阶段 eager 实现。

## 当前结论

NPU AFD ubatch 的主线工作项是：

1. `_determine_batch_execution_and_padding()` 使用和 GPU 类似的
   `coordinate_batch_across_dp(...)` 逻辑生成真实 `should_ubatch`。
2. `_dummy_run()` 和 `execute_model()` 都真正计算 `ubatch_slices`。
3. 增加 plugin-owned NPU `UBatchWrapper`。
4. 适配 `AttentionMetadata` split。
5. 补齐 platform reset 绕开策略、workspace `num_ubatches` 初始化、per-ubatch
   AFD/DP metadata、forward context 注入、FFN per-ubatch 协议和 graph fail-fast
   边界。

