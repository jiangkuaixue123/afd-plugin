# Async CAM 源码算子适配计划

## 1. 目标与基线

- 计划基线：`afb0b83`，已合入 PR #361。
- 当前状态：源码算子已接入构建，但 `CAMAsyncAFDConnector` 仍调用
  `afd_plugin/connectors/npu/bin` 中旧 wheel 提供的算子。
- 目标：将异步通信切换到插件源码构建的四个
  `torch.ops.afd_ascend.afd_async_*` 算子，并通过 DSV2-Lite 和
  DSV4-Flash 的 E2E 验收。
- 本文是实施计划，不代表适配、设备编译或 E2E 已完成。

总体顺序：源码算子验证 → connector 协议及加载逻辑迁移 → 共享专家迁到
Attention → 双模型 E2E → 依赖与文档收尾。

迁移时必须同步移除原有二进制 `.so` 相关加载逻辑，不能只替换算子调用，
也不能将旧加载逻辑的移除推迟到 E2E 通过之后。

旧侧分析依据为当前调用链和 wheel 内注册的 schema，未对旧二进制内核进行
逐行比较。PR #361 的源码算子仍需在目标环境完成 CANN 编译及 NPU 验证。

## 2. 范围与执行契约

本轮目标为 910C、eager、现有 Async CAM 和现有两阶段 MoE ubatching。
同步 CAMP2P、GPU 及其他既有模型路径保持兼容。
ACL graph 和原生 DBO 不作为本轮异步路径新增能力。

| 部分 | Attention 侧 | FFN 侧 |
| --- | --- | --- |
| 路由 | 计算 routed expert IDs、weights | 使用 dispatch 返回的专家分组 |
| Routed experts | 发送输入、接收加权结果 | 计算本地 routed experts |
| Shared experts | 加载权重、计算、完成必要的 TP 通信 | 不再构造或计算 |
| 最终 MoE 输出 | 按模型原生缩放规则合并 routed 与 shared | 只返回 routed output |
| 跨 A/F 通信 | 使用 `afd_ascend.afd_async_*` | 使用相同 compact 协议 |

数学语义保持为：

```text
MoE(x) = Shared(x) + Routed(x)
```

Routed 权重及 scaling factor 必须按各模型原生实现应用，避免重复缩放。
所有参与 rank 必须一起切换协议，不允许新旧算子混用或自动回退旧算子。

## 3. 已知接口差异

| 阶段 | 当前二进制接口 | #361 源码接口 | 适配影响 |
| --- | --- | --- | --- |
| A dispatch send | `umdk_cam_op_lib.async_dispatch_send` | `afd_ascend.afd_async_dispatch_send` | 参数排列基本一致，新协议只发送 routed token |
| F dispatch recv | 返回 7 个 tensor，包含 shared 输入、scale、token 数 | 返回 4 个 tensor：routed 输入、scale、batch metadata、专家 token 数 | 修改解包、transfer state 和 FFN work item |
| F combine send | 接收 routed output 和 `expand_x_shared` | 删除 shared output 参数 | 修改 FFN 输出组织和发送接口 |
| A combine recv | 当前模型依赖远端专家合并结果 | 只返回 routed experts 加权结果 | 模型侧另行合并 shared output |

新协议没有共享专家预留槽，routed expert 的本地索引从 0 开始。
令 `R` 为每个 FFN rank 的 routed expert 数，新 metadata 为：

```text
batch_info: int64[5 + TP + R * TP]

前五项：
[token_num, attention_rank_id, layer_idx, start_expert, end_expert]

后续：
TP 个 prefix + R * TP 个 routed expert count
```

`start_expert/end_expert` 是当前 chunk 的专家区间，结束位置为闭区间。
`batch_info[0]` 是该 rank 的 TP 汇总 routed token 数；返回的 `counts[R]`
描述本次接收 chunk 的专家 token 数。本次计算输入长度取 `counts.sum()`，
不能直接用前者裁剪。

其他重要约束：

- 新 dispatch recv、combine send 的相关标量参数明确为 `max_seq_len`，
  用于通信窗口布局，不应替换成本次实际 routed token 数。
- dispatch recv 分配 `floor(262144 * BATCH_SIZE_FACTOR)` 行，只有 counts
  描述的有效行可用于计算。
- binding 要求 contiguous、相同 device、无梯度；expert IDs 为 INT32，
  routing weights 为 FP32。
- send 的单 rank batch 需要满足 `B <= max_seq_len / TP`。
- 零 routed token 的 rank 仍必须调用 combine send，完成通信通知。
- 单专家的 TP 汇总 token 必须能放入一个接收 chunk。
- 后续 dispatch 不得覆盖尚未完成 combine recv 的统计数据。

接口来源：
[源码 binding](../../csrc/npu/pybind/torch_binding_cam_async.cpp)、
[当前 connector](../../afd_plugin/connectors/npu/async_cam.py)、
[源码算子协议说明](CAM_ASYNC_ROUTED_OPS.md)。

## 4. 阶段 1：验证源码算子和部署环境

### 4.1 实施内容

1. 固定 AFD commit、vLLM/vLLM-Ascend revision、CANN/HCCL、
   PyTorch/torch-npu 版本及模型 checkpoint。
2. 在目标 910C 环境完整构建 AFD 算子包；首次验收不跳过 ACLNN 构建。
3. 验证 `_C_ascend` 加载成功，四个 `afd_async_*` 均已注册，
   原有 A2E/E2A 同时可用。
4. 修改异步 loader，加载 AFD 自有扩展并显式检查四个异步算子。
   缺失时启动失败；现有仅检查 A2E/E2A 的逻辑不足以验证异步算子可用性。
   同步删除旧 `umdk_cam_op_lib` 导入、旧 namespace 注册检查，以及旧 CAM
   `.so` 的显式加载、动态库查找和兼容回退逻辑。保留源码构建的 AFD
   `_C_ascend`、AFD vendor 库及正常 CANN/HCCL 运行时加载逻辑。
5. 增加启动诊断，记录 namespace、插件 revision 和实际 vendor library
   路径，确认测试没有误用旧 CAM。

构建入口：

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 \
  pip install -e . -v --no-build-isolation
```

### 4.2 四阶段通信验证

| 验证维度 | 必须覆盖 |
| --- | --- |
| 基础通信 | dispatch send/recv、combine send/recv 完整闭环 |
| 精度 | FP16、BF16；dynamicQuant=0/1 |
| 路由 | expert 0、末尾 expert、非均匀路由、空专家、空 FFN rank |
| 拓扑 | TP1、TP2，以及 DSV4 所需 TP4 |
| 生命周期 | 重复使用窗口、连续 batch、多个接收 chunk |
| 正确性 | 与独立 routed 加权求和实现比较，量化时纳入实际量化误差 |

测试前固定 dtype/量化容差，禁止通过扩大容差掩盖迁移错误。
通过条件为无通信挂起、越界、NaN/Inf，且数值误差满足预设容差。

阶段产出：可部署的源码算子包、环境清单和四阶段通信验证记录。

## 5. 阶段 2：迁移 connector 和 FFN 工作项

主要修改：
[async_cam.py](../../afd_plugin/connectors/npu/async_cam.py)、
[ops.py](../../afd_plugin/compat/npu/ops.py)、
[FFN runner](../../afd_plugin/v1/worker/npu/ffn_model_runner.py)。

### 5.1 算子调用与状态

1. 替换四个调用，dispatch recv 从七个返回值改为四个，
   combine send 删除 `expand_x_shared`。
2. 明确区分通信容量 `max_seq_len`、实际输入 batch 和本次 routed token 数。
3. transfer state 保留 routed 输入、dynamic scales、expert counts 和原始
   compact metadata，删除异步路径的 shared 输入、scale 和 token count。
4. compact metadata 原样保留到对应 combine send，不能用裁剪或重建的
   header 替代。
5. 公共 payload 若仍被其他 connector 使用，只移除异步路径依赖，
   不全局删除。

### 5.2 FFN 工作项和调度

1. 使用 `counts.sum()` 确定本次计算长度，从 metadata 读取实际 layer
   和专家区间。
2. FFN 只调用 routed expert 计算，不再要求 shared tensor。
3. 零 routed token 仍执行 combine send，保留有效的浮点空工作处理方式。
4. 核实“一次接收”与“一层完整工作”的关系，检查现有按层循环和
   ubatching 调度能否消费同层多个 chunk。
5. 必要时依据协议完成状态推进，不能收到一个 chunk 就默认整层完成。
6. 验证空 rank 通知不会被跳过，也不会阻塞其他 rank。

### 5.3 配置检查

- 专家数与 FFN 分布满足 compact 协议。
- TP 整除 Attention rank 数。
- IDs、weights 的 dtype、shape、contiguous 满足新 binding。
- padding 后的 batch 满足 `B <= max_seq_len / TP`。
- 所有 rank 的协议、量化、维度和窗口参数一致。

阶段产出：纯 routed connector 闭环，以及普通和两阶段执行均可消费的
FFN 工作项。

## 6. 阶段 3：迁移两种模型的共享专家

共享专家迁移与 connector 切换必须一起落地，不能在缺少模型适配时直接
启用 routed-only 通信。

| 工作项 | DSV2-Lite | DSV4-Flash |
| --- | --- | --- |
| Attention 构造 | 在现有 gate/remote proxy 旁构造 shared MLP | 在 Attention MoE 结构中构造原生 shared 分支 |
| 权重归属 | shared 权重转交 Attention | 同时处理 checkpoint 的 `ffn` 与运行时 `mlp` 名称映射 |
| FFN 构造 | 仅保留 routed experts | 保留 routed experts 所需量化和计算属性 |
| 路由 | 保持原有 top-k 语义 | 保持 Hash、普通路由、`tid2eid` 与 token 对齐 |
| 模型数值 | 保持 scaling 和 dtype 分支 | 保持 scaling、SwiGLU clamp 和原生量化行为 |

### 6.1 构造与权重加载

复用固定版本上游 shared MLP 构造及量化实现。先核实模型实际量化配置，
不能将旧 INT8 通信输入计算路径直接作为 Attention 本地计算路径。

shared 权重及 scale、bias 等相关参数完整加载到 Attention；FFN 不再分配
shared 模块。测试需检查参数归属，避免“构造但未加载”或两侧重复加载。

主要涉及：
[DSV2 模型](../../afd_plugin/model_executor/models/deepseek_v2.py)、
[DSV2 MoE 计算](../../afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py)、
[DSV4 模型](../../afd_plugin/model_executor/models/npu/deepseek_v4.py)。

### 6.2 TP/SP 布局和数值语义

以 native shared 分支的输入输出布局为准，逐项明确：

- shared MLP 输入是完整 token 还是 SP shard。
- shared 权重是否按 TP 分片。
- 所需 all-gather、all-reduce、reduce-scatter 在何处完成。
- 与 routed output 相加前，两者是否处于同一个 token 坐标空间。

不能在某个 TP rank 的局部 token 上计算出不完整 shared 结果后直接相加。
分模型核对 routed scale 是否已经进入 top-k weights，以及现有 FP16
特殊处理；对 shared、routed 和最终输出分别建立参考比较。

先完成普通路径，再接入两阶段路径。初始实现优先保证依赖顺序正确，
利用现有 dispatch/recv 间隙安排 shared 计算，不额外引入复杂流调度。

阶段产出：两个模型都能完成“本地 shared + 远端 routed”，且逐层数值
与参考实现对齐。

## 7. 阶段 4：两阶段 ubatching 和资源生命周期

每个 pending stage 保存自己的 layer、dispatch layout、路由信息及
shared 结果，不能共用一个可覆盖的 shared buffer。

必须验证：

- stage 0/1 的 shared 与 routed output 按层、按 stage 正确配对。
- padding token 在布局恢复时正确裁剪。
- 普通 TP 和 FlashComm1/SP 不重复计算或遗漏真实 token。
- 同一 TP group 的 shared collective 执行顺序一致。
- 后续 dispatch 不提前覆盖前一个 combine recv 所需窗口数据。
- 请求结束、异常和取消时，pending 状态及 tensor 引用能够释放。
- Attention 新增 shared 权重、激活和两阶段暂存后，profiling/KV cache
  预算仍正确。

DSV4 的目标是 Attention DP2/TP4，同时存在两个 Attention DP 组和每组
内部 TP 通信，必须在目标拓扑验证，不能用 TP1 成功代替。

主要涉及两个模型的 async forward 文件、
[async_cam_layout.py](../../afd_plugin/model_executor/models/npu/async_cam_layout.py)
及相关 ubatching 调度。

## 8. 验证矩阵与通过标准

### 8.1 必须通过的现有 E2E

| 验收项 | 拓扑和配置 | 通过标准 |
| --- | --- | --- |
| DSV2-Lite async CAM | Attention DP1/TP2 + FFN EP2，4 卡 | 现有 completion 用例通过，确认使用源码算子，补充固定输入结果对比 |
| DSV2-Lite async ubatch | Attention DP1/TP2 + FFN EP1，3 卡，token split、两阶段 | GSM8K 前 7 条，实际样本数为 7，strict-match ≥0.27，并与固定参考结果比较 |
| DSV4-Flash async CAM | Attention DP2/TP4 + FFN EP8，16 卡，dynamicQuant=1、两阶段 | 10 请求并发全部正常返回，非空、`finish_reason=stop`，并通过新增答案校验 |

现有入口：

- [DSV2-Lite async E2E](../../tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py)
- [DSV4-Flash async E2E](../../tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py)
- [DSV4 固定配置](../../tests/e2e/models/deepseek_v4_flash/config.py)

### 8.2 正确性补强

DSV2 的 0.27 是现有最低门槛，不能单独作为精度不退化的证据。
保存相同 checkpoint、prompt、seed 和采样参数下的参考结果，对变化样本
回查逐层 routed/shared 输出。

DSV4 当前用例仅检查并发、非空输出和正常结束，不检查算术答案。
在现有十请求测试中加入预期答案检查，保留原始响应和时间信息；另外增加
固定输入的模型数值对比，覆盖 Hash 和非 Hash 层，不能只凭 HTTP 成功
认定模型正确。

补充边界测试覆盖：单 token decode、不等长请求、TP 非整除长度 padding、
prefill/decode 连续切换、空专家/空 FFN rank、多 chunk 和重复窗口使用。
真实 E2E 保持原生路由，不启用强制均衡绕过稀疏分布。

### 8.3 单元测试与兼容性回归

针对实际行为变化补充或更新：

- loader 的四算子注册检查和缺失失败路径。
- 不安装旧 wheel、不提供旧 CAM vendor `.so` 时，新 loader 和 E2E
  正常运行；源码算子缺失时明确失败，不尝试旧二进制加载或回退。
- 四返回值解包、compact metadata、chunk 有效长度和空 rank 通知。
- shared 参数构造及权重归属、量化参数加载。
- shared/routed 缩放与布局恢复。
- 两阶段结果配对、Hash input IDs 对齐和异常清理。

复用现有 connector、模型、layout、ubatching、package 和 Meta 测试。
Meta 测试只能证明注册及分配契约，不能替代 NPU 通信验证。

另外执行同步 CAMP2P 的既有 DSV2-Lite gate 回归，确认公共 loader、
权重归属及模型修改没有影响原有路径。

### 8.4 运行前提

运行前确认模型、GSM8K 缓存、`vllm`、pytest、lm_eval、datasets、
huggingface_hub、torch-npu 和 Ascend runtime 可用；设置 `HF_HOME` 并确认
数据源可达。固定模型路径及实际量化配置。

以下命令应在 E2E 环境初始化已迁移到 AFD vendor 后执行。
用例串行运行，不共享并发占用的设备；缺依赖、设备不足、skip 或通信超时
均不算通过。取消时发送 SIGTERM，给予充分清理时间，并核实进程、端口和
设备资源释放。

### 8.5 执行命令

DSV2-Lite 两个用例分别执行：普通 completion 使用四张 910C，
ubatch 使用三张。设备列表必须与各用例要求严格一致：

```bash
AFD_E2E_BACKEND=npu \
AFD_E2E_DEVICES=0,1,2,3 \
AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite \
HF_HOME=/path/to/huggingface \
AFD_GSM8K_LIMIT=7 \
AFD_GSM8K_THRESHOLD=0.27 \
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_cam

AFD_E2E_BACKEND=npu \
AFD_E2E_DEVICES=0,1,2 \
AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite \
HF_HOME=/path/to/huggingface \
AFD_GSM8K_LIMIT=7 \
AFD_GSM8K_THRESHOLD=0.27 \
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_ubatch
```

DSV4-Flash 使用十六张 910C；提前设置有效的 `HCCL_IF_IP` 和
`HCCL_SOCKET_IFNAME`：

```bash
AFD_E2E_BACKEND=npu \
AFD_E2E_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V4-Flash \
HF_HOME=/path/to/huggingface \
python -m pytest -q -s \
  tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py
```

## 9. 旧依赖清理与交付顺序

当前 E2E 环境仍硬编码旧 CAM vendor；DSV4 还设置
`CAM_CUST_OPAPI_LIB_PATH` 和 `LD_PRELOAD`。这些必须在迁移过程中同步
移除，避免旧运行环境掩盖包加载问题；本节清理要求是迁移的前置组成部分，
不是 E2E 之后的可选工作。

移除范围包括：

- Python 中旧 `umdk_cam_op_lib` 导入和旧算子注册检查。
- 旧 CAM `.so` 的显式加载调用、库路径探测、存在性检查及回退分支。
- 为旧 CAM 设置的 `CAM_CUST_OPAPI_LIB_PATH`、`CAM_VENDOR` 和硬编码
  vendor 路径。
- `LD_PRELOAD` 中旧 CAM `.so` 条目，以及 `LD_LIBRARY_PATH`、
  `ASCEND_CUSTOM_OPP_PATH` 中针对旧 CAM 的路径注入。
- E2E、启动/部署脚本和测试 fixture 中对应的环境初始化、mock 和断言。
- 文档中的旧 wheel、旧 `.run` 包安装及 `.so` 预加载要求。

路径清理只针对旧 CAM 条目，不删除无关环境配置，也不移除源码构建的
AFD 扩展、AFD vendor 库和 CANN/HCCL 运行时所需加载逻辑。
验收前检查相关代码和脚本没有残留旧加载入口，并在不安装旧 wheel、
不提供旧 CAM vendor 库的环境中执行目标 E2E，记录实际加载的 AFD 库路径。

| 顺序 | 交付内容 | 完成条件 |
| --- | --- | --- |
| 1 | 源码构建、loader、旧 `.so` 加载逻辑移除、设备通信测试 | 四算子完成数值闭环，不加载或回退到旧 CAM 库 |
| 2 | connector、FFN work item、双模型 shared 迁移 | 普通路径逐层正确，全部参与 rank 使用新协议 |
| 3 | TP/SP 与两阶段调度 | DSV2 两个 async 用例通过 |
| 4 | DSV4 量化、Hash 路由、16 卡验收 | 十请求并发及新增正确性检查通过 |
| 5 | 文档与依赖收尾、残留检查 | 旧加载入口已移除，无旧 CAM wheel/vendor 环境下完成全部目标验收 |

实现遵循仓库 AGENTS.md：优先继承和组合；涉及复制或包装上游实现时，
注明固定版本来源、patch 原因与行为，并标记 AFD 差异。

## 10. 最终完成清单

- [x] 910C 完整构建成功，四个源码算子及 A2E/E2A 注册正常。
  构建 exit 0、产物 hash 与运行验证已记录；初次构建 stdout 未持久化。
- [x] 原有二进制 `.so` 导入、显式加载、路径注入、预加载、存在性检查和
  回退逻辑已同步移除，E2E 与部署脚本没有残留旧 CAM 加载入口。
- [x] 独立四阶段通信测试通过，含量化、TP、空 rank 和多 chunk。
  12 组真实设备矩阵均通过，独立 CPU oracle 最大绝对误差为 0。
- [x] connector 与 FFN runner 使用 compact routed-only 协议。
  真实通信矩阵、双模型 E2E 与 compact/empty/chunk 相关单测提供证据。
- [x] 两种模型的 shared 权重和计算迁到 Attention，逐层数值对齐。
  DSV2 BF16/FP16 layer1、DSV4 BF16 Hash0/非 Hash3 的 shared/routed/final
  固定输入 checkpoint probe 均通过；非完整模型 baseline，误差见验收记录。
- [x] TP/SP、两阶段调度、窗口生命周期及异常清理通过验证。
  真实矩阵和 E2E、stage/异常单测及独立资源检查共同覆盖。
- [x] DSV2-Lite 两个 async E2E 用例通过，零 skip。
  S6 四卡 completion 与三卡 ubatch 分别通过；GSM8K strict 为 2/7（0.2857）。
- [x] DSV4-Flash 16 卡十请求并发及新增正确性检查通过，零 skip。
  S10 pytest exit 0、scenario passed，十条答案为 19–28、finish_reason 均为 stop；
  原始响应与独立退出资源清理记录已归档。
- [x] 同步 CAMP2P 相关回归通过。
  `afd-eager-2a2f` gate 通过，GSM8K strict 2/7；退出阶段现象见验收记录。
- [x] 在未安装旧 CAM wheel、未提供旧 CAM vendor `.so` 的环境中完成
  目标验收，并记录实际加载的 AFD 源码构建库路径。
- [x] 附构建记录、软件与模型版本、命令、pytest 结果、GSM8K 结果、
  DSV4 十条响应、数值误差报告和资源清理记录。
  构建记录含真实命令、exit 0 观察与产物 hash，明确缺少初次原始 stdout。

只有两个模型的目标 E2E 全部通过，并证明实际运行的是源码算子，才算
本次适配完成。

验收进度、证据与待完成项见 [验收记录](CAM_ASYNC_SOURCE_MIGRATION_VALIDATION.md)。
