# CAM Async 源码算子迁移验收记录

本记录对应 [迁移计划](CAM_ASYNC_SOURCE_MIGRATION_PLAN.md)。目标数值验证与 E2E 已通过；
初次构建原始 stdout 未持久化，该证据限制在下文明确记录。算子源码 `csrc/` 未修改。

## 环境与快照

- 验证环境：itask `jcz_afd1`，16 张 Ascend 910C。
- 远端工作树：`/a3_inference/itask/workdir/wb02075298/jcz_afd7/code/afd-code/afd-plugin`。
- 已观测 runtime：CANN 9.0.1，torch 2.10.0+cpu，torch-npu 2.10.0.post2，
  vLLM 0.26.0+empty；Python 3.12.13，Transformers 5.14.1，
  huggingface_hub 1.26.0，datasets 5.0.1，lm_eval 0.4.12。
  CANN 版本读取自 `/usr/local/Ascend/cann-9.0.1/opp/version.info`。
  HCCL 9.0.1，读取自 `share/info/hccl/version.info` 和
  `python/site-packages/hccl/sys_version.py`；实际 `lib64/libhccl.so` SHA256
  `1188f6acb2df4de12b887ab3d0a270c59d0b745e11f692929a6a6635c8cf3e4c`。
- vLLM 源码 `/vllm-workspace/vllm`，commit
  `2e882b2b64acd5180a1548a8f7f26e15992e3cae`；vLLM-Ascend 源码
  `/vllm-workspace/vllm-ascend`，commit `cf26a4d4b491a17956bc0e41c8523a4e14cbf0e7`，
  package `0.19.1rc2.dev1308+gcf26a4d4b`。两包均无 direct_url 元数据。
  从真实 E2E 工作目录使用同一 `python3` 确认 vLLM 导入文件为
  `/vllm-workspace/vllm/vllm/__init__.py`；证据
  `s11-e2e-cwd-vllm-import-path.log`。早期 itask 默认目录下的 namespace
  shadow 不代表 E2E runtime，最终 provenance 已纠正。
- DSV2 模型：`/home/admin/model-csi/models/modelhub_97542_deepseek-v2-lite-36500041_20260318110950/model`。
- DSV4 模型：`/home/admin/model-csi/models/modelhub_51300042_deepseek-v4-flash-w8a8-mtp-126600065_20260527131010/model`。
- 数值验证快照：S10；DSV2 在 S6 验证通过，S7 为格式化快照，19 个已跟踪改动 Python 文件
  格式化前后 AST 完全相同，所有改动与新入口均通过 Ruff check。修正 HF strict config 与 native MoE 的 shared sentinel
  类型差异；仅对 native 构造使用的配置浅拷贝设置 `None`，原始配置保持不变。
- S7 Python 快照：21 个改动或新增文件，aggregate SHA256
  `4eb705c27ff478cc9ee70fdc273fd7ae5fce581a9c2d6c9473934129ca111156`。
  逐文件 manifest：`/home/nielinfeng/afd-download/cam-source-migration-20260921/s7-python-snapshot.json`。
- S10-final Python 快照：23 个文件，aggregate SHA256
  `8098d0ed2a511d464ff0ac128caa14bf6bd99e588eb4d70ef041f627fa3dce0d`。
  manifest：`/home/nielinfeng/afd-download/cam-source-migration-20260921/s10-final-python-snapshot.json`。
  与数值验证时的 S10 相比，仅追加 selector 测试覆盖。
- S11 仅将启动诊断 logger 移入 `vllm.afd_plugin.compat.npu.ops` namespace，
  继承 worker 的 INFO handler，避免被 root WARNING 过滤；数值实现保持 S10。
  最终 23 个 Python 文件的 aggregate SHA256 为
  `bbad1d6740774c78e30d16d6d674393bd712e061cc052b64896b325d43a684bc`，
  manifest：`/home/nielinfeng/afd-download/cam-source-migration-20260921/s11-python-snapshot.json`。
- 插件版本：`0.26.0rc2.dev42+gafb0b83d2.d20260921`。真实 S11 Attention
  worker 启动日志记录 `namespace=afd_ascend` 与完整 vendor library 路径：
  `/a3_inference/itask/workdir/wb02075298/jcz_afd7/code/afd-code/afd-plugin/afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_api/lib/libcust_opapi.so`。

DSV4 检查点确认使用 W8A8：shared `w1/w2/w3.weight` 为 int8，每个投影
拥有 FP32 `weight_scale` 和 `weight_offset`；132 个 offset 张量、360448 个
元素均为零。未使用 FP8 `weight_scale_inv` 约定。DSV2 检查点无
`quant_model_description.json`，其 config 没有量化配置。

## 已完成证据

| 项目 | 结果 | 范围与限制 |
| --- | --- | --- |
| 四阶段真实设备通信矩阵 | 12/12 通过，全部 CPU oracle 最大绝对误差 0 | TP1/2/4 × FP16/BF16 × dynamicQuant0/1；含稀疏路由、首尾 expert、空 FFN rank、两 chunk、decode 和重复窗口 |
| 源码加载隔离 | 已确认 `_C_ascend` 从源码构建加载，旧 `umdk_cam_op_lib` 不可导入 | 实际 worker 日志和产物 hash 已归档；初次构建 stdout 缺失 |
| S6 DSV2 两个 async E2E | 两例均通过，零 skip；ubatch GSM8K strict 2/7（0.2857） | 四卡 completion 与三卡 ubatch 分别执行；不替代逐层数值验收 |
| DSV2 实际 checkpoint shared/routed/final | BF16、FP16 均通过，容差未变 | layer 1，非 Hash；FP16 由同一 BF16 checkpoint 转换，误差见下表 |
| S10 DSV4 checkpoint Hash0/非 Hash3 | 两层均通过固定容差，shared 完全一致 | native/AFD FP32 logits 最大误差 0，选中 ID mismatch 0；路由/最终误差见下文 |
| S10 DSV4 16 卡 E2E | pytest exit 0、scenario passed，10/10 算术答案为 19–28 | Attention NPU0–7：DP2TP4；FFN NPU8–15：EP8；dynamicQuant1、ubatch2；10 条原始响应均为 stop；资源清理记录已归档 |
| 同步 CAMP2P gate 回归 | `afd-eager-2a2f` 通过，GSM8K strict 2/7 | 与 async 总分一致；样本、prompt、target hash 与过滤后答案逐项一致 |
| strict 配置问题定位 | S5 失败已明确；S6 两个 E2E 通过 | 实际 HF 配置修正后测试通过，日志已归档 |

矩阵入口：

```bash
torchrun --standalone --nproc-per-node=$((TP + 2)) \
  -m tests.e2e.operators.async_cam_roundtrip \
  --tp "$TP" --dtype "$DTYPE" --dynamic-quant "$QUANT"
```

DSV2 E2E 必须分别以四卡运行普通 completion、三卡运行 ubatch；
具体命令见迁移计划 8.5。使用四卡环境运行整个文件会被 ubatch harness 拒绝，
这种配置错误不计为模型通过或失败。

分别取 `TP=1,2,4`、`DTYPE=float16,bfloat16`、`QUANT=0,1`。
该 oracle 独立构造输入、路由和量化参考，不从接收的 payload/counts 反推预期。


DSV2 实际检查点 layer 1 数值误差（每格为最大绝对误差 / 相对 L2）：

| 比较项 | BF16 | FP16 |
| --- | --- | --- |
| native shared vs AFD shared | 0 / 0 | 0 / 0 |
| 原始 checkpoint CPU shared vs AFD shared | 7.629e-6 / 1.1628e-4 | 1.9073e-6 / 8.504e-5 |
| native routed vs AFD routed | 2.2888e-5 / 4.1096e-3 | 5.722e-6 / 5.6959e-4 |
| native final vs AFD final | 1.5259e-5 / 7.4437e-4 | 1.9073e-6 / 1.6014e-4 |

证据根目录为远端 `/a3_inference/itask/workdir/wb02075298/jcz_afd7/code/afd-code/validation-logs`。
DSV2 数值产物位于其 `dsv2-moe-bf16/` 和 `dsv2-moe-fp16/`，
对应进程日志为同名 `.log` 文件。
这是固定输入的真实 MoE 单层比较，不是完整模型 baseline。

## DSV4 诊断进度

S8 实际 checkpoint probe 的 Hash layer 0 通过，但非 Hash layer 3 的 routed
相对 L2 为 0.6787、final 为 0.5334，未通过固定容差。S9 诊断确认 AFD 原有
BF16 gate logits 与 native FP32 internal router 不一致：29 个选中 expert ID
不同；给 AFD 选择器相同 FP32 logits 后，ID mismatch 为 0，权重误差为 0。

S10 将 DSV4 gate 改为 native 相同的
`F.linear(hidden_states.float(), gate.weight_fp32)`；仅修改 Python 适配层，
未修改算子源码。S10 双层复验通过：Hash0 的 routed/final 相对 L2 为
0.00757/0.00688，非 Hash3 为 0.00960/0.00773；shared 完全一致。两层
native/AFD FP32 logits 最大误差和选中 ID mismatch 均为 0。容差未改变。
相关单元测试 12 项通过，随后补充的 selector 分支覆盖仅修改测试。
S7/S10/S11 fingerprint 分别标识对应验证快照，不互相替代。

## 测试汇总与执行入口

目标单元测试组 152 passed、2 skipped；两个 skip 是已安装 Ascend 扩展环境
下的“扩展缺失”反向测试，不是目标 E2E skip。最终 gate selector 测试 12
passed；S11 loader 日志测试 8 passed、同样 2 个环境 skip。所有改动 Python
通过 Ruff，compileall、diff 检查通过，`git diff -- csrc` 为空。

完整 E2E 命令见迁移计划 8.5，必须分别执行四卡 DSV2 completion、三卡
DSV2 ubatch 和十六卡 DSV4。同步回归使用既有 `afd-eager-2a2f` gate 场景。
详细服务命令、环境和结果保留在对应原始日志中。

真实检查点数值入口（模型路径必须指向实际验证检查点）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python -m tests.e2e.operators.moe_checkpoint_reference \
  --family dsv2 --model /path/to/DeepSeek-V2-Lite \
  --dtype bfloat16 --output /path/to/evidence/dsv2-moe-bf16
```

DSV2 还需执行 `--dtype float16`。DSV4 使用 `--family dsv4`，默认同时检查
Hash 和普通路由层。脚本保存固定输入、token IDs、路由、shared/routed/final
张量及误差 JSON；独立 CPU shared MLP 直接读取原始 checkpoint 权重。
容差固定，不因设备结果调整。单层 probe 不替代目标拓扑 E2E；GSM8K 和
HTTP 响应也不能替代该逐层数值证明。

## Profiling 与退出行为

DSV4 S10 profile 报告可用内存约 27.78 GiB，KV 容量约 1.994M tokens。
DSV2 ubatch 的可用 KV 内存为 38.56 GiB，KV 容量为 1,331,072 tokens。
这些数字来自已归档实际 profile 日志，属于当前实际拓扑/配置，
不表示其他 TP/DP 配置下的容量保证。

DSV4 推理结果保存后，退出日志包含 SIGTERM/SIGKILL 和 FFN
`KeyboardInterrupt` 告警。随后独立资源检查确认 NPU 无运行进程，服务端口和服务进程已释放。

## 已知边界

同步 CAMP2P 回归在响应完成、结果保存后的退出阶段（约 1–3 秒）记录了
FFN control plane `recv_dp_metadata_list` 的 Gloo `Connection closed by peer`。
所有 NPU 和端口随后确认已释放。当前没有旧快照 baseline 证据，不能认定
该退出行为由本次迁移引入；它不改变本次已保存的推理结果。

- Async CAM 不支持 `mix_placement=true`，在配置验证时明确拒绝。
- shared MLP 权重在 Attention TP rank 上复制，按本地 token shard 计算。
  实际权重内存和 KV 预算应以两模型 profile 日志为准。
- Python 异常清理释放 pending payload 引用；失败的通信 group 必须关闭，
  不承诺在未关闭的硬件窗口上恢复执行。
- S11 为采集启动诊断进行的 scratch 启动存在 FFN 配置拼写错误，未达到
  整体 ready，也未获取完整服务 maps；该次仅提供真实 Attention worker
  loader 日志，不作为新一轮 E2E。数值与十请求证据来自完整通过的 S10。
  scratch 资源已清理。

## 模型与最终环境指纹

| 文件 | SHA256 |
| --- | --- |
| DSV2 `config.json` | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| DSV4 `config.json` | `c3137398b85ef26b9592debadc424349f0b949b7ebe2f20603dda12efab4d533` |
| DSV4 `quant_model_description.json` | `612d87d0a9a7545fb19f8cbaff75af3cc467d16ada72d2b6d84d22149a511cf0` |

最终环境数据在本地归档根目录 `final-provenance-s11.json`，SHA256
`b3516c30a621e34a1237e679ea2bad0aaa757e164933dc75ef007a1670b71964`。
S11 远端 23/23 文件 hash 与本地一致，见
`s11-final-remote-python-snapshot-check.log`。模型指纹记录配置与量化描述，
没有声称对全部 checkpoint shard 做完整内容哈希。

## 构建产物与归档

实际构建命令为：

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 pip install -e . -v --no-build-isolation
```

验证者观察到 exit 0，但原始 stdout 未持久化；没有重建或伪造该构建日志。现存产物、实际注册/通信执行和后续 worker 加载提供运行证据。

| 产物 | SHA256 |
| --- | --- |
| `afd_plugin/_C_ascend.cpython-312-aarch64-linux-gnu.so` | `c928e1ab587a632ace0246e3a7a8aa0e5931439d32df64206389b9cdc0d3d900` |
| `afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_api/lib/libcust_opapi.so` | `6eb4a7c0280f8b76594bd1b2484b05acec86570ca149688199553e68d1b4a03d` |

本地归档根目录为 `/home/nielinfeng/afd-download/cam-source-migration-20260921`：

- `validation-logs-cam-source-migration-20260921.tar.gz`：独立通信矩阵。
- `cam-source-migration-20260921-incremental-s8.tar.gz`：DSV2 E2E、两 dtype
  layer reference、同步回归和逐样本对照；也保留早期失败，勿将失败日志当最终结果。
- `cam-source-migration-20260921-final-s10.tar.gz`：S10 DSV4 双层 reference、
  十六卡 E2E、十条响应、资源释放、旧 CAM 隔离检查及 selector 单测。
- `cam-source-migration-20260921-s11-provenance.tar.gz`：S11 loader 测试、
  实际 worker 日志、scratch 失败与清理、runtime/产物信息；SHA256
  `0384079f61fc3327fb7cfa753678e2e59f6becc29aec183ff1cac96da3b54b72`。

远端 evidence 根目录内的关键文件：

| 证据 | 文件 |
| --- | --- |
| DSV2 completion / ubatch | `s6-dsv2-async-completion-and-ubatch-gsm8k7.log` / `s6-dsv2-async-ubatch-gsm8k7-rerun.log` |
| DSV2 数值 | `dsv2-moe-bf16/report.json`、`dsv2-moe-fp16/report.json` 与 `layer-1.pt` |
| 同步回归/逐样本对照 | `s7-dsv2-camp2p-eager-2a2f-gsm8k7.log`、`dsv2-async-vs-camp2p-gsm8k7-compare.json` |
| DSV4 数值 | `dsv4-moe-bf16-s10/report.json`、`layer-0.pt`、`layer-3.pt` |
| DSV4 E2E/响应 | `s10-dsv4-async-cam-16npu.log`、`s10-dsv4-10-concurrent-responses.json` |
| DSV4 清理 | `s10-post-dsv4-16npu-resource.log` |
| 旧 CAM 隔离 | `s10-old-cam-isolation.log` |
| S11 真实 worker loader | `s10-dsv4-source-maps-attn.log`（文件名沿用 S10，内容为 S11 诊断启动） |
| S11 scratch 清理 | `s11-post-scratch-cleanup-final.log` |

旧 `umdk_cam_op_lib` 未安装且 import spec 为 None；当前环境没有旧 CAM
vendor 路径注入。隔离扫描包含其他历史工作树中的 wheel/run 安装包文件，
它们不是当前环境已安装或已加载的依赖，不应声称整个磁盘不存在这些文件。

最终源码残留检查：产品、E2E、部署脚本与当前 recipe 文本无旧 CAM 加载入口；
仓库保留的历史 wheel 二进制不是运行依赖，未作为 fallback 安装或加载。
