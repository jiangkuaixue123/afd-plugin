# AFD Plugin 迁移指南

本仓库的目标是把原本位于 vLLM 主仓内的 AFD 实现迁移为一个
out-of-tree 的 vLLM external plugin。

## 项目目标

- 将 `afd-plugin` 构建为 vLLM 的 external plugin，用于支持
  Attention-FFN Disaggregation (AFD)。
- 目标运行版本：vLLM `v0.19.1`。本地参考 checkout 位于 `../vllm`，
  当前已确认 tag 为 `v0.19.1`。
- 不修改 vLLM `v0.19.1` 源码树。所有行为都必须由本插件包、运行时注册、
  显式 CLI class path、console script，或本仓库内范围清晰的兼容 shim 提供。
- 保持原始 AFD 实现的行为，参考 vLLM 分支 `afd_gpu` 中的 commit
  `0ce8b91b937ec5d47b6902867c4275e0c5fb895e`。
- 以 `../dllm-plugin` / `../dllm-plugin/dllm_plugin` 作为 external plugin
  包结构、可选 vLLM 依赖、`vllm.general_plugins` entry point 注册、
  兼容辅助层、校验逻辑和测试组织方式的主要参考。

## 参考来源

- vLLM 目标版本：`../vllm`
- 原始 AFD commit：
  `0ce8b91b937ec5d47b6902867c4275e0c5fb895e`
- External plugin 参考项目：`../dllm-plugin`

重建行为时，应先查看原始 AFD commit，再设计新代码。

## GPU 迁移计划

GPU 迁移阶段、初始目录结构和当前设计决策记录在
`docs/gpu/MIGRATION_PLAN.md`。

## NPU 迁移计划
GPU 迁移阶段、当前设计决策记录在
`docs/npu/`下面

## 开发规则

- 不要编辑 `../vllm` 或 `../vllm-ascend` 下的文件。
- 优先用 `git -C ../vllm show <commit>:<path>` 阅读原始 AFD 代码。
- 保持 vLLM `v0.19.1` 兼容性显式可见，并通过测试覆盖。
- 优先使用 plugin-owned class 和显式 dotted class path，而不是 monkey patch。
  如果 monkey patch 无法避免，必须保证它幂等、受版本保护、有文档说明且有测试。
- 每个包含真实行为的 phase 都要配套添加测试。GPU test 应是 opt-in，或在缺少
  CUDA/vLLM runtime dependency 时干净 skip。
- 如果数据结构使用了 vLLM 或 vLLM Ascend 中的数据结构，请直接访问其函数或
  成员变量，不要使用 `getattr` 或 `hasattr` 之类的方法，也不要主动抛出异常；
  这样后续升级时可以直接根据原始报错判断哪里不适配。

## 代码开发风格

以下规则参考 `../vllm-ascend/AGENTS.md`，并按 AFD external plugin 的边界调整。

### Python 约定

- import 默认放在文件顶部。允许例外：
  - 解决循环 import；
  - 顶层插件入口、entry point、package `__getattr__`、worker/isolation 进程边界
    的受控 lazy import；
  - 仅用于类型检查的 `if TYPE_CHECKING:` import。
- 避免新增全局变量。允许 `ALL_UPPER_CASE` 常量和不可变配置对象；新增可变全局状态
  必须有明确生命周期、并发语义和测试覆盖，例如 custom op 注册表或进程级通信句柄。
- 不要写 magic number。阈值、默认值、协议常量和环境变量名应使用有描述性的常量。
- 命名保持直接、语义化：
  - class 使用 `PascalCase`；
  - 函数、方法、变量使用 `snake_case`；
  - 常量使用 `ALL_UPPER_CASE`；
  - 名称描述功能和语义，不用 `flag1`、`tmp_var` 这类实现细节名。
- 新增环境变量必须集中定义和说明；如果没有现成集中模块，先补充清晰的 env/helper
  模块，不要把 `os.getenv()` 和环境变量字符串散落到运行时代码中。

### CPU-Safe 与 Runtime 边界

- 只要求顶层插件入口保持 CPU-safe，包括 `afd_plugin/__init__.py`、entry point
  注册、配置解析、class path 字符串校验和兼容 patch 注册层。这些位置可以使用少量
  受控 lazy import、显式 `ImportError` 处理和 debug 日志，以便没有
  vLLM/CUDA/NPU 环境时仍能运行包级、配置级单元测试。
- runtime 模块不追求 CPU-safe。`afd_plugin/v1/worker/**`、
  `afd_plugin/model_executor/models/**` 以及实际数据路径上的 GPU/NPU connector
  代码，应按 vLLM / vLLM-Ascend 风格在模块层直接导入 `torch`、`vllm`、
  `vllm_ascend` 等运行时依赖；缺少依赖或 API 不兼容时应 fail fast。
- package `__init__.py` 可以通过 `__getattr__` 暴露显式 class path，避免仅导入
  package 时加载 runtime 模块；不要在 runtime 类内部再用 fallback base class、
  `optional_class`、假对象或为了 CPU-only 测试而设计的兼容壳。
- runtime 代码不要用宽泛 `try/except Exception` 隐藏真实错误。只有入口注册、
  版本兼容 shim、资源清理线程边界等明确需要容错的地方可以捕获异常；应优先捕获
  具体异常，并保留可诊断日志。

### NPU 性能注意事项

- 在 NPU 热路径中谨慎使用 device tensor 的 `item()`。它会触发 NPU 到 CPU 的同步，
  可能阻塞调度或降低吞吐；使用前要确认必要性，能批量同步就不要逐个同步，能留在
  device 上计算就不要搬到 CPU。
- 避免热路径中的不必要 CPU-NPU 数据搬运和频繁同步；优先使用 device-side 操作和
  安全的 in-place 操作。
- 涉及 NPU runtime、ACL graph、custom op、通信路径的改动，需要在真实 NPU 硬件上
  做 smoke 或性能验证。

### 架构与迁移风格

- 从 vLLM / vLLM-Ascend 迁移或贴近其行为的代码，应尽量保持 upstream 风格：
  使用真实类型、真实 helper、直接成员访问和原始错误路径。不要为未来未知版本过度
  封装，也不要复制已有 runtime 数据结构的 fallback 版本。
- `model_runner` 新增行为需要严格审查：先确认是否必须放在 plugin-owned runner 中，
  是否可以通过更小范围的 patch、wrapper 或 connector 解决，并说明性能影响和测试
  覆盖。
- patch 必须最小化、幂等、受版本保护，并说明长期方案。新增 patch 前要确认目标
  upstream 组件正确，避免用 patch 掩盖设计边界不清的问题。
- 测试也要遵守同样边界：CPU-safe 测试集中在 package/config/validation/compat
  注册层；直接 import runtime 类或依赖 `torch`、`vllm`、`vllm_ascend` 的测试应
  使用 marker 或 `pytest.importorskip()` 干净跳过，而不是迫使生产代码支持假
  runtime。

## 远程 GPU 验证

- 远程 L20X GPU 服务器上的 `afd-plugin` 代码目录为
  `/home/jcz/sources/afd-plugin`。
- 需要验证本地代码分支时，先从当前分支拉出临时测试分支并 push 到远程仓库，
  再登录远程 L20X 服务器，在 `/home/jcz/sources/afd-plugin` 中 pull/checkout
  该测试分支进行验证。
- 远程验证完成后，删除本地和远程的临时测试分支，避免测试分支长期残留。
