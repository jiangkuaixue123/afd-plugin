# AFD Plugin 迁移指南

本仓库的目标是把原本位于 vLLM 主仓内的 AFD 实现迁移为一个
out-of-tree 的 vLLM external plugin。

## 项目目标

- 将 `afd-plugin` 构建为 vLLM 的 external plugin，用于支持
  Attention-FFN Disaggregation (AFD)。
- 目标运行版本：vLLM `v0.20.2`。本地参考 checkout 位于 `../vllm`，
  当前已确认 tag 为 `v0.20.2`。
- 不修改 vLLM `v0.20.2` 源码树。所有行为都必须由本插件包、运行时注册、
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

重建行为时，应先查看原始 AFD commit，再设计新代码。重要的原始文件包括：

- `vllm/config/afd.py`
- `vllm/distributed/afd_transfer/afd_connector/*`
- `vllm/entrypoints/afd_ffn_server.py`
- `vllm/entrypoints/cli/fserver.py`
- `vllm/forward_context.py`
- `vllm/distributed/parallel_state.py`
- `vllm/model_executor/models/deepseek_v2.py`
- `vllm/model_executor/models/step3_text.py`
- `vllm/model_executor/models/step3_vl.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/worker/gpu_ffn_model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_ubatch_wrapper.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/ubatching.py`

## 迁移阶段

### Phase 0：基线与兼容性盘点

- 确认 vLLM `v0.20.2` 的 plugin hook 和 class-path 扩展点。
- 将 AFD commit 与目标 vLLM 版本做 diff，并按 config、connector、
  distributed state、engine、worker、model、CLI、example 分类。
- 判断哪些 in-tree 改动可以变成普通 plugin class，哪些需要兼容 shim。

### Phase 1：包骨架与 Plugin 注册

- 添加 Python packaging metadata，并提供 `vllm.general_plugins` entry point。
- 如果需要支持 macOS/本地无 CUDA wheel 的开发体验，参考 `dllm-plugin`，
  将 `vllm` 保持为可选 runtime extra。
- 添加幂等的 `register_afd()` 函数。
- 添加面向 vLLM `v0.20.2` 的早期 stack validation 和版本检查。

### Phase 2：AFD 配置面

- 将 `AFDConfig` 迁移到插件内。
- 在不修改 `vllm.config.VllmConfig` 的前提下，提供构造 AFD config 的方式。
- 支持通过 CLI/env/extra-config 加载 attention 和 FFN 两种 role 的配置。
- 保留原始实现中会影响 hash 的字段和 role helper。

### Phase 3：Connector 层

- 迁移 connector contract 和 metadata。
- 先接入 dummy connector，用于本地 smoke test。
- contract 稳定后再迁移 P2P connector。
- 通过 factory 隔离 connector 创建逻辑，让 backend 选择不泄漏到 worker
  和 runner。

### Phase 4：Worker 与 Runner 集成

- 将 FFN 侧 model runner 行为迁移为 plugin-owned class。
- FFNModelRunner 第一版可以直接基于原始 AFD 实现中的 `GPUFFNModelRunner`。
- AttentionModelRunner 第一版应继承 vLLM v1 原生 `GPUModelRunner`，只覆盖
  AFD 必需行为。
- Attention 侧 runner/worker 行为只通过显式 class path 或兼容 shim 接入。
- 除非 vLLM `v0.20.2` 没有可用 class-path hook，否则避免隐式全局 patch。

### Phase 5：Model 集成

- 将 DeepSeek V2 和 Step3 模型中的 AFD 相关改动迁移为 plugin-owned model
  implementation 或 wrapper。
- 在 `register_afd()` 中通过 vLLM model registry 注册模型架构。
- 模型特定的 AFD 逻辑不要放进通用 connector 或 worker 模块。

### Phase 6：FFN 侧 vLLM Serve 集成

- FFN 不沿用原始 AFD commit 中新增的 `fserver` 入口。
- FFN 侧应优先使用原生 `vllm serve` 启动方式，通过插件配置、role 配置、
  显式 class path 或必要兼容 shim 切换到 FFN runtime。
- 不设计 `vllm fserver` 子命令，也不优先设计 `vllm-afd-fserver` console
  script。

### Phase 7：端到端验证

- 为 config、metadata、connector factory 和 validation 添加单元测试。
- 添加 CPU-safe 的 import 和 registration smoke test。
- 添加 GPU-gated integration test，覆盖 attention/FFN role 启动、
  dummy connector 行为和 P2P connector 行为。
- 为 DeepSeek V2 和 Step3 添加示例 runbook。

### Phase 8：硬化与运维文档

- 文档化支持的 topology 组合。
- 文档化已知限制、必需环境变量和故障模式。
- 添加 FFN server 和 connector traffic 的 profiling/debugging 说明。
- 在明确测试其他版本之前，兼容性说明都绑定到 vLLM `v0.20.2`。

## 初始目录结构

下面先只约定目录边界和目录名，暂不决定具体文件。类级别 API、文件拆分方式、
模块命名后续继续讨论；在讨论完成前，不要为了占位而提前创建细粒度文件。

```text
afd-plugin/
  AGENTS.md
  README.md
  pyproject.toml

  afd_plugin/
    compat/
      patches/
    connectors/
    distributed/
    entrypoints/
    models/
    runtime/

  docs/
  examples/
  tests/
```

### 目录职责

- `afd_plugin`：插件主包。顶层只放全局注册、配置、校验、轻量公共入口等跨目录
  模块；具体文件名后续再定。
- `afd_plugin.compat`：vLLM `v0.20.2` 的版本保护、延迟 import、兼容 helper
  和 shim。所有与目标 vLLM 版本强绑定的兼容逻辑优先集中在这里。
- `afd_plugin.compat.patches`：不得不 monkey patch vLLM 时使用的隔离区。这里的
  patch 必须幂等、受版本保护、有文档说明，并且只在没有可用 plugin/class-path
  扩展点时使用。
- `afd_plugin.connectors`：AFD 通信 contract 和 backend implementation。
- `afd_plugin.distributed`：从原始 `parallel_state.py` 改动中抽取出的
  AFD-specific distributed helper。这里应优先放插件自己调用的 helper，而不是
  patch。
- `afd_plugin.models`：plugin-owned model implementation 或 wrapper。
- `afd_plugin.runtime`：vLLM 可通过显式 class path 加载的运行时 adapter/class，
  包括 worker、runner、ubatching、forward-context 相关能力。这里不作为 patch
  目录使用。
- `afd_plugin.entrypoints`：保留给必须由插件提供的 executable entrypoint。
  当前方向是 FFN 也使用原生 `vllm serve`，因此不要优先在这里实现原始
  `fserver` 等价入口。
- `docs`：迁移说明、架构决策、operator runbook 和已知限制。
- `examples`：可运行示例和部署/online serving 样例。
- `tests`：测试目录。单元测试、集成测试、GPU-gated 测试的具体分层后续再定。

## 当前设计决策

- FFN 侧不要沿用原始 AFD commit 中新增的 `fserver` 入口；优先复用原生
  `vllm serve` 启动路径。
- FFNModelRunner 第一版可以直接使用或迁移原始 AFD 实现中的
  `GPUFFNModelRunner`。
- AttentionModelRunner 第一版继承 vLLM v1 原生 `GPUModelRunner`，并只加入
  AFD 必需的最小覆盖逻辑。

## 开发规则

- 不要编辑 `../vllm` 或 `../dllm-plugin` 下的文件。
- 优先用 `git -C ../vllm show <commit>:<path>` 阅读原始 AFD 代码。
- 保持 vLLM `v0.20.2` 兼容性显式可见，并通过测试覆盖。
- 优先使用 plugin-owned class 和显式 dotted class path，而不是 monkey patch。
  如果 monkey patch 无法避免，必须保证它幂等、受版本保护、有文档说明且有测试。
- 尽量保持 package import CPU-safe。CUDA-heavy module 应延迟 import。
- 每个包含真实行为的 phase 都要配套添加测试。GPU test 应是 opt-in，或在缺少
  CUDA/vLLM runtime dependency 时干净 skip。
- 除非有 AFD-specific 的理由，否则遵循 `../dllm-plugin` 的风格和 packaging 约定。
