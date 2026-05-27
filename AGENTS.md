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
GPU 迁移阶段、当前设计决策记录再
`docs/npu/`下面

## 开发规则

- 不要编辑 `../vllm` 或 `../vllm-ascend` 下的文件。
- 优先用 `git -C ../vllm show <commit>:<path>` 阅读原始 AFD 代码。
- 保持 vLLM `v0.19.1` 兼容性显式可见，并通过测试覆盖。
- 优先使用 plugin-owned class 和显式 dotted class path，而不是 monkey patch。
  如果 monkey patch 无法避免，必须保证它幂等、受版本保护、有文档说明且有测试。
- 尽量保持 package import CPU-safe。CUDA-heavy module 应延迟 import。
- 每个包含真实行为的 phase 都要配套添加测试。GPU test 应是 opt-in，或在缺少
  CUDA/vLLM runtime dependency 时干净 skip。
- 如果数据结构使用了 vLLM 或 vLLM Ascend 中的数据结构，请直接访问其函数或
  成员变量，不要使用 `getattr` 或 `hasattr` 之类的方法，也不要主动抛出异常；
  这样后续升级时可以直接根据原始报错判断哪里不适配。

## 远程 GPU 验证

- 远程 L20X GPU 服务器上的 `afd-plugin` 代码目录为
  `/home/jcz/sources/afd-plugin`。
- 需要验证本地代码分支时，先从当前分支拉出临时测试分支并 push 到远程仓库，
  再登录远程 L20X 服务器，在 `/home/jcz/sources/afd-plugin` 中 pull/checkout
  该测试分支进行验证。
- 远程验证完成后，删除本地和远程的临时测试分支，避免测试分支长期残留。
