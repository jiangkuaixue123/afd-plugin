# Ascend AFD 自定义算子构建

AFD CAMP2P 依赖插件自有的 Ascend 自定义算子，用于
Attention-to-Expert 和 Expert-to-Attention 传输：

- 隔离 dispatcher namespace：`torch.ops.afd_ascend.a2e`
- 隔离 dispatcher namespace：`torch.ops.afd_ascend.e2a`

完整的 A2E/E2A C++/AscendC 源码随仓库放在：

```text
csrc/a2e
csrc/e2a
csrc/aclnn_torch_adapter
```

## 默认行为

本地 CPU/macOS 开发默认不构建 Ascend 算子。除非显式开启构建，否则 Python 包保持
import-safe。

## 在 Ascend 环境中构建

使用以下命令安装：

```bash
AFD_BUILD_ASCEND_OPS=1 \
SOC_VERSION=910c \
pip install -e .
```

常用环境变量：

- `ASCEND_HOME_PATH`：CANN toolkit 路径。默认值为
  `/usr/local/Ascend/ascend-toolkit/latest`。
- `TORCH_NPU_PATH`：可选，用于显式指定 `torch_npu` 包路径。
- `SOC_VERSION`：当前 `910c` / `ascend910_93*` 会构建 `a2e;e2a`。
- `MAX_JOBS`：并行 CMake 构建任务数。
- `AFD_SKIP_ACLNN_BUILD=1`：跳过 ACLNN 算子包重建，只基于已有自定义算子库构建
  PyTorch extension。

构建产物会安装到包内：

```text
afd_plugin/_C_ascend*.so
afd_plugin/_cann_ops_custom/
```

## vLLM-Ascend 共存要求

AFD 自定义算子必须能与 vLLM-Ascend `v0.19.1rc1` 在同一进程中共存。迁移目标是：

- Python extension 保持由插件包拥有，例如 `afd_plugin._C_ascend`；
- 不要把 A2E/E2A 注册到 vLLM-Ascend 的 `torch.ops._C_ascend` namespace；
- AFD CANN 自定义算子安装到 AFD 自有 vendor 路径下，例如
  `afd_plugin/_cann_ops_custom/vendors/afd-plugin/...`，而不是
  `vendors/vllm-ascend`；
- 通过明确的包内路径加载 AFD 的 `libcust_opapi.so`，不要使用裸
  `dlopen("libcust_opapi.so")`；loader 会设置 `AFD_CUST_OPAPI_LIB_PATH`；
- 如果当前激活的 vLLM-Ascend 包已经提供 A2E/E2A，应 fail fast 或显式复用已有实现。

运行时通过以下方式加载 extension：

```python
from afd_plugin.compat.ascend import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
```

## 当前范围

这条构建路径只迁移第一版单流 CAMP2P connector 所需的 AFD A2E/E2A 算子。
多流、量化路径、ACL graph 和 gate-on-attention 仍放在后续阶段处理。
