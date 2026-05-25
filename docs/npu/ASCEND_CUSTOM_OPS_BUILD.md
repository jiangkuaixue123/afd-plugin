# Ascend AFD Custom Ops Build

AFD CAMP2P depends on plugin-owned Ascend custom ops for Attention-to-Expert
and Expert-to-Attention transfer:

- `torch.ops._C_ascend.a2e`
- `torch.ops._C_ascend.e2a`

The full A2E/E2A C++/AscendC sources are vendored under:

```text
csrc/a2e
csrc/e2a
csrc/aclnn_torch_adapter
```

## Default Behavior

Local CPU/macOS development does not build Ascend ops. The Python package stays
import-safe unless the build is explicitly enabled.

## Build In An Ascend Environment

Install with:

```bash
AFD_BUILD_ASCEND_OPS=1 \
SOC_VERSION=910c \
pip install -e .
```

Useful environment variables:

- `ASCEND_HOME_PATH`: CANN toolkit path. Defaults to
  `/usr/local/Ascend/ascend-toolkit/latest`.
- `TORCH_NPU_PATH`: optional explicit `torch_npu` package path.
- `SOC_VERSION`: currently `910c` / `ascend910_93*` builds `a2e;e2a`.
- `MAX_JOBS`: parallel CMake build jobs.
- `AFD_SKIP_ACLNN_BUILD=1`: skip ACLNN op package rebuild and only build the
  PyTorch extension against existing custom op libraries.

Build outputs are installed into the package:

```text
afd_plugin/_C_ascend*.so
afd_plugin/_cann_ops_custom/
```

At runtime, load the extension through:

```python
from afd_plugin.compat.ascend import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
```

## Current Scope

This build path migrates only the AFD A2E/E2A ops required by the first
single-stream CAMP2P connector. Multistream, quantized paths, ACL graph, and
gate-on-attention remain separate phases.
