# Running NPU unit tests

NPU unit tests depend on vLLM and vLLM-Ascend. Run them in an Ascend development
environment that uses the versions pinned by this repository.

From the repository root, run:

```bash
python3 -m pytest -q tests/unit -m "not gpu and not vllm_runtime"
```

The suite covers the NPU worker and model-runner contracts, CAM connector
behavior, and module isolation. Tests that require unavailable NPU dependencies
are skipped in CPU-only CI, so a passing CPU run does not replace this check.

`tests/conftest.py` establishes the vLLM-Ascend import order required by the
test suite. No manual module preloading is needed.

When diagnosing a failure, run the failing pytest node by itself first. A test
that passes alone but fails in the full suite usually indicates leaked module
or monkeypatch state. Test fixtures and mocks must also follow the concrete
vLLM types and signatures used by the pinned runtime.

## Layered W4A8 验证

先运行无需 NPU 的契约与旧路径回归：

```bash
pytest -q tests/unit/test_envs.py \
  tests/unit/model_executor/test_async_cam_w4a8.py \
  tests/unit/model_executor/models/test_w4a8_attention_gate.py
```

使用匹配 CANN / torch-npu 的环境重新编译 AFD 扩展，验证上游已合入的
device Tensor `group_list` 接口及本次新增的 W4A8 算子链：

```bash
SOC_VERSION=910c pytest -q tests/unit/compat/npu/test_gmm_layered.py
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c \
  pytest -q tests/npu/test_async_cam_layered_w4a8.py
```

NPU 用例使用两层不同的非零 INT4 权重、非零补偿、per-channel/per-group、
非连续且交错层号，以及零/单行/不均衡/满容量专家计数；与 FP32 参考比较有效
行并扰动容量尾部。用例只测试当前 `swiglu_limit=0` 算子链，不覆盖分布式 CAM
完成通知，也不代替真实 checkpoint 旧/新路径对比。此变更在无 NPU 环境完成，
NPU 用例尚未运行，其阈值与支持范围需按项目精度标准在目标设备确认。

端到端验收需以同一份 W4A8 启动配置切换 `AFD_ASYNC_CAM_LAYERED_GMM=0/1`，
记录 checkpoint、后处理参数 shape/dtype/format、软件版本、实际路径日志、
精度、峰值内存和 profiler。特别覆盖空 rank 的完成通知、同层多个 chunk、
DP 交错和 async CAM ubatching；接收至发送期间不应出现层号/计数的元数据 D2H。
保留的 worker 组间同步应单独计时。本次没有已测性能提升。

第二个 GMM 的 device Tensor `group_list` 接口已由上游 #384 合入。本次直接
使用该接口，不修改算子源码或 binding。
