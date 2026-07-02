from __future__ import annotations

from pathlib import Path


def test_ascend_a2e_e2a_sources_are_vendored():
    root = Path(__file__).resolve().parents[3]
    required = [
        "csrc/npu/a2e/op_host/aclnn_a2e.cpp",
        "csrc/npu/a2e/op_kernel/a2e.cpp",
        "csrc/npu/a2e/op_kernel/comm_args.h",
        "csrc/npu/a2e/op_kernel/moe_distribute_base.h",
        "csrc/npu/e2a/op_host/aclnn_e2a.cpp",
        "csrc/npu/e2a/op_kernel/e2a.cpp",
        "csrc/npu/e2a/op_kernel/comm_args.h",
        "csrc/npu/e2a/op_kernel/moe_distribute_base.h",
        "csrc/npu/build_aclnn.sh",
        "csrc/npu/torch_extension/CMakeLists.txt",
        "csrc/npu/torch_extension/torch_binding.cpp",
        "csrc/npu/torch_extension/torch_binding_meta.cpp",
    ]

    for relpath in required:
        assert (root / relpath).is_file(), relpath


def test_ascend_ops_build_is_enabled_by_default():
    root = Path(__file__).resolve().parents[3]
    setup_py = (root / "setup.py").read_text()

    assert 'AFD_BUILD_ASCEND_OPS", "1"' in setup_py
    assert "csrc/npu/build_aclnn.sh" in setup_py


def test_ascend_ops_use_isolated_namespace_and_vendor_path():
    root = Path(__file__).resolve().parents[3]
    torch_binding = (root / "csrc/npu/torch_extension/torch_binding.cpp").read_text()
    torch_binding_meta = (
        root / "csrc/npu/torch_extension/torch_binding_meta.cpp"
    ).read_text()
    torch_cmake = (root / "csrc/npu/torch_extension/CMakeLists.txt").read_text()
    cann_cmake = (root / "csrc/npu/CMakeLists.txt").read_text()
    op_api_common = (root / "csrc/npu/aclnn_torch_adapter/op_api_common.h").read_text()

    assert "TORCH_LIBRARY(afd_ascend" in torch_binding
    assert "TORCH_LIBRARY(_C_ascend" not in torch_binding
    assert "TORCH_LIBRARY_IMPL(afd_ascend, Meta" in torch_binding_meta
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta" not in torch_binding_meta
    assert "vendors/afd-plugin/op_api/lib" in torch_cmake
    assert "vendors/vllm-ascend/op_api/lib" not in torch_cmake
    assert '"afd-plugin"' in cann_cmake
    assert '"vllm-ascend"' not in cann_cmake
    assert "AFD_CUST_OPAPI_LIB_PATH" in op_api_common
    assert 'return "libcust_opapi.so"' not in op_api_common
