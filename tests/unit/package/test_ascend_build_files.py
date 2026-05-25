from __future__ import annotations

from pathlib import Path


def test_ascend_a2e_e2a_sources_are_vendored():
    root = Path(__file__).resolve().parents[3]
    required = [
        "csrc/a2e/op_host/aclnn_a2e.cpp",
        "csrc/a2e/op_kernel/a2e.cpp",
        "csrc/e2a/op_host/aclnn_e2a.cpp",
        "csrc/e2a/op_kernel/e2a.cpp",
        "csrc/build_aclnn.sh",
        "csrc/torch_extension/CMakeLists.txt",
        "csrc/torch_extension/torch_binding.cpp",
        "csrc/torch_extension/torch_binding_meta.cpp",
    ]

    for relpath in required:
        assert (root / relpath).is_file(), relpath


def test_ascend_ops_build_is_opt_in_by_default():
    root = Path(__file__).resolve().parents[3]
    setup_py = (root / "setup.py").read_text()

    assert 'AFD_BUILD_ASCEND_OPS", "0"' in setup_py
    assert "csrc/build_aclnn.sh" in setup_py
