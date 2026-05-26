from __future__ import annotations

import pytest

from afd_plugin.compat.ascend import ops
from afd_plugin.compat.ascend.ops import (
    ensure_afd_ascend_ops_loaded,
    get_afd_cann_vendor_path,
    get_afd_cust_opapi_path,
    has_afd_ascend_ops,
)


def test_ascend_ops_loader_fails_clearly_without_extension():
    if has_afd_ascend_ops():
        pytest.skip("AFD Ascend extension is installed in this environment")

    with pytest.raises(RuntimeError, match="AFD Ascend custom ops"):
        ensure_afd_ascend_ops_loaded()


def test_ascend_ops_paths_are_plugin_owned():
    vendor_path = get_afd_cann_vendor_path()
    cust_opapi_path = get_afd_cust_opapi_path()

    assert vendor_path.parts[-3:] == ("_cann_ops_custom", "vendors", "afd-plugin")
    assert "vllm-ascend" not in vendor_path.parts
    assert cust_opapi_path.name == "libcust_opapi.so"
    assert cust_opapi_path.parts[-6:] == (
        "_cann_ops_custom",
        "vendors",
        "afd-plugin",
        "op_api",
        "lib",
        "libcust_opapi.so",
    )


def test_ascend_ops_loader_sets_explicit_cust_opapi_path(tmp_path, monkeypatch):
    vendor_path = (
        tmp_path / "afd_plugin" / "_cann_ops_custom" / "vendors" / "afd-plugin"
    )
    op_api_lib = vendor_path / "op_api" / "lib"
    op_api_lib.mkdir(parents=True)
    cust_opapi_path = op_api_lib / "libcust_opapi.so"
    cust_opapi_path.touch()
    monkeypatch.setattr(ops, "get_afd_cann_vendor_path", lambda: vendor_path)
    monkeypatch.setattr(ops, "get_afd_cust_opapi_path", lambda: cust_opapi_path)
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("AFD_CUST_OPAPI_LIB_PATH", raising=False)

    ops._ensure_custom_opp_env()

    assert str(vendor_path) in ops.os.environ["ASCEND_CUSTOM_OPP_PATH"]
    assert str(op_api_lib) in ops.os.environ["LD_LIBRARY_PATH"]
    assert ops.os.environ["AFD_CUST_OPAPI_LIB_PATH"] == str(cust_opapi_path)


def test_ascend_ops_namespace_check_allows_vllm_ascend_coexistence():
    class _Namespace:
        def __init__(self, names):
            self._names = set(names)

        def __getattr__(self, name):
            if name not in self._names:
                raise AttributeError(name)
            return object()

    class _Ops:
        _C_ascend = _Namespace({"a2e", "e2a"})
        afd_ascend = _Namespace({"a2e", "e2a"})

    class _Torch:
        ops = _Ops()

    assert ops._has_torch_op(_Torch, "_C_ascend", "a2e")
    assert ops._has_torch_op(_Torch, "afd_ascend", "a2e")
    assert ops._has_torch_op(_Torch, "afd_ascend", "e2a")
