# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path

AFD_ASCEND_OPS_NAMESPACE = "afd_ascend"
AFD_ASCEND_VENDOR_NAME = "afd-plugin"
AFD_CUST_OPAPI_ENV = "AFD_CUST_OPAPI_LIB_PATH"


def get_afd_cann_vendor_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "_cann_ops_custom"
        / "vendors"
        / AFD_ASCEND_VENDOR_NAME
    )


def get_afd_cust_opapi_path() -> Path:
    return get_afd_cann_vendor_path() / "op_api" / "lib" / "libcust_opapi.so"


def _prepend_env_path(name: str, path: Path) -> None:
    path_str = str(path)
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(":") if entry]
    if path_str in entries:
        return
    os.environ[name] = ":".join([path_str, *entries])


def _ensure_custom_opp_env() -> None:
    vendor_dir = get_afd_cann_vendor_path()
    if not vendor_dir.exists():
        return

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    op_api_lib = vendor_dir / "op_api" / "lib"
    if op_api_lib.exists():
        _prepend_env_path("LD_LIBRARY_PATH", op_api_lib)
    cust_opapi = get_afd_cust_opapi_path()
    if cust_opapi.exists():
        os.environ[AFD_CUST_OPAPI_ENV] = str(cust_opapi)


def _assert_afd_namespace_registered() -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to load AFD Ascend custom ops") from exc

    missing = [
        name
        for name in ("a2e", "e2a")
        if not _has_torch_op(torch, AFD_ASCEND_OPS_NAMESPACE, name)
    ]
    if missing:
        joined = ", ".join(
            f"torch.ops.{AFD_ASCEND_OPS_NAMESPACE}.{name}" for name in missing
        )
        raise RuntimeError(
            "AFD Ascend extension loaded but did not register the isolated "
            f"dispatcher namespace: {joined}",
        )


def _has_torch_op(torch: object, namespace: str, op_name: str) -> bool:
    try:
        ops = torch.ops
        ns = getattr(ops, namespace)
        getattr(ns, op_name)
    except (AttributeError, RuntimeError):
        return False
    return True


@lru_cache(maxsize=1)
def ensure_afd_ascend_ops_loaded() -> None:
    """Import the compiled extension that registers ``torch.ops.afd_ascend``.

    The extension is optional at package import time.  It is built only when
    ``AFD_BUILD_ASCEND_OPS=1`` is set in an Ascend environment.
    """

    _ensure_custom_opp_env()
    try:
        importlib.import_module("torch")
        importlib.import_module("afd_plugin._C_ascend")
    except Exception as exc:
        raise RuntimeError(
            "AFD Ascend custom ops are not available. Build the package with "
            "AFD_BUILD_ASCEND_OPS=1 in a torch-npu/CANN environment.",
        ) from exc
    _assert_afd_namespace_registered()


def has_afd_ascend_ops() -> bool:
    try:
        ensure_afd_ascend_ops_loaded()
    except RuntimeError:
        return False
    return True


__all__ = [
    "AFD_ASCEND_OPS_NAMESPACE",
    "AFD_ASCEND_VENDOR_NAME",
    "AFD_CUST_OPAPI_ENV",
    "ensure_afd_ascend_ops_loaded",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "has_afd_ascend_ops",
]
