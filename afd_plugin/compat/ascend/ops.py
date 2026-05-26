# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path


def _prepend_env_path(name: str, path: Path) -> None:
    path_str = str(path)
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(":") if entry]
    if path_str in entries:
        return
    os.environ[name] = ":".join([path_str, *entries])


def _ensure_custom_opp_env() -> None:
    vendor_dir = Path(__file__).resolve().parents[2] / "_cann_ops_custom" / "vendors" / "vllm-ascend"
    if not vendor_dir.exists():
        return

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    op_api_lib = vendor_dir / "op_api" / "lib"
    if op_api_lib.exists():
        _prepend_env_path("LD_LIBRARY_PATH", op_api_lib)


@lru_cache(maxsize=1)
def ensure_afd_ascend_ops_loaded() -> None:
    """Import the compiled extension that registers ``torch.ops._C_ascend``.

    The extension is optional at package import time.  It is built only when
    ``AFD_BUILD_ASCEND_OPS=1`` is set in an Ascend environment.
    """

    _ensure_custom_opp_env()
    try:
        importlib.import_module("afd_plugin._C_ascend")
    except Exception as exc:
        raise RuntimeError(
            "AFD Ascend custom ops are not available. Build the package with "
            "AFD_BUILD_ASCEND_OPS=1 in a torch-npu/CANN environment.",
        ) from exc


def has_afd_ascend_ops() -> bool:
    try:
        ensure_afd_ascend_ops_loaded()
    except RuntimeError:
        return False
    return True


__all__ = ["ensure_afd_ascend_ops_loaded", "has_afd_ascend_ops"]
