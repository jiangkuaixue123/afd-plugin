# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def ensure_afd_ascend_ops_loaded() -> None:
    """Import the compiled extension that registers ``torch.ops._C_ascend``.

    The extension is optional at package import time.  It is built only when
    ``AFD_BUILD_ASCEND_OPS=1`` is set in an Ascend environment.
    """

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
