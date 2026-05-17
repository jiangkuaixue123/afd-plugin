# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

from __future__ import annotations

from typing import Any

from afd_plugin.tracing import afd_trace


def maybe_apply_dbo_yield(
    tensor: Any,
    *,
    role: str,
    ubatching_module: Any | None = None,
) -> Any:
    """Yield to the peer ubatch thread when vLLM DBO is active."""

    if ubatching_module is None:
        try:
            from vllm.v1.worker import ubatching as ubatching_module
        except Exception:
            return tensor

    dbo_enabled = getattr(ubatching_module, "dbo_enabled", None)
    dbo_yield = getattr(ubatching_module, "dbo_yield", None)
    if not callable(dbo_enabled) or not callable(dbo_yield):
        return tensor
    if not bool(dbo_enabled()):
        return tensor

    afd_trace("dbo_yield_begin", role=role)
    dbo_yield()
    afd_trace("dbo_yield_done", role=role)
    return tensor


__all__ = ["maybe_apply_dbo_yield"]
