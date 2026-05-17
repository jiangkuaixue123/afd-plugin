# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Lightweight opt-in tracing for AFD runtime debugging."""

from __future__ import annotations

import os
import time
from typing import Any


def afd_trace_enabled() -> bool:
    return os.environ.get("AFD_TRACE") == "1" or os.environ.get("AFD_P2P_TRACE") == "1"


def afd_trace(event: str, **fields: Any) -> None:
    if not afd_trace_enabled():
        return
    rendered_fields = " ".join(
        f"{key}={_format_value(value)}" for key, value in sorted(fields.items())
    )
    print(
        f"AFD_TRACE ts={time.time():.6f} event={event} {rendered_fields}",
        flush=True,
    )


def tensor_summary(value: Any) -> str:
    if value is None:
        return "None"
    shape = tuple(getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    return f"shape={shape},dtype={dtype},device={device}"


def dp_metadata_summary(dp_metadata_list: dict[int, Any]) -> str:
    parts = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        token_counts = getattr(metadata, "num_tokens_across_dp_cpu", None)
        if token_counts is None:
            token_counts = getattr(metadata, "num_tokens_across_dp", None)
        if token_counts is None:
            token_counts = getattr(metadata, "num_tokens", None)
        parts.append(f"{stage_idx}:{_format_value(token_counts)}")
    return "[" + ",".join(parts) + "]"


def _format_value(value: Any) -> str:
    if hasattr(value, "tolist"):
        try:
            return repr(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return repr([_format_value(item) for item in value])
    return repr(value)


__all__ = [
    "afd_trace",
    "afd_trace_enabled",
    "dp_metadata_summary",
    "tensor_summary",
]
