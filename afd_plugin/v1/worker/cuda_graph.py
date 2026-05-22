# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA graph policy helpers for AFD runtimes.

This module intentionally avoids importing torch or vLLM at module import time.
It works with real vLLM config objects and with the small SimpleNamespace fakes
used by CPU-safe tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
_SUPPORTED_GRAPH_MODES = {FULL_DECODE_ONLY}


class AFDGraphRunMode(str, Enum):
    EAGER = "eager"
    WARMUP = "warmup"
    CAPTURE = "capture"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class AFDCUDAGraphPolicy:
    """Resolved AFD CUDA graph policy for one runtime role."""

    enabled: bool
    mode_name: str | None
    allow_attention_full_decode_only: bool
    enable_ffn_graph_cache: bool
    allow_cuda_graph_with_ubatching: bool = False


def validate_cuda_graph_mode(
    vllm_config: object,
    *,
    role: str | None = None,
) -> AFDCUDAGraphPolicy:
    """Return the CUDA graph policy or raise for unsupported AFD modes."""

    enforce_eager = bool(getattr(vllm_config.model_config, "enforce_eager", False))
    mode_name = cudagraph_mode_name(vllm_config)
    graph_enabled = not enforce_eager

    if not graph_enabled:
        return AFDCUDAGraphPolicy(
            enabled=False,
            mode_name=mode_name,
            allow_attention_full_decode_only=False,
            enable_ffn_graph_cache=False,
        )

    if mode_name not in _SUPPORTED_GRAPH_MODES:
        role_suffix = f" for {role}" if role else ""
        raise RuntimeError(
            "AFD Phase 6 only supports CUDA graph mode "
            f"{FULL_DECODE_ONLY}{role_suffix}; got {mode_name!r}.",
        )

    allow_ubatching = _allow_cuda_graph_with_ubatching(vllm_config)
    if _is_ubatching_enabled(vllm_config) and not allow_ubatching:
        num_ubatches = getattr(vllm_config.parallel_config, "num_ubatches", None)
        raise RuntimeError(
            "AFD CUDA graph support currently supports ubatching only for "
            f"{FULL_DECODE_ONLY} with exactly two ubatches; "
            f"got num_ubatches={num_ubatches!r}.",
        )

    return AFDCUDAGraphPolicy(
        enabled=True,
        mode_name=mode_name,
        allow_attention_full_decode_only=role in (None, "attention"),
        enable_ffn_graph_cache=role in (None, "ffn"),
        allow_cuda_graph_with_ubatching=allow_ubatching,
    )


def cudagraph_mode_name(vllm_config: object) -> str | None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None:
        return None

    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name

    text = str(mode)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text or None


def make_ffn_graph_key(dp_metadata_list: dict[int, Any]) -> tuple[tuple[int, tuple]]:
    """Extract the original AFD-style hashable key from DP metadata."""

    key_parts: list[tuple[int, tuple]] = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values = getattr(metadata, "num_tokens_across_dp_cpu", None)
        if values is None:
            values_tuple = (repr(metadata),)
        else:
            tolist = getattr(values, "tolist", None)
            if callable(tolist):
                values = tolist()
            elif hasattr(values, "item"):
                values = [values.item()]
            try:
                values_tuple = tuple(int(value) for value in values)
            except TypeError:
                values_tuple = (int(values),)
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)


def graph_run_mode(
    *,
    is_warmup: bool,
    is_graph_capturing: bool,
    graph_enabled: bool,
    graph_exists: bool,
) -> AFDGraphRunMode:
    if is_warmup:
        return AFDGraphRunMode.WARMUP
    if is_graph_capturing:
        return AFDGraphRunMode.CAPTURE
    if graph_enabled and graph_exists:
        return AFDGraphRunMode.REPLAY
    return AFDGraphRunMode.EAGER


def _is_ubatching_enabled(vllm_config: object) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return bool(getattr(parallel_config, "use_ubatching", False))


def _allow_cuda_graph_with_ubatching(vllm_config: object) -> bool:
    if not _is_ubatching_enabled(vllm_config):
        return False
    parallel_config = vllm_config.parallel_config
    return int(getattr(parallel_config, "num_ubatches", 0)) == 2


__all__ = [
    "AFDCUDAGraphPolicy",
    "AFDGraphRunMode",
    "FULL_DECODE_ONLY",
    "cudagraph_mode_name",
    "graph_run_mode",
    "make_ffn_graph_key",
    "validate_cuda_graph_mode",
]
