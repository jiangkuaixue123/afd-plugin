# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe validation helpers for AFD runtime wiring."""

from __future__ import annotations

import importlib
from typing import Any, Final

from afd_plugin.config import AFDConfig, parse_afd_config

ATTENTION_WORKER_FQCN: Final[str] = "afd_plugin.runtime.AFDAttentionWorker"
FFN_WORKER_FQCN: Final[str] = "afd_plugin.runtime.AFDFFNWorker"
ATTENTION_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.runtime.AFDAttentionModelRunner"
FFN_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.runtime.GPUFFNModelRunner"
UBATCH_WRAPPER_FQCN: Final[str] = "afd_plugin.runtime.AFDUBatchWrapper"


def normalize_qualname(value: str) -> str:
    return value.replace(":", ".")


def resolve_class_from_qualname(qualname: str, *, role: str = "class") -> type[Any]:
    """Resolve a dotted or colon-separated class path."""

    normalized = normalize_qualname(qualname.strip())
    if not normalized or "." not in normalized:
        raise ValueError(
            f"{role} must be a dotted qualname, got {qualname!r}",
        )
    module_name, obj_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    if not isinstance(obj, type):
        raise TypeError(
            f"{role} resolved to {type(obj).__name__}, expected a class",
        )
    return obj


def expected_worker_qualname(role: str) -> str:
    if role == "attention":
        return ATTENTION_WORKER_FQCN
    if role == "ffn":
        return FFN_WORKER_FQCN
    raise ValueError(f"unknown AFD role {role!r}")


def assert_compatible_afd_stack(
    vllm_config: object,
    *,
    caller: str,
    expected_role: str | None = None,
    require_enabled: bool = True,
) -> AFDConfig:
    """Validate AFD config and worker class wiring.

    This helper intentionally uses duck typing so unit tests and local CPU
    development do not need to construct a real vLLM ``VllmConfig``.
    """

    def _ctx() -> str:
        return f" (context: {caller!r})"

    config = parse_afd_config(vllm_config, expected_role=expected_role)
    if require_enabled and not config.enabled:
        raise ValueError(f"AFD is not enabled in additional_config['afd']{_ctx()}")

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        raise ValueError(f"missing parallel_config for AFD runtime stack{_ctx()}")

    worker_cls_raw = getattr(parallel_config, "worker_cls", "")
    if not isinstance(worker_cls_raw, str):
        raise ValueError(
            "parallel_config.worker_cls must be a qualname string "
            f"(got type {type(worker_cls_raw).__name__}){_ctx()}",
        )
    if worker_cls_raw.strip() == "auto":
        raise ValueError(
            "parallel_config.worker_cls is still 'auto'; pass --worker-cls "
            f"{expected_worker_qualname(config.role)}{_ctx()}",
        )

    worker_cls = resolve_class_from_qualname(
        worker_cls_raw,
        role="parallel_config.worker_cls",
    )
    expected_qualname = expected_worker_qualname(config.role)
    expected_worker_cls = resolve_class_from_qualname(
        expected_qualname,
        role="expected AFD worker class",
    )
    if worker_cls is not expected_worker_cls:
        worker_fqcn = normalize_qualname(
            f"{worker_cls.__module__}.{worker_cls.__name__}",
        )
        raise ValueError(
            "invalid worker class for AFD runtime stack: "
            f"got={worker_fqcn!r} expected={expected_qualname!r}; "
            f"pass --worker-cls {expected_qualname}{_ctx()}",
        )

    return config


__all__ = [
    "ATTENTION_MODEL_RUNNER_FQCN",
    "ATTENTION_WORKER_FQCN",
    "FFN_MODEL_RUNNER_FQCN",
    "FFN_WORKER_FQCN",
    "UBATCH_WRAPPER_FQCN",
    "assert_compatible_afd_stack",
    "expected_worker_qualname",
    "normalize_qualname",
    "resolve_class_from_qualname",
]
