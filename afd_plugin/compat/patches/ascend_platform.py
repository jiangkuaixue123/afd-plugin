# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""vLLM-Ascend platform compatibility patch for AFD-owned ubatching.

vLLM-Ascend 0.19.1rc1 resets native DBO/ubatch CLI flags in its platform
config fixup because the generic Ascend DBO path is not supported.  AFD NPU
uses plugin-owned workers and connector-driven ubatch handling, so preserve the
native vLLM CLI intent only for the explicit AFD NPU runtime class paths.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

from afd_plugin.config import parse_afd_config
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
    normalize_qualname,
)


def _get_logger() -> Any:
    try:
        from vllm.logger import logger as vllm_logger
    except Exception:
        import logging

        return logging.getLogger(__name__)
    return vllm_logger


logger = _get_logger()

TARGET_VLLM_ASCEND_VERSION: Final[str] = "0.19.1rc1"
_PATCH_ATTR = "_afd_plugin_ascend_platform_patch_state"
_NPU_AFD_WORKERS: Final[frozenset[str]] = frozenset(
    {
        NPU_ATTENTION_WORKER_FQCN,
        NPU_FFN_WORKER_FQCN,
    },
)


@dataclass
class _PatchState:
    npu_platform_fix_incompatible_config: Callable[[Any], None]


@dataclass(frozen=True)
class _NativeUbatchIntent:
    enable_dbo: bool
    ubatch_size: int


def apply_ascend_platform_patch() -> None:
    """Preserve native ubatch CLI flags for AFD NPU on vLLM-Ascend 0.19.1rc1."""

    if not _is_target_vllm_ascend_compatible():
        return
    try:
        platform_module = importlib.import_module("vllm_ascend.platform")
    except Exception:
        logger.debug("AFD Ascend platform patch skipped: vLLM-Ascend unavailable")
        return

    npu_platform_cls = getattr(platform_module, "NPUPlatform", None)
    if npu_platform_cls is None or hasattr(npu_platform_cls, _PATCH_ATTR):
        return

    state = _PatchState(
        npu_platform_fix_incompatible_config=(
            npu_platform_cls._fix_incompatible_config
        ),
    )

    def patched_fix_incompatible_config(vllm_config: Any) -> None:
        intent = _native_ubatch_intent_for_afd_npu(vllm_config)
        if intent is None:
            state.npu_platform_fix_incompatible_config(vllm_config)
            return

        parallel_config = vllm_config.parallel_config
        parallel_config.enable_dbo = False
        parallel_config.ubatch_size = 0
        try:
            state.npu_platform_fix_incompatible_config(vllm_config)
        finally:
            parallel_config.enable_dbo = intent.enable_dbo
            parallel_config.ubatch_size = intent.ubatch_size

    npu_platform_cls._fix_incompatible_config = staticmethod(
        patched_fix_incompatible_config,
    )
    setattr(npu_platform_cls, _PATCH_ATTR, state)
    logger.debug("AFD Ascend platform DBO preservation patch applied")


def _native_ubatch_intent_for_afd_npu(
    vllm_config: Any,
) -> _NativeUbatchIntent | None:
    try:
        afd_config = parse_afd_config(vllm_config, validate=False)
        parallel_config = vllm_config.parallel_config
        worker_cls = normalize_qualname(str(parallel_config.worker_cls).strip())
        enable_dbo = bool(parallel_config.enable_dbo)
        ubatch_size = int(parallel_config.ubatch_size or 0)
    except Exception:
        return None

    if not afd_config.enabled or afd_config.connector != "camp2pconnector":
        return None
    if worker_cls not in _NPU_AFD_WORKERS:
        return None
    if not enable_dbo and ubatch_size <= 1:
        return None
    return _NativeUbatchIntent(enable_dbo=enable_dbo, ubatch_size=ubatch_size)


def _is_target_vllm_ascend_compatible() -> bool:
    installed_version = _get_installed_vllm_ascend_version()
    if installed_version is None:
        return True
    version_text = str(installed_version)
    if "dev" in version_text:
        return True
    return version_text.startswith(TARGET_VLLM_ASCEND_VERSION)


def _get_installed_vllm_ascend_version() -> str | None:
    for distribution_name in ("vllm_ascend", "vllm-ascend"):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    try:
        version_module = importlib.import_module("vllm_ascend._version")
    except Exception:
        return None
    return str(getattr(version_module, "__version__", None) or "") or None


__all__ = ["TARGET_VLLM_ASCEND_VERSION", "apply_ascend_platform_patch"]
