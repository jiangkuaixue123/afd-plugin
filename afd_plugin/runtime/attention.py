# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side runtime class-path placeholders."""

from __future__ import annotations

from typing import Any

from afd_plugin.config import parse_afd_config
from afd_plugin.runtime._optional import optional_class, phase1_placeholder_error

_GPUWorker, _GPUWorker_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.gpu_worker",
    "Worker",
)
_GPUModelRunner, _GPUModelRunner_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner",
)


class AFDAttentionWorker(_GPUWorker):  # type: ignore[misc, valid-type]
    """Phase 1 placeholder for the Attention worker class path."""

    afd_expected_role = "attention"
    vllm_base_import_error = _GPUWorker_IMPORT_ERROR

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise phase1_placeholder_error(type(self).__name__, _GPUWorker_IMPORT_ERROR)


class AFDAttentionModelRunner(_GPUModelRunner):  # type: ignore[misc, valid-type]
    """Phase 1 placeholder for the Attention model runner class path."""

    afd_expected_role = "attention"
    vllm_base_import_error = _GPUModelRunner_IMPORT_ERROR

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise phase1_placeholder_error(
            type(self).__name__,
            _GPUModelRunner_IMPORT_ERROR,
        )

    @staticmethod
    def parse_config(vllm_config: object):
        return parse_afd_config(vllm_config, expected_role="attention")


__all__ = ["AFDAttentionModelRunner", "AFDAttentionWorker"]
