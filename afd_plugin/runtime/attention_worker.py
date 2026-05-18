# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for the Phase 2 MVP."""

from __future__ import annotations

from typing import Any

from afd_plugin.runtime._optional import optional_class
from afd_plugin.runtime.attention_model_runner import (
    AFDAttentionModelRunner,
    fail_if_unsupported_ubatching,
)
from afd_plugin.validation import assert_compatible_afd_stack

_GPUWorker, _GPUWorker_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.gpu_worker",
    "Worker",
)


class AFDAttentionWorker(_GPUWorker):  # type: ignore[misc, valid-type]
    """Attention worker that injects :class:`AFDAttentionModelRunner`."""

    afd_expected_role = "attention"
    vllm_base_import_error = _GPUWorker_IMPORT_ERROR

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _GPUWorker_IMPORT_ERROR is not None:
            raise RuntimeError(
                "AFDAttentionWorker requires an importable vLLM runtime",
            ) from _GPUWorker_IMPORT_ERROR
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        """Initialize the native GPU worker and swap in the AFD runner."""

        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDAttentionWorker.init_device",
            expected_role="attention",
        )
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD Attention runtime currently supports only the vLLM v1 "
                "GPUModelRunner; unset VLLM_USE_V2_MODEL_RUNNER for Phase 2",
            )

        fail_if_unsupported_ubatching(self.vllm_config)

        super().init_device()
        native_model_runner = self.model_runner
        self.model_runner = AFDAttentionModelRunner(self.vllm_config, self.device)
        del native_model_runner

        try:
            import torch

            torch.accelerator.empty_cache()
        except Exception:
            pass


__all__ = ["AFDAttentionWorker"]
