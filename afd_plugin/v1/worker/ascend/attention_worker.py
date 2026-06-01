# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for the first AFD runtime version."""

from __future__ import annotations

from typing import Any

from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.ascend import (
    apply_afd_ascend_patches_if_needed,
    ensure_ascend_runtime_available,
    fail_if_unsupported_npu_afd_features,
    init_ascend_workspace_for_afd,
)
from afd_plugin.v1.worker.ascend.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ensure_ascend_runtime_available()
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUAttentionWorker.init_device",
            expected_role="attention",
            expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD NPU Attention supports only vllm-ascend model runner v1",
            )

        self.device = self._init_device()
        init_ascend_workspace_for_afd(self.device, num_ubatches=1)
        self.model_runner = AFDNPUAttentionModelRunner(
            self.vllm_config,
            self.device,
        )


__all__ = ["AFDNPUAttentionWorker"]
