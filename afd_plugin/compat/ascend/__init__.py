# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend/vLLM-Ascend compatibility helpers for AFD runtime classes."""

from afd_plugin.compat.ascend.runtime import (
    apply_afd_ascend_patches_if_needed,
    ascend_forward_context,
    ensure_ascend_runtime_available,
    ensure_vllm_config_has_afd_proxy,
    fail_if_unsupported_npu_afd_features,
    init_ascend_workspace_for_afd,
    mirror_afd_metadata_on_forward_context,
)

__all__ = [
    "apply_afd_ascend_patches_if_needed",
    "ascend_forward_context",
    "ensure_ascend_runtime_available",
    "ensure_vllm_config_has_afd_proxy",
    "fail_if_unsupported_npu_afd_features",
    "init_ascend_workspace_for_afd",
    "mirror_afd_metadata_on_forward_context",
]
