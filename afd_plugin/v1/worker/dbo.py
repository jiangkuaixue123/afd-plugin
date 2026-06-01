# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

from typing import Any

import torch

_AFD_DBO_YIELD_OP_REGISTERED = False


def maybe_apply_dbo_yield(
    tensor: Any,
    *,
    role: str,
    ubatching_module: Any | None = None,
) -> Any:
    """Yield to the peer ubatch thread when vLLM DBO is active."""

    if ubatching_module is not None:
        if ubatching_module.dbo_enabled():
            ubatching_module.dbo_yield()
        return tensor

    try:
        from afd_plugin.v1.worker.ascend import ubatching as ascend_ubatching
    except ImportError:
        ascend_ubatching = None

    if ascend_ubatching is not None and ascend_ubatching.dbo_enabled():
        ascend_ubatching.dbo_yield()
        return tensor

    try:
        register_dbo_yield_custom_op()
    except ImportError:
        return tensor

    return torch.ops.vllm.manual_dbo_yield(tensor)


def register_dbo_yield_custom_op() -> None:
    global _AFD_DBO_YIELD_OP_REGISTERED

    if _AFD_DBO_YIELD_OP_REGISTERED:
        return

    from vllm.utils.torch_utils import direct_register_custom_op

    def afd_manual_dbo_yield_op(x: torch.Tensor) -> torch.Tensor:
        from vllm.v1.worker.ubatching import dbo_enabled, dbo_yield

        if dbo_enabled():
            dbo_yield()
        return x

    def afd_manual_dbo_yield_fake(x: torch.Tensor) -> torch.Tensor:
        return x

    try:
        direct_register_custom_op(
            op_name="manual_dbo_yield",
            op_func=afd_manual_dbo_yield_op,
            fake_impl=afd_manual_dbo_yield_fake,
            mutates_args=["x"],
        )
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise
    _AFD_DBO_YIELD_OP_REGISTERED = True


__all__ = ["maybe_apply_dbo_yield", "register_dbo_yield_custom_op"]
