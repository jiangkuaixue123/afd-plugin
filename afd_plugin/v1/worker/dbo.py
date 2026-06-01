# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

from typing import Any

import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker import ubatching

_AFD_DBO_YIELD_OP_REGISTERED = False


def maybe_apply_dbo_yield(
    tensor: Any,
    *,
    role: str,
    ubatching_module: Any | None = None,
) -> Any:
    """Yield to the peer ubatch thread when vLLM DBO is active."""

    dbo_module = ubatching if ubatching_module is None else ubatching_module
    if not bool(dbo_module.dbo_enabled()):
        return tensor

    if ubatching_module is not None:
        dbo_module.dbo_yield()
        return tensor

    if not _AFD_DBO_YIELD_OP_REGISTERED:
        if _torch_is_compiling():
            return tensor
        register_dbo_yield_custom_op()
    if not _AFD_DBO_YIELD_OP_REGISTERED:
        return tensor
    return torch.ops.vllm.manual_dbo_yield(tensor)


def register_dbo_yield_custom_op() -> None:
    global _AFD_DBO_YIELD_OP_REGISTERED

    if _AFD_DBO_YIELD_OP_REGISTERED:
        return

    def afd_manual_dbo_yield_op(x: torch.Tensor) -> torch.Tensor:
        if ubatching.dbo_enabled():
            ubatching.dbo_yield()
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


def _torch_is_compiling() -> bool:
    return bool(torch.compiler.is_compiling())


__all__ = ["maybe_apply_dbo_yield", "register_dbo_yield_custom_op"]
