# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

from typing import Any

_AFD_DBO_YIELD_OP_REGISTERED = False


def maybe_apply_dbo_yield(
    tensor: Any,
    *,
    role: str,
    ubatching_module: Any | None = None,
) -> Any:
    """Yield to the peer ubatch thread when vLLM DBO is active."""

    module_was_provided = ubatching_module is not None
    if module_was_provided:
        dbo_enabled = getattr(ubatching_module, "dbo_enabled", None)
        dbo_yield = getattr(ubatching_module, "dbo_yield", None)
        if not callable(dbo_enabled) or not callable(dbo_yield):
            return tensor
        if not bool(dbo_enabled()):
            return tensor
        dbo_yield()
        return tensor

    try:
        from vllm.v1.worker import ubatching
    except Exception:
        return tensor

    if not _torch_is_compiling():
        if ubatching.dbo_enabled():
            ubatching.dbo_yield()
        return tensor

    if not _AFD_DBO_YIELD_OP_REGISTERED:
        if _torch_is_compiling():
            return tensor
        register_dbo_yield_custom_op()
    try:
        if not _AFD_DBO_YIELD_OP_REGISTERED:
            return tensor
        import torch

        return torch.ops.vllm.manual_dbo_yield(tensor)
    except Exception:
        return tensor


def register_dbo_yield_custom_op() -> None:
    global _AFD_DBO_YIELD_OP_REGISTERED

    if _AFD_DBO_YIELD_OP_REGISTERED:
        return

    import torch
    from vllm.utils.torch_utils import direct_register_custom_op

    def afd_manual_dbo_yield_op(x: torch.Tensor) -> torch.Tensor:
        try:
            from vllm.v1.worker.ubatching import dbo_enabled, dbo_yield
        except Exception:
            return x
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


def _torch_is_compiling() -> bool:
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


__all__ = ["maybe_apply_dbo_yield", "register_dbo_yield_custom_op"]
