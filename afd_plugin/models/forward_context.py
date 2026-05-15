# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for plugin-owned model wrappers to read AFD forward metadata."""

from __future__ import annotations

from typing import Any


def get_afd_metadata_from_forward_context(forward_context: object | None = None) -> Any:
    """Return AFD metadata from vLLM ``ForwardContext.additional_kwargs``.

    The out-of-tree plugin deliberately avoids adding ``ForwardContext`` fields
    in Phase 2. Model wrappers should use this helper instead of reading a
    patched ``forward_ctx.afd_metadata`` attribute.
    """

    if forward_context is None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()

    additional_kwargs = getattr(forward_context, "additional_kwargs", None) or {}
    return additional_kwargs.get("afd_metadata")


__all__ = ["get_afd_metadata_from_forward_context"]
