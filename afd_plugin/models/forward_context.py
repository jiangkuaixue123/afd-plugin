# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for plugin-owned model wrappers to read AFD forward metadata."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_afd_metadata_provider: ContextVar[Any | None] = ContextVar(
    "afd_metadata_provider",
    default=None,
)


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
    metadata = additional_kwargs.get("afd_metadata")
    if metadata is not None:
        return metadata

    provider = _afd_metadata_provider.get()
    install = getattr(provider, "_install_afd_metadata_on_forward_context", None)
    if callable(install):
        install(forward_context)
        additional_kwargs = getattr(forward_context, "additional_kwargs", None) or {}
        return additional_kwargs.get("afd_metadata")
    return None


@contextmanager
def use_afd_metadata_provider(provider: Any) -> Iterator[None]:
    token = _afd_metadata_provider.set(provider)
    try:
        yield
    finally:
        _afd_metadata_provider.reset(token)


__all__ = [
    "get_afd_metadata_from_forward_context",
    "use_afd_metadata_provider",
]
