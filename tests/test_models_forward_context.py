from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.models import get_afd_metadata_from_forward_context
from afd_plugin.runtime.dbo import maybe_apply_dbo_yield


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_maybe_apply_dbo_yield_only_when_dbo_enabled():
    calls = []
    tensor = object()
    module = SimpleNamespace(
        dbo_enabled=lambda: True,
        dbo_yield=lambda: calls.append("yield"),
    )

    assert (
        maybe_apply_dbo_yield(tensor, role="attention", ubatching_module=module)
        is tensor
    )
    assert calls == ["yield"]

    disabled = SimpleNamespace(
        dbo_enabled=lambda: False,
        dbo_yield=lambda: calls.append("disabled"),
    )
    assert (
        maybe_apply_dbo_yield(tensor, role="attention", ubatching_module=disabled)
        is tensor
    )
    assert calls == ["yield"]
