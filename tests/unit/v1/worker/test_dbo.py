from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.v1.worker import dbo
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


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


def test_maybe_apply_dbo_yield_uses_custom_op_while_compiling(monkeypatch):
    calls = []
    tensor = object()
    yielded = object()

    monkeypatch.setattr(dbo.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        dbo,
        "register_dbo_yield_custom_op",
        lambda: calls.append("register"),
    )
    monkeypatch.setattr(
        dbo.torch.ops.vllm,
        "manual_dbo_yield",
        lambda x: yielded if x is tensor else x,
        raising=False,
    )

    assert maybe_apply_dbo_yield(tensor, role="attention") is yielded
    assert calls == ["register"]

    disabled = SimpleNamespace(
        dbo_enabled=lambda: False,
        dbo_yield=lambda: calls.append("disabled"),
    )
    assert (
        maybe_apply_dbo_yield(tensor, role="attention", ubatching_module=disabled)
        is tensor
    )
    assert calls == ["yield"]
