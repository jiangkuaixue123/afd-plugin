from __future__ import annotations

import builtins

import pytest

pytest.importorskip("torch")

from afd_plugin.v1.worker import dbo
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


def test_maybe_apply_dbo_yield_uses_custom_op(monkeypatch):
    calls = []
    tensor = object()
    yielded = object()

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


def test_maybe_apply_dbo_yield_does_not_probe_ascend(monkeypatch):
    tensor = object()

    real_import = builtins.__import__

    def fail_on_ascend_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("afd_plugin.v1.worker.ascend"):
            pytest.fail(f"unexpected Ascend import from DBO yield helper: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_on_ascend_import)
    monkeypatch.setattr(
        dbo,
        "register_dbo_yield_custom_op",
        lambda: (_ for _ in ()).throw(ImportError),
    )

    assert maybe_apply_dbo_yield(tensor, role="attention") is tensor
