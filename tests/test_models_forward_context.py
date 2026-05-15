from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.models import get_afd_metadata_from_forward_context


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}
