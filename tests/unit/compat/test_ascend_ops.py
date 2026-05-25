from __future__ import annotations

import pytest

from afd_plugin.compat.ascend.ops import (
    ensure_afd_ascend_ops_loaded,
    has_afd_ascend_ops,
)


def test_ascend_ops_loader_fails_clearly_without_extension():
    if has_afd_ascend_ops():
        pytest.skip("AFD Ascend extension is installed in this environment")

    with pytest.raises(RuntimeError, match="AFD Ascend custom ops"):
        ensure_afd_ascend_ops_loaded()
