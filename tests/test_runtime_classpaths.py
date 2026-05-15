from __future__ import annotations

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    resolve_class_from_qualname,
)


@pytest.mark.parametrize(
    "qualname",
    [
        ATTENTION_WORKER_FQCN,
        ATTENTION_MODEL_RUNNER_FQCN,
        FFN_WORKER_FQCN,
        FFN_MODEL_RUNNER_FQCN,
        "afd_plugin.runtime:AFDAttentionWorker",
    ],
)
def test_runtime_class_paths_resolve(qualname):
    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.runtime")


def test_phase1_placeholders_fail_if_instantiated():
    cls = resolve_class_from_qualname(ATTENTION_WORKER_FQCN)

    with pytest.raises(NotImplementedError, match="Phase 1 class-path placeholder"):
        cls()
