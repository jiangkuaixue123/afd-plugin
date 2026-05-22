from __future__ import annotations

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    UBATCH_WRAPPER_FQCN,
    resolve_class_from_qualname,
)


@pytest.mark.parametrize(
    "qualname",
    [
        ATTENTION_WORKER_FQCN,
        ATTENTION_MODEL_RUNNER_FQCN,
        FFN_WORKER_FQCN,
        FFN_MODEL_RUNNER_FQCN,
        UBATCH_WRAPPER_FQCN,
        "afd_plugin.v1.worker:AFDAttentionWorker",
    ],
)
def test_runtime_class_paths_resolve(qualname):
    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker")


def test_phase3_ffn_worker_requires_vllm_when_instantiated_without_runtime():
    cls = resolve_class_from_qualname(FFN_WORKER_FQCN)

    with pytest.raises(RuntimeError, match="requires an importable vLLM runtime"):
        cls()


def test_phase2_attention_worker_requires_vllm_when_instantiated_without_runtime():
    cls = resolve_class_from_qualname(ATTENTION_WORKER_FQCN)

    with pytest.raises(RuntimeError, match="requires an importable vLLM runtime"):
        cls()
