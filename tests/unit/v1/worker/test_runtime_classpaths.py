from __future__ import annotations

import importlib.util

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    UBATCH_WRAPPER_FQCN,
    resolve_class_from_qualname,
)

NPU_CLASS_PATHS = [
    NPU_ATTENTION_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
]


def _has_npu_runtime() -> bool:
    return all(
        importlib.util.find_spec(module_name) is not None
        for module_name in ("vllm", "vllm_ascend", "torch_npu")
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


@pytest.mark.parametrize("qualname", NPU_CLASS_PATHS)
def test_npu_runtime_class_paths_resolve_with_runtime(qualname):
    if not _has_npu_runtime():
        pytest.skip("NPU runtime class paths require vLLM, vLLM-Ascend, and torch-npu")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker.ascend")


def test_phase3_ffn_worker_requires_vllm_when_instantiated_without_runtime():
    cls = resolve_class_from_qualname(FFN_WORKER_FQCN)

    with pytest.raises(RuntimeError, match="requires an importable vLLM runtime"):
        cls()


def test_phase2_attention_worker_requires_vllm_when_instantiated_without_runtime():
    cls = resolve_class_from_qualname(ATTENTION_WORKER_FQCN)

    with pytest.raises(RuntimeError, match="requires an importable vLLM runtime"):
        cls()


def test_npu_attention_worker_requires_vllm_ascend_when_instantiated_without_runtime():
    if _has_npu_runtime():
        pytest.skip("runtime is available")

    try:
        cls = resolve_class_from_qualname(NPU_ATTENTION_WORKER_FQCN)
    except (ImportError, ModuleNotFoundError):
        return
    with pytest.raises(RuntimeError, match="vLLM-Ascend runtime"):
        cls()


def test_npu_ffn_worker_requires_vllm_ascend_when_instantiated_without_runtime():
    if _has_npu_runtime():
        pytest.skip("runtime is available")

    try:
        cls = resolve_class_from_qualname(NPU_FFN_WORKER_FQCN)
    except (ImportError, ModuleNotFoundError):
        return
    with pytest.raises(RuntimeError, match="vLLM-Ascend runtime"):
        cls()
