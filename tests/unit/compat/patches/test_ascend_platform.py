from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from afd_plugin.compat.patches.ascend_platform import apply_ascend_platform_patch
from afd_plugin.validation import (
    ATTENTION_WORKER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
)


def _install_fake_vllm_ascend_platform(monkeypatch):
    vllm_ascend_module = types.ModuleType("vllm_ascend")
    platform_module = types.ModuleType("vllm_ascend.platform")

    class NPUPlatform:
        @staticmethod
        def _fix_incompatible_config(vllm_config):
            parallel_config = vllm_config.parallel_config
            vllm_config.fixup_calls.append(
                (parallel_config.enable_dbo, parallel_config.ubatch_size),
            )
            if parallel_config.numa_bind:
                parallel_config.numa_bind = False
                vllm_config.additional_config.setdefault(
                    "enable_cpu_binding",
                    True,
                )
            if parallel_config.enable_dbo:
                parallel_config.enable_dbo = False
            if parallel_config.ubatch_size != 0:
                parallel_config.ubatch_size = 0

    platform_module.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", platform_module)
    return platform_module


def _vllm_config(
    *,
    worker_cls=NPU_ATTENTION_WORKER_FQCN,
    afd_enabled=True,
    connector="camp2pconnector",
    enable_dbo=True,
    ubatch_size=0,
):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": afd_enabled,
                "role": "attention",
                "connector": connector,
            },
        },
        parallel_config=SimpleNamespace(
            worker_cls=worker_cls,
            enable_dbo=enable_dbo,
            ubatch_size=ubatch_size,
            numa_bind=True,
        ),
        fixup_calls=[],
    )


def test_ascend_platform_patch_preserves_native_dbo_for_afd_npu(monkeypatch):
    platform_module = _install_fake_vllm_ascend_platform(monkeypatch)
    apply_ascend_platform_patch()
    apply_ascend_platform_patch()
    config = _vllm_config(worker_cls=NPU_ATTENTION_WORKER_FQCN)

    platform_module.NPUPlatform._fix_incompatible_config(config)

    assert config.fixup_calls == [(False, 0)]
    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config.ubatch_size == 0
    assert config.parallel_config.numa_bind is False
    assert config.additional_config["enable_cpu_binding"] is True


def test_ascend_platform_patch_preserves_native_ubatch_size_for_afd_npu(
    monkeypatch,
):
    platform_module = _install_fake_vllm_ascend_platform(monkeypatch)
    apply_ascend_platform_patch()
    config = _vllm_config(
        worker_cls=NPU_FFN_WORKER_FQCN,
        enable_dbo=False,
        ubatch_size=2,
    )

    platform_module.NPUPlatform._fix_incompatible_config(config)

    assert config.fixup_calls == [(False, 0)]
    assert config.parallel_config.enable_dbo is False
    assert config.parallel_config.ubatch_size == 2


def test_ascend_platform_patch_keeps_non_afd_reset_behavior(monkeypatch):
    platform_module = _install_fake_vllm_ascend_platform(monkeypatch)
    apply_ascend_platform_patch()
    config = _vllm_config(afd_enabled=False, enable_dbo=True, ubatch_size=2)

    platform_module.NPUPlatform._fix_incompatible_config(config)

    assert config.fixup_calls == [(True, 2)]
    assert config.parallel_config.enable_dbo is False
    assert config.parallel_config.ubatch_size == 0


def test_ascend_platform_patch_keeps_non_npu_afd_worker_reset_behavior(
    monkeypatch,
):
    platform_module = _install_fake_vllm_ascend_platform(monkeypatch)
    apply_ascend_platform_patch()
    config = _vllm_config(worker_cls=ATTENTION_WORKER_FQCN, enable_dbo=True)

    platform_module.NPUPlatform._fix_incompatible_config(config)

    assert config.fixup_calls == [(True, 0)]
    assert config.parallel_config.enable_dbo is False
