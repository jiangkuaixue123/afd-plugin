# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.compat.ascend.runtime import fix_all2all_backend_for_afd


def _vllm_config(*, enable_sp=False, all2all_backend="allgather_reducescatter"):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
        ),
        parallel_config=SimpleNamespace(
            all2all_backend=all2all_backend,
        ),
    )


def test_fix_all2all_backend_overrides_to_flashinfer_when_sp_disabled():
    config = _vllm_config(enable_sp=False, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_fix_all2all_backend_skips_when_sp_enabled():
    config = _vllm_config(enable_sp=True, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_fix_all2all_backend_skips_when_already_flashinfer():
    config = _vllm_config(enable_sp=False, all2all_backend="flashinfer_all2allv")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"
