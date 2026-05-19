# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in GPU E2E tests for DeepSeekV2 AFD.

These tests intentionally shell out to the manual runner so the command line
stays close to production usage. They are skipped unless AFD_GPU_E2E_MODEL is
set to a local DeepSeekV2-Lite model path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tests" / "e2e_deepseek_v2_afd.py"


def _gpu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_GPU_E2E_GPUS", "0,1,2,3").split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_GPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_GPU_E2E_MODEL to run DeepSeekV2 AFD GPU E2E tests")
    return model


def _run_e2e(
    *,
    num_attention: int,
    num_ffn: int,
    gpus: list[str],
    api_port_base: int,
    afd_port: int,
    extra_args: list[str] | None = None,
) -> None:
    model = _model_path()
    required_gpus = num_attention + num_ffn
    if len(gpus) < required_gpus:
        pytest.skip(f"requires {required_gpus} GPUs; got {len(gpus)}")

    attention_gpus = ",".join(gpus[:num_attention])
    ffn_gpus = ",".join(gpus[num_attention:required_gpus])
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-servers",
        str(num_attention),
        "--num-ffn-servers",
        str(num_ffn),
        "--attention-gpus",
        attention_gpus,
        "--ffn-gpus",
        ffn_gpus,
        "--api-port-base",
        str(api_port_base),
        "--afd-port",
        str(afd_port),
        "--max-tokens",
        os.environ.get("AFD_GPU_E2E_MAX_TOKENS", "8"),
        "--startup-timeout",
        os.environ.get("AFD_GPU_E2E_STARTUP_TIMEOUT", "900"),
        "--ffn-start-delay",
        os.environ.get("AFD_GPU_E2E_FFN_START_DELAY", "25"),
        "--common-vllm-arg=--trust-remote-code",
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _graph_capture_size() -> int:
    return _env_int("AFD_GPU_E2E_GRAPH_CAPTURE_SIZE", 64)


def _graph_request_count() -> int:
    return _env_int("AFD_GPU_E2E_GRAPH_REQUESTS", _graph_capture_size())


def _graph_args(*, expect_ubatch_replay: bool = False) -> list[str]:
    capture_size = _graph_capture_size()
    requests = (
        _env_int("AFD_GPU_E2E_DBO_REQUESTS", capture_size * 2)
        if expect_ubatch_replay
        else _graph_request_count()
    )
    args = [
        "--cuda-graph-full-decode-only",
        "--use-decode-bench-connector",
        "--expect-ffn-cudagraph-replay",
        "--cudagraph-capture-size",
        str(capture_size),
        "--num-requests",
        str(requests),
        "--request-concurrency",
        str(requests),
        "--max-tokens",
        os.environ.get("AFD_GPU_E2E_GRAPH_MAX_TOKENS", "8"),
    ]
    if expect_ubatch_replay:
        args.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                os.environ.get("AFD_GPU_E2E_DBO_DECODE_THRESHOLD", "1"),
                "--dbo-prefill-token-threshold",
                os.environ.get(
                    "AFD_GPU_E2E_DBO_PREFILL_THRESHOLD",
                    str(capture_size),
                ),
                "--expect-ffn-ubatch-cudagraph-replay",
            ],
        )
    return args


@pytest.mark.gpu
def test_deepseek_v2_eager_1a1f_end_to_end():
    _run_e2e(
        num_attention=1,
        num_ffn=1,
        gpus=_gpu_list(),
        api_port_base=int(os.environ.get("AFD_GPU_E2E_1A1F_API_PORT_BASE", "18000")),
        afd_port=int(os.environ.get("AFD_GPU_E2E_1A1F_AFD_PORT", "6239")),
    )


@pytest.mark.gpu
def test_deepseek_v2_eager_2a2f_end_to_end():
    _run_e2e(
        num_attention=2,
        num_ffn=2,
        gpus=_gpu_list(),
        api_port_base=int(os.environ.get("AFD_GPU_E2E_2A2F_API_PORT_BASE", "18100")),
        afd_port=int(os.environ.get("AFD_GPU_E2E_2A2F_AFD_PORT", "6249")),
    )


@pytest.mark.gpu
def test_deepseek_v2_full_decode_cudagraph_1a1f_end_to_end():
    _run_e2e(
        num_attention=1,
        num_ffn=1,
        gpus=_gpu_list(),
        api_port_base=_env_int("AFD_GPU_E2E_GRAPH_1A1F_API_PORT_BASE", 18200),
        afd_port=_env_int("AFD_GPU_E2E_GRAPH_1A1F_AFD_PORT", 6259),
        extra_args=_graph_args(),
    )


@pytest.mark.gpu
def test_deepseek_v2_full_decode_cudagraph_2a2f_end_to_end():
    _run_e2e(
        num_attention=2,
        num_ffn=2,
        gpus=_gpu_list(),
        api_port_base=_env_int("AFD_GPU_E2E_GRAPH_2A2F_API_PORT_BASE", 18300),
        afd_port=_env_int("AFD_GPU_E2E_GRAPH_2A2F_AFD_PORT", 6269),
        extra_args=_graph_args(),
    )


@pytest.mark.gpu
def test_deepseek_v2_full_decode_cudagraph_2a2f_dbo_replays_ubatch_graph():
    _run_e2e(
        num_attention=2,
        num_ffn=2,
        gpus=_gpu_list(),
        api_port_base=_env_int("AFD_GPU_E2E_DBO_GRAPH_2A2F_API_PORT_BASE", 18400),
        afd_port=_env_int("AFD_GPU_E2E_DBO_GRAPH_2A2F_AFD_PORT", 6279),
        extra_args=_graph_args(expect_ubatch_replay=True),
    )
