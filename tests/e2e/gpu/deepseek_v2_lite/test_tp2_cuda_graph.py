# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU E2E correctness test: DeepSeekV2-Lite with TP=2 CUDA graph.

Skipped unless AFD_GPU_E2E_MODEL is set to a local model path.
Requires 4 GPUs (2 attention + 2 FFN).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "tests" / "e2e" / "gpu" / "deepseek_v2_lite" / "runner.py"


def _gpu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_GPU_E2E_GPUS", "0,1,2,3").split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_GPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_GPU_E2E_MODEL to run TP=2 CUDA graph E2E tests")
    return model


@pytest.mark.gpu
def test_deepseek_v2_tp2_cuda_graph_e2e():
    """TP=2, DP=1, CUDA graph FULL_DECODE_ONLY correctness test."""
    import subprocess
    import sys

    gpus = _gpu_list()
    if len(gpus) < 4:
        pytest.skip(f"requires 4 GPUs; got {len(gpus)}")

    model = _model_path()
    capture_size = os.environ.get("AFD_GPU_E2E_GRAPH_CAPTURE_SIZE", "8")

    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-servers",
        "2",
        "--num-ffn-servers",
        "2",
        "--attention-gpus",
        ",".join(gpus[:2]),
        "--ffn-gpus",
        ",".join(gpus[2:4]),
        "--api-port-base",
        os.environ.get("AFD_GPU_E2E_TP2_API_PORT", "18006"),
        "--afd-port",
        os.environ.get("AFD_GPU_E2E_TP2_AFD_PORT", "6246"),
        "--tp-size",
        "2",
        "--cuda-graph-full-decode-only",
        "--cudagraph-capture-size",
        capture_size,
        "--max-tokens",
        "7",
        "--prompt",
        "San Francisco is a",
        "--startup-timeout",
        os.environ.get("AFD_GPU_E2E_STARTUP_TIMEOUT", "900"),
        "--ffn-start-delay",
        os.environ.get("AFD_GPU_E2E_FFN_START_DELAY", "25"),
        "--common-vllm-arg=--trust-remote-code",
        "--expect-text",
        "city of neighborhoods, and each one",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
