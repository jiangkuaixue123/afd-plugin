# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU E2E correctness test: DeepSeekV2-Lite with TP=2 ACL graph.

Skipped unless AFD_NPU_E2E_MODEL is set to a local model path.
Requires 4 NPUs (2 attention + 2 FFN).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tests" / "e2e" / "gpu" / "deepseek_v2_lite" / "runner.py"


def _npu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_NPU_E2E_DEVICES", "0,1,2,3").split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_NPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_NPU_E2E_MODEL to run TP=2 ACL graph E2E tests")
    return model


@pytest.mark.gpu
def test_deepseek_v2_tp2_acl_graph_e2e():
    """TP=2, DP=1, ACL graph FULL_DECODE_ONLY correctness test on Ascend NPU."""
    npus = _npu_list()
    if len(npus) < 4:
        pytest.skip(f"requires 4 NPUs; got {len(npus)}")

    model = _model_path()
    capture_size = os.environ.get("AFD_NPU_E2E_CAPTURE_SIZE", "8")

    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-servers",
        "2",
        "--num-ffn-servers",
        "2",
        "--attention-gpus",
        ",".join(npus[:2]),
        "--ffn-gpus",
        ",".join(npus[2:4]),
        "--api-port-base",
        os.environ.get("AFD_NPU_E2E_API_PORT", "18006"),
        "--afd-port",
        os.environ.get("AFD_NPU_E2E_AFD_PORT", "6246"),
        "--tp-size",
        "2",
        "--device-backend",
        "npu",
        "--cuda-graph-full-decode-only",
        "--cudagraph-capture-size",
        capture_size,
        "--max-tokens",
        "7",
        "--prompt",
        "San Francisco is a",
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        "--ffn-start-delay",
        os.environ.get("AFD_NPU_E2E_FFN_START_DELAY", "30"),
        "--common-vllm-arg=--trust-remote-code",
        "--expect-text",
        "city of neighborhoods, and each one",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
