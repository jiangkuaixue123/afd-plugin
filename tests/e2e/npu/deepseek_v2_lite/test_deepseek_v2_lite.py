# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in NPU E2E tests for DeepSeekV2 AFD.

These tests shell out to the manual runner so the command line stays close to
production usage. They are skipped unless AFD_NPU_E2E_MODEL is set to a local
DeepSeekV2-Lite model path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "tests" / "e2e" / "npu" / "deepseek_v2_lite" / "runner.py"


@dataclass(frozen=True)
class NPUE2ECase:
    case_id: str
    num_attention: int
    num_ffn: int
    enable_ubatching: bool
    full_graph: bool
    api_port_base: int
    afd_port: int

    @property
    def topology(self) -> str:
        return f"{self.num_attention}A{self.num_ffn}F"

    @property
    def pytest_id(self) -> str:
        ubatch = "ubatch" if self.enable_ubatching else "no-ubatch"
        mode = "full-graph" if self.full_graph else "eager"
        return f"{self.case_id}-{self.topology}-{ubatch}-{mode}"


NPU_E2E_CASES = [
    NPUE2ECase("NPU-E2E-001", 1, 1, False, False, 19000, 6339),
    NPUE2ECase("NPU-E2E-002", 1, 1, True, False, 19020, 6340),
    NPUE2ECase("NPU-E2E-003", 1, 1, False, True, 19040, 6341),
    NPUE2ECase("NPU-E2E-004", 1, 1, True, True, 19060, 6342),
    NPUE2ECase("NPU-E2E-005", 2, 2, False, False, 19080, 6343),
    NPUE2ECase("NPU-E2E-006", 2, 2, True, False, 19100, 6344),
    NPUE2ECase("NPU-E2E-007", 2, 2, False, True, 19120, 6345),
    NPUE2ECase("NPU-E2E-008", 2, 2, True, True, 19140, 6346),
    NPUE2ECase("NPU-E2E-009", 2, 1, False, False, 19160, 6347),
    NPUE2ECase("NPU-E2E-010", 2, 1, True, False, 19180, 6348),
    NPUE2ECase("NPU-E2E-011", 2, 1, False, True, 19200, 6349),
    NPUE2ECase("NPU-E2E-012", 2, 1, True, True, 19220, 6350),
]


def _npu_device_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_NPU_E2E_DEVICES", "0,1,2,3").split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_NPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_NPU_E2E_MODEL to run DeepSeekV2 AFD NPU E2E tests")
    return model


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _graph_capture_size() -> int:
    return _env_int("AFD_NPU_E2E_GRAPH_CAPTURE_SIZE", 12)


def _log_dir(case: NPUE2ECase) -> Path:
    root = Path(os.environ.get("AFD_NPU_E2E_LOG_DIR", "/tmp/afd_npu_e2e_logs"))
    return root / case.pytest_id


def _run_e2e(case: NPUE2ECase, devices: list[str]) -> None:
    model = _model_path()
    required_devices = case.num_attention + case.num_ffn
    if len(devices) < required_devices:
        pytest.skip(f"requires {required_devices} NPUs; got {len(devices)}")

    ffn_devices = ",".join(devices[: case.num_ffn])
    attention_devices = ",".join(devices[case.num_ffn : required_devices])
    capture_size = _graph_capture_size()
    env_case_id = case.case_id.replace("-", "_")
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-servers",
        str(case.num_attention),
        "--num-ffn-servers",
        str(case.num_ffn),
        "--attention-devices",
        attention_devices,
        "--ffn-devices",
        ffn_devices,
        "--api-port-base",
        str(_env_int(f"AFD_NPU_E2E_{env_case_id}_API_PORT_BASE", case.api_port_base)),
        "--afd-port",
        str(_env_int(f"AFD_NPU_E2E_{env_case_id}_AFD_PORT", case.afd_port)),
        "--graph-capture-size",
        str(capture_size),
        "--max-tokens",
        os.environ.get("AFD_NPU_E2E_MAX_TOKENS", "8"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        "--log-dir",
        str(_log_dir(case)),
        "--common-vllm-arg=--trust-remote-code",
    ]
    if case.full_graph:
        command.append("--full-graph")
    if case.enable_ubatching:
        command.extend(
            [
                "--enable-ubatching",
                "--dbo-decode-token-threshold",
                os.environ.get("AFD_NPU_E2E_DBO_DECODE_THRESHOLD", "2"),
                "--dbo-prefill-token-threshold",
                os.environ.get("AFD_NPU_E2E_DBO_PREFILL_THRESHOLD", "2"),
            ],
        )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def test_run_e2e_passes_case_log_dir_to_runner(monkeypatch, tmp_path):
    calls = []
    case = NPU_E2E_CASES[0]
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/DeepSeek-V2-Lite")
    monkeypatch.setenv("AFD_NPU_E2E_LOG_DIR", str(tmp_path))

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(subprocess, "run", fake_run)

    _run_e2e(case, ["0", "1"])

    command, cwd, check = calls[0]
    assert cwd == REPO_ROOT
    assert check is True
    assert command[command.index("--log-dir") + 1] == str(tmp_path / case.pytest_id)


@pytest.mark.npu
@pytest.mark.parametrize(
    "case",
    NPU_E2E_CASES,
    ids=[case.pytest_id for case in NPU_E2E_CASES],
)
def test_deepseek_v2_npu_end_to_end_matrix(case: NPUE2ECase):
    _run_e2e(case, _npu_device_list())
