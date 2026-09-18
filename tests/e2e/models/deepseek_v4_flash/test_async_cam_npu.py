# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local 16-NPU DSV4 Flash async CAM concurrent-request acceptance case."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conftest import run_runner
from tests.e2e.runner import (
    DSV4_ASYNC_CAM_SCENARIO,
    DSV4_ATTENTION_RANKS,
    DSV4_FFN_RANKS,
)

CAM_VENDOR_PATH = Path("/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def build_runner_command(output_path: Path) -> list[str]:
    if _required_env("AFD_E2E_BACKEND") != "npu":
        raise RuntimeError("DSV4 async CAM E2E requires AFD_E2E_BACKEND=npu")
    devices = [device.strip() for device in _required_env("AFD_E2E_DEVICES").split(",")]
    device_count = DSV4_ATTENTION_RANKS + DSV4_FFN_RANKS
    if (
        len(devices) != device_count
        or len(set(devices)) != device_count
        or "" in devices
    ):
        raise RuntimeError(
            f"AFD_E2E_DEVICES must contain {device_count} unique devices"
        )
    return [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        _required_env("AFD_NPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--attention-devices",
        ",".join(devices[:DSV4_ATTENTION_RANKS]),
        "--ffn-devices",
        ",".join(devices[DSV4_ATTENTION_RANKS:]),
        "--scenario",
        DSV4_ASYNC_CAM_SCENARIO,
        "--served-model-name-prefix",
        "dsv4-flash",
        "--afd-host",
        _required_env("HCCL_IF_IP"),
        "--api-port-base",
        os.environ.get("AFD_NPU_DSV4_E2E_API_PORT", "19280"),
        "--afd-port",
        os.environ.get("AFD_NPU_DSV4_E2E_AFD_PORT", "6455"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        "--completion-output-path",
        str(output_path),
    ]


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    interface = _required_env("HCCL_SOCKET_IFNAME")
    _required_env("HCCL_IF_IP")
    vendor = Path(env.get("CAM_VENDOR", str(CAM_VENDOR_PATH)))
    op_api = vendor / "op_api"
    library = op_api / "lib" / "libopapi.so"
    if not library.is_file():
        raise RuntimeError(f"CAM operator library does not exist: {library}")
    env.update(
        {
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            "HCCL_BUFFSIZE": "4096",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_CONNECT_TIMEOUT": "1200",
            "HCCL_EXEC_TIMEOUT": "2000",
            "VLLM_RPC_TIMEOUT": "3600000",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "TASK_QUEUE_ENABLE": "1",
            "AFD_FORCE_BALANCED_TOPK_IDS": "0",
            "AFD_CAM_OP_IO_LOG": "0",
            "AFD_ASYNC_MOE_LAYOUT_LOG": "0",
            "GLOO_SOCKET_IFNAME": interface,
            "TP_SOCKET_IFNAME": interface,
            "CAM_CUST_OPAPI_LIB_PATH": str(library),
        }
    )
    for name, paths in (
        ("ASCEND_CUSTOM_OPP_PATH", (vendor,)),
        ("LD_LIBRARY_PATH", (op_api / "lib", op_api)),
        ("LD_PRELOAD", (library,)),
    ):
        env[name] = os.pathsep.join(
            [
                *(str(path) for path in paths),
                *filter(None, env.get(name, "").split(os.pathsep)),
            ],
        )
    return env


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", [DSV4_ASYNC_CAM_SCENARIO])
def test_deepseek_v4_flash_async_cam(scenario: str, tmp_path: Path) -> None:
    run_runner(
        build_runner_command(tmp_path / f"{scenario}.json"),
        env=build_environment(),
    )
