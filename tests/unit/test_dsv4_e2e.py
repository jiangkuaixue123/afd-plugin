# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import io
import json
import sys
import threading
import urllib.error
from email.message import Message
from types import SimpleNamespace

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as entrypoint


def _arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_E2E_DEVICES", ",".join(map(str, range(16))))
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/dsv4")
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    command = entrypoint.build_runner_command(tmp_path / "responses.json")
    monkeypatch.setattr(sys, "argv", ["runner", *command[3:]])
    return runner.parse_args()


def test_dsv4_fixed_deployment_and_cleanup(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    runner.configure_scenario(args)
    runner.validate_topology(
        args,
        runner.parse_csv(args.attention_devices),
        runner.parse_csv(args.ffn_devices),
    )
    assert runner.uses_npu_async_process_cleanup(args)
    assert args.gsm8k_output_path is None
    for role, dp, tp in (("attention", "2", "4"), ("ffn", "8", "1")):
        command = runner.build_vllm_command(args, role=role)
        assert command[command.index("--data-parallel-size") + 1] == dp
        assert command[command.index("--tensor-parallel-size") + 1] == tp
        assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
        assert command[command.index("--max-model-len") + 1] == "1048576"
        assert "--enforce-eager" in command
        assert "--enable-expert-parallel" in command
        assert "--enable-dbo" not in command
        assert "--kv-transfer-config" not in command
        config = json.loads(command[command.index("--additional-config") + 1])
        assert config["enable_dsv4_shared_compressor_workspace"] is False
        assert config["enable_cpu_binding"] is True
        assert config["afd"] == {
            "role": role,
            "connector": "CAMAsyncAFDConnector",
            "host": "192.0.2.1",
            "port": 6455,
            "num_attention_ranks": 8,
            "num_ffn_ranks": 8,
            "async": True,
            "compute_gate_on_attention": True,
            "connector_extra_config": {
                "dynamicQuant": 1,
                "attn_ranks_per_dp": 4,
                "async_moe_ubatching": True,
                "async_moe_num_ubatches": 2,
                "async_moe_split": "token",
            },
        }
        env = runner.build_env("0", args, role=role, e2e_run_id="test")
        assert env["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == (
            "1" if role == "attention" else "0"
        )
        assert env[runner.E2E_RUN_ID_ENV] == "test"


def test_dsv4_main_uses_concurrent_requests_and_longer_cleanup(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    cleanup_options = {}
    evaluations = []
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "start_process", lambda *_args: process)
    monkeypatch.setattr(
        runner,
        "stream_output",
        lambda *_args: SimpleNamespace(join=lambda **_kwargs: None),
    )
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "run_concurrent_completion_evaluation",
        lambda _args: evaluations.append("concurrent"),
    )
    monkeypatch.setattr(
        runner,
        "run_gsm8k_evaluation",
        lambda _args: pytest.fail("must not run GSM8K"),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **kwargs: cleanup_options.update(kwargs),
    )
    assert runner.main() == 0
    assert evaluations == ["concurrent"]
    assert cleanup_options["termination_timeout_s"] == 60
    assert (
        cleanup_options["force_kill_environment"][runner.E2E_PROCESS_ROLE_ENV] == "ffn"
    )


@pytest.mark.parametrize("devices", ["0,1,2,3", ",".join(["0"] * 16)])
def test_dsv4_entrypoint_rejects_wrong_devices(monkeypatch, tmp_path, devices):
    _arguments(monkeypatch, tmp_path)
    monkeypatch.setenv("AFD_E2E_DEVICES", devices)
    with pytest.raises(RuntimeError, match="16 unique devices"):
        entrypoint.build_runner_command(tmp_path / "responses.json")


@pytest.mark.parametrize(
    "field", ["common_vllm_arg", "attention_vllm_arg", "ffn_vllm_arg"]
)
def test_dsv4_rejects_deployment_overrides(monkeypatch, tmp_path, field):
    args = _arguments(monkeypatch, tmp_path)
    setattr(args, field, ["--max-num-batched-tokens=1"])
    with pytest.raises(ValueError, match="extra vLLM arguments"):
        runner.configure_scenario(args)


def test_dsv4_rejects_gpu(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    args.device_backend = "gpu"
    runner.configure_scenario(args)
    with pytest.raises(ValueError, match="require NPU"):
        runner.validate_topology(
            args, list(map(str, range(8))), list(map(str, range(8, 16)))
        )


def test_dsv4_environment_requires_cam_and_preserves_network(monkeypatch, tmp_path):
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth-test")
    monkeypatch.setenv("CAM_VENDOR", str(tmp_path))
    with pytest.raises(RuntimeError, match="CAM operator library"):
        entrypoint.build_environment()
    library = tmp_path / "op_api/lib/libopapi.so"
    library.parent.mkdir(parents=True)
    library.touch()
    env = entrypoint.build_environment()
    assert env["HCCL_BUFFSIZE"] == "4096"
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test"
    assert env["TP_SOCKET_IFNAME"] == "eth-test"
    assert env["AFD_FORCE_BALANCED_TOPK_IDS"] == "0"
    assert env["CAM_CUST_OPAPI_LIB_PATH"] == str(library)


@pytest.mark.parametrize(
    "failure", [None, "empty", "truncated", "choices", "http", "json"]
)
def test_ten_requests_overlap_and_validate_every_response(
    monkeypatch,
    tmp_path,
    failure,
):
    args = _arguments(monkeypatch, tmp_path)
    runner.configure_scenario(args)
    # Each fake server request blocks until all ten arrive. Sequential clients
    # cannot pass this test, even if they merely claim concurrency in metadata.
    arrived = threading.Barrier(10)

    def respond(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url.endswith("/v1/chat/completions")
        assert payload["chat_template_kwargs"] == {"thinking": False}
        assert timeout == 300
        operand = int(payload["messages"][0]["content"].split()[1])
        arrived.wait(timeout=5)
        result: dict = {
            "choices": [
                {
                    "message": {"content": str(operand + 7)},
                    "finish_reason": "stop",
                }
            ]
        }
        if operand == 21:
            if failure == "empty":
                result["choices"][0]["message"]["content"] = ""
            elif failure == "truncated":
                result["choices"][0]["finish_reason"] = "length"
            elif failure == "choices":
                result["choices"] = []
            elif failure == "http":
                raise urllib.error.HTTPError(
                    request.full_url, 500, "failed", Message(), io.BytesIO(b"error")
                )
            elif failure == "json":
                return io.BytesIO(b"not-json")
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(runner.urllib.request, "urlopen", respond)
    if failure:
        with pytest.raises((RuntimeError, json.JSONDecodeError)):
            runner.run_concurrent_completion_evaluation(args)
    else:
        runner.run_concurrent_completion_evaluation(args)
        results = json.loads((tmp_path / "responses.json").read_text())
        assert len(results) == 10
        assert [
            row["response"]["choices"][0]["message"]["content"] for row in results
        ] == [str(value) for value in range(19, 29)]
