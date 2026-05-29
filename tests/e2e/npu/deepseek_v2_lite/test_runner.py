from __future__ import annotations

import argparse
import io
import json
import urllib.error

import pytest

from tests.e2e.npu.deepseek_v2_lite import runner
from tests.e2e.npu.deepseek_v2_lite.runner import build_vllm_command


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        num_attention_servers=2,
        num_ffn_servers=1,
        api_host="127.0.0.1",
        api_port_base=19100,
        afd_host="127.0.0.1",
        afd_port=6349,
        startup_timeout=900,
        ffn_start_delay=25,
        log_dir=None,
        served_model_name_prefix="deepseek-v2-lite-afd-npu",
        prompt="San Francisco is a",
        max_tokens=16,
        temperature=0.0,
        num_requests=None,
        request_concurrency=None,
        full_graph=False,
        graph_capture_size=12,
        enable_ubatching=False,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=None,
        common_vllm_arg=["--trust-remote-code"],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
    )


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_npu_runner_uses_native_dp_for_attention_topology():
    command = build_vllm_command(_args(), role="attention")

    assert command.count("serve") == 1
    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _arg_value(command, "--worker-cls") == (
        "afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker"
    )
    assert _arg_value(command, "--port") == "19100"
    assert _arg_value(command, "--served-model-name") == (
        "deepseek-v2-lite-afd-npu-attention"
    )

    additional_config = json.loads(_arg_value(command, "--additional-config"))
    assert additional_config["afd"]["connector"] == "camp2pconnector"
    assert additional_config["afd"]["num_attention_servers"] == 2
    assert additional_config["afd"]["num_ffn_servers"] == 1
    assert additional_config["afd"]["extra_config"]["afd_size"] == "2A1F"


def test_npu_runner_uses_native_dp_for_ffn_topology():
    command = build_vllm_command(_args(), role="ffn")

    assert _arg_value(command, "--data-parallel-size") == "1"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _arg_value(command, "--worker-cls") == (
        "afd_plugin.v1.worker.ascend.AFDNPUFFNWorker"
    )
    assert _arg_value(command, "--port") == "19101"
    assert _arg_value(command, "--served-model-name") == (
        "deepseek-v2-lite-afd-npu-ffn"
    )


def test_npu_runner_full_graph_uses_decode_bench_connector_for_attention():
    args = _args()
    args.full_graph = True

    command = build_vllm_command(args, role="attention")
    kv_transfer_config = json.loads(_arg_value(command, "--kv-transfer-config"))
    compilation_config = json.loads(_arg_value(command, "--compilation-config"))

    assert "--enforce-eager" not in command
    assert _arg_value(command, "--max-num-seqs") == "12"
    assert _arg_value(command, "--max-num-batched-tokens") == "12"
    assert compilation_config == {
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [12],
    }
    assert kv_transfer_config["kv_connector"] == "AFDDecodeBenchConnector"
    assert kv_transfer_config["kv_connector_module_path"] == (
        "afd_plugin.connectors.decode_bench"
    )


def test_npu_runner_full_graph_does_not_attach_kv_connector_to_ffn():
    args = _args()
    args.full_graph = True

    command = build_vllm_command(args, role="ffn")

    assert "--kv-transfer-config" not in command
    assert "--enforce-eager" not in command


def test_npu_runner_ubatching_sets_cli_and_afd_extra_config():
    args = _args()
    args.enable_ubatching = True

    command = build_vllm_command(args, role="attention")
    additional_config = json.loads(_arg_value(command, "--additional-config"))

    assert "--enable-dbo" in command
    assert _arg_value(command, "--ubatch-size") == "2"
    assert _arg_value(command, "--dbo-decode-token-threshold") == "2"
    assert _arg_value(command, "--dbo-prefill-token-threshold") == "2"
    assert additional_config["afd"]["extra_config"]["enable_ubatching"] is True
    assert additional_config["afd"]["extra_config"]["num_ubatches"] == 2


def test_npu_runner_sends_graph_capture_size_requests_by_default(monkeypatch):
    args = _args()
    args.full_graph = True
    calls = []

    def fake_request_completion(_args):
        calls.append(_args)
        return {"id": len(calls)}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    responses = runner.request_completions(args)

    assert len(calls) == 12
    assert len(responses) == 12


def test_npu_request_completion_includes_http_error_body(monkeypatch):
    args = _args()
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:19100/v1/completions",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"NPU out of memory"}'),
    )

    def fake_urlopen(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="NPU out of memory"):
        runner.request_completion(args)


def test_npu_wait_for_openai_api_fails_when_attention_exits():
    process = argparse.Namespace(poll=lambda: 3)

    with pytest.raises(RuntimeError, match="Attention process exited"):
        runner.wait_for_openai_api(_args(), process)


def test_npu_runner_log_expectations_require_graph_and_ubatch_markers():
    args = _args()
    args.full_graph = True
    args.enable_ubatching = True

    runner.assert_log_expectations(
        args,
        {
            "attention": [
                "AFD_NPU_E2E: Attention sending 2 ubatch DP metadata slices",
                "AFDDecodeBenchConnector",
                "Graph capturing finished",
            ],
            "ffn": [
                "AFD_NPU_E2E: FFN processing 2 ubatch stages",
            ],
        },
        [{"id": request_idx} for request_idx in range(24)],
    )


def test_npu_runner_env_keeps_ascend_platform_plugin_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path / "afd-plugin")
    (tmp_path / "afd-plugin").mkdir()
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm-ascend").mkdir()
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    env = runner.build_env("4,5")

    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert env["VLLM_USE_V1"] == "1"
    assert env["VLLM_PLUGINS"] == "ascend,afd"
    assert env["PYTHONPATH"].split(":") == [
        str(tmp_path / "afd-plugin"),
        "/existing/pythonpath",
    ]


def test_npu_runner_streams_engine_output_to_role_log_files(tmp_path):
    log_files = runner.open_log_files(str(tmp_path))
    try:
        process = argparse.Namespace(stdout=io.StringIO("engine line\n"))
        log_lines: list[str] = []

        thread = runner.stream_output("ffn", process, log_lines, log_files["ffn"])
        thread.join(timeout=2)
    finally:
        for log_file in log_files.values():
            log_file.close()

    assert log_lines == ["engine line\n"]
    assert (tmp_path / "ffn.log").read_text(encoding="utf-8") == "engine line\n"
    assert (tmp_path / "attention.log").read_text(encoding="utf-8") == ""


def test_npu_runner_starts_attention_and_ffn_before_startup_checks(monkeypatch):
    args = _args()
    args.num_attention_servers = 1
    args.attention_devices = "1"
    args.ffn_devices = "0"
    started_roles: list[str] = []

    def fake_build_vllm_command(_args, *, role):
        return [role]

    def fake_start_process(command, env):
        del env
        started_roles.append(command[0])
        return argparse.Namespace(stdout=io.StringIO(""), poll=lambda: None)

    def fake_ensure_alive(_process, _message):
        assert set(started_roles) == {"ffn", "attention"}
        assert len(started_roles) == 2

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "build_vllm_command", fake_build_vllm_command)
    monkeypatch.setattr(runner, "build_env", lambda visible_devices: {})
    monkeypatch.setattr(runner, "print_command", lambda *args: None)
    monkeypatch.setattr(runner, "start_process", fake_start_process)
    monkeypatch.setattr(runner, "ensure_alive", fake_ensure_alive)
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *args: None)
    monkeypatch.setattr(runner, "request_completions", lambda _args: [{"id": "ok"}])
    monkeypatch.setattr(runner, "assert_log_expectations", lambda *args: None)
    monkeypatch.setattr(runner, "terminate_processes", lambda processes: None)

    assert runner.main() == 0
    assert set(started_roles) == {"ffn", "attention"}
