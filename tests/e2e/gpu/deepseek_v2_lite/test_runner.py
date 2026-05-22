from __future__ import annotations

import argparse
import json

from tests.e2e.gpu.deepseek_v2_lite import runner
from tests.e2e.gpu.deepseek_v2_lite.runner import build_vllm_command


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        num_attention_servers=2,
        num_ffn_servers=2,
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="127.0.0.1",
        afd_port=6249,
        served_model_name_prefix="deepseek-v2-lite-afd",
        num_requests=None,
        request_concurrency=None,
        cuda_graph_full_decode_only=False,
        cudagraph_capture_size=64,
        enable_dbo=False,
        dbo_decode_token_threshold=1,
        dbo_prefill_token_threshold=None,
        use_decode_bench_connector=False,
        common_vllm_arg=["--trust-remote-code"],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
    )


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_runner_uses_native_dp_for_attention_topology():
    command = build_vllm_command(_args(), role="attention")

    assert command.count("serve") == 1
    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _arg_value(command, "--worker-cls") == (
        "afd_plugin.v1.worker.AFDAttentionWorker"
    )
    assert _arg_value(command, "--port") == "18100"
    assert _arg_value(command, "--served-model-name") == (
        "deepseek-v2-lite-afd-attention"
    )

    additional_config = json.loads(_arg_value(command, "--additional-config"))
    assert additional_config["afd"]["num_attention_servers"] == 2
    assert additional_config["afd"]["num_ffn_servers"] == 2
    assert additional_config["afd"]["extra_config"]["afd_size"] == "2A2F"
    assert "afd_server_rank" not in additional_config["afd"]


def test_runner_uses_native_dp_for_ffn_topology():
    command = build_vllm_command(_args(), role="ffn")

    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _arg_value(command, "--worker-cls") == "afd_plugin.v1.worker.AFDFFNWorker"
    assert _arg_value(command, "--port") == "18101"
    assert _arg_value(command, "--served-model-name") == "deepseek-v2-lite-afd-ffn"


def test_runner_sends_one_concurrent_request_per_attention_dp_rank(monkeypatch):
    args = _args()
    calls = []

    def fake_request_completion(_args):
        calls.append(_args)
        return {"id": len(calls)}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    responses = runner.request_completions(args)

    assert len(calls) == 2
    assert len(responses) == 2
