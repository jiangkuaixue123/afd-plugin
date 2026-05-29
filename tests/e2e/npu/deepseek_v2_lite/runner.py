#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run opt-in DeepSeekV2 AFD NPU end-to-end smoke tests.

The runner starts one FFN-side ``vllm serve`` process and one Attention-side
OpenAI-compatible API process. XAYF topologies are represented as native vLLM
data parallelism: Attention runs with ``DP=X, TP=1`` and FFN runs with
``DP=Y, TP=1``. FULL graph mode always enables the decode-bench KV connector on
the Attention side so replay can be exercised with stable decode shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    args = parse_args()
    attention_devices = parse_csv(args.attention_devices)
    ffn_devices = parse_csv(args.ffn_devices)
    validate_topology(args, attention_devices, ffn_devices)

    processes: list[subprocess.Popen[str]] = []
    log_threads: list[threading.Thread] = []
    logs: dict[str, list[str]] = {"ffn": [], "attention": []}
    log_files = open_log_files(args.log_dir)

    try:
        ffn_cmd = build_vllm_command(args, role="ffn")
        ffn_visible_devices = ",".join(ffn_devices)
        attention_cmd = build_vllm_command(args, role="attention")
        attention_visible_devices = ",".join(attention_devices)

        print_command("FFN", ffn_cmd, ffn_visible_devices)
        print_command("ATTN", attention_cmd, attention_visible_devices)

        ffn_proc, attention_proc = start_processes_together(
            ffn_cmd,
            build_env(ffn_visible_devices),
            attention_cmd,
            build_env(attention_visible_devices),
        )
        processes.extend([ffn_proc, attention_proc])

        log_threads.append(
            stream_output("ffn", ffn_proc, logs["ffn"], log_files.get("ffn")),
        )
        log_threads.append(
            stream_output(
                "attention",
                attention_proc,
                logs["attention"],
                log_files.get("attention"),
            ),
        )

        ensure_alive(ffn_proc, "FFN process exited during startup")
        ensure_alive(attention_proc, "Attention process exited during startup")
        wait_for_openai_api(args, attention_proc)

        responses = request_completions(args)
        if args.full_graph:
            responses.extend(request_completions(args))

        for request_idx, response in enumerate(responses):
            print(f"\n=== Completion response: request {request_idx} ===")
            print(json.dumps(response, ensure_ascii=False, indent=2))

        assert_log_expectations(args, logs)
        return 0
    finally:
        terminate_processes(processes)
        for thread in log_threads:
            thread.join(timeout=2)
        for log_file in log_files.values():
            log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual DeepSeekV2 AFD NPU E2E smoke test.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="DeepSeekV2-Lite model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--vllm-bin",
        default="vllm",
        help="vLLM executable to run. Defaults to 'vllm'.",
    )
    parser.add_argument("--num-attention-servers", type=int, default=1)
    parser.add_argument("--num-ffn-servers", type=int, default=1)
    parser.add_argument(
        "--attention-devices",
        default="1",
        help=(
            "Comma-separated ASCEND_RT_VISIBLE_DEVICES values for Attention. "
            "The count must match --num-attention-servers."
        ),
    )
    parser.add_argument(
        "--ffn-devices",
        default="0",
        help=(
            "Comma-separated ASCEND_RT_VISIBLE_DEVICES values for FFN. "
            "The count must match --num-ffn-servers."
        ),
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port-base", type=int, default=8000)
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument("--afd-port", type=int, default=1239)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--ffn-start-delay",
        type=float,
        default=0,
        help="Deprecated compatibility option; FFN and Attention now start together.",
    )
    parser.add_argument("--prompt", default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help=(
            "Number of completion requests per round. Defaults to Attention DP "
            "size for eager, or --graph-capture-size for FULL graph."
        ),
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent completion requests. Defaults to --num-requests.",
    )
    parser.add_argument(
        "--served-model-name-prefix",
        default="deepseek-v2-lite-afd-npu",
        help="Prefix used for role-specific served model names.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory for this case's engine logs. When set, writes "
            "ffn.log and attention.log under this directory."
        ),
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Run without --enforce-eager and set cudagraph_mode=FULL.",
    )
    parser.add_argument(
        "--graph-capture-size",
        type=int,
        default=12,
        help="Capture size used for max-num-seqs and max-num-batched-tokens.",
    )
    parser.add_argument(
        "--enable-ubatching",
        action="store_true",
        help="Enable vLLM DBO/ubatching and AFD NPU two-ubatch metadata.",
    )
    parser.add_argument(
        "--dbo-decode-token-threshold",
        type=int,
        default=2,
        help="Value passed to --dbo-decode-token-threshold when ubatching is enabled.",
    )
    parser.add_argument(
        "--dbo-prefill-token-threshold",
        type=int,
        default=None,
        help=(
            "Value passed to --dbo-prefill-token-threshold when ubatching is "
            "enabled. Defaults to --graph-capture-size."
        ),
    )
    parser.add_argument(
        "--common-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added to all processes.",
    )
    parser.add_argument(
        "--attention-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to Attention.",
    )
    parser.add_argument(
        "--ffn-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to FFN.",
    )
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_topology(
    args: argparse.Namespace,
    attention_devices: list[str],
    ffn_devices: list[str],
) -> None:
    if args.num_attention_servers != len(attention_devices):
        raise ValueError(
            "--num-attention-servers must match the number of --attention-devices",
        )
    if args.num_ffn_servers != len(ffn_devices):
        raise ValueError("--num-ffn-servers must match the number of --ffn-devices")
    if args.num_attention_servers < 1 or args.num_ffn_servers < 1:
        raise ValueError("AFD NPU E2E requires at least one Attention and FFN server")


def build_vllm_command(args: argparse.Namespace, *, role: str) -> list[str]:
    role_dp_size = (
        args.num_attention_servers if role == "attention" else args.num_ffn_servers
    )
    afd_config = {
        "afd": {
            "enabled": True,
            "role": role,
            "connector": "camp2pconnector",
            "host": args.afd_host,
            "port": args.afd_port,
            "num_attention_servers": args.num_attention_servers,
            "num_ffn_servers": args.num_ffn_servers,
            "extra_config": {
                "afd_size": f"{args.num_attention_servers}A{args.num_ffn_servers}F",
            },
        },
    }
    if args.enable_ubatching:
        afd_config["afd"]["extra_config"].update(
            {
                "enable_ubatching": True,
                "num_ubatches": 2,
            },
        )

    worker_cls = (
        "afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker"
        if role == "attention"
        else "afd_plugin.v1.worker.ascend.AFDNPUFFNWorker"
    )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--worker-cls",
        worker_cls,
        "--served-model-name",
        served_model_name(args, role),
        "--data-parallel-size",
        str(role_dp_size),
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--additional-config",
        json.dumps(afd_config, separators=(",", ":")),
    ]

    if args.full_graph:
        capture_size = str(args.graph_capture_size)
        cmd.extend(
            [
                "--max-num-seqs",
                capture_size,
                "--max-num-batched-tokens",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {
                        "cudagraph_mode": "FULL",
                        "cudagraph_capture_sizes": [args.graph_capture_size],
                    },
                    separators=(",", ":"),
                ),
            ],
        )
    else:
        cmd.append("--enforce-eager")

    if args.enable_ubatching:
        prefill_threshold = (
            args.dbo_prefill_token_threshold
            if args.dbo_prefill_token_threshold is not None
            else args.graph_capture_size
        )
        cmd.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                str(args.dbo_decode_token_threshold),
                "--dbo-prefill-token-threshold",
                str(prefill_threshold),
            ],
        )

    if role == "attention":
        cmd.extend(
            ["--host", args.api_host, "--port", str(attention_api_port(args))],
        )
        if args.full_graph:
            cmd.extend(["--kv-transfer-config", decode_bench_connector_config()])
        cmd.extend(args.attention_vllm_arg)
    else:
        cmd.extend(["--host", args.api_host, "--port", str(ffn_api_port(args))])
        cmd.extend(args.ffn_vllm_arg)

    cmd.extend(args.common_vllm_arg)
    return cmd


def decode_bench_connector_config() -> str:
    return json.dumps(
        {
            "kv_connector": "AFDDecodeBenchConnector",
            "kv_connector_module_path": "afd_plugin.connectors.decode_bench",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0,
            },
        },
        separators=(",", ":"),
    )


def served_model_name(args: argparse.Namespace, role: str) -> str:
    return f"{args.served_model_name_prefix}-{role}"


def attention_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base


def ffn_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base + 1


def build_env(visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    env["VLLM_USE_V1"] = "1"
    env["VLLM_PLUGINS"] = "ascend,afd"
    env["PYTHONUNBUFFERED"] = "1"
    current_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [
        str(path)
        for path in (
            REPO_ROOT.parent / "vllm",
            REPO_ROOT,
            REPO_ROOT.parent / "vllm-ascend",
        )
        if path.exists()
    ]
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def start_process(
    command: list[str],
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def start_processes_together(
    ffn_command: list[str],
    ffn_env: dict[str, str],
    attention_command: list[str],
    attention_env: dict[str, str],
) -> tuple[subprocess.Popen[str], subprocess.Popen[str]]:
    futures_by_role = {}
    processes_by_role: dict[str, subprocess.Popen[str]] = {}
    start_error: BaseException | None = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures_by_role[executor.submit(start_process, ffn_command, ffn_env)] = "ffn"
        futures_by_role[
            executor.submit(start_process, attention_command, attention_env)
        ] = "attention"

        for future in as_completed(futures_by_role):
            role = futures_by_role[future]
            try:
                processes_by_role[role] = future.result()
            except BaseException as exc:
                start_error = exc

    if start_error is not None:
        terminate_processes(list(processes_by_role.values()))
        raise start_error

    return processes_by_role["ffn"], processes_by_role["attention"]


def open_log_files(log_dir: str | None) -> dict[str, Any]:
    if not log_dir:
        return {}

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "ffn": (directory / "ffn.log").open("w", encoding="utf-8"),
        "attention": (directory / "attention.log").open("w", encoding="utf-8"),
    }


def stream_output(
    name: str,
    process: subprocess.Popen[str],
    log_lines: list[str],
    log_file: Any | None = None,
) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            log_lines.append(line)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
            print(f"[{name}] {line}", end="")

    thread = threading.Thread(target=worker, name=f"{name}-log-stream", daemon=True)
    thread.start()
    return thread


def wait_for_openai_api(
    args: argparse.Namespace,
    attention_process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + args.startup_timeout
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/models"
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        ensure_alive(attention_process, "Attention process exited before API ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"\nAttention API is ready at {url}")
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for Attention API at {url}; last error={last_error!r}",
    )


def request_completion(args: argparse.Namespace) -> dict[str, Any]:
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/completions"
    payload = {
        "model": served_model_name(args, "attention"),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = read_http_error_body(exc)
        raise RuntimeError(
            f"Completion request to {url} failed with HTTP {exc.code} "
            f"{exc.reason}: {body}",
        ) from exc
    return json.loads(body)


def read_http_error_body(error: urllib.error.HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    return body if body else "<empty response body>"


def request_completions(args: argparse.Namespace) -> list[dict[str, Any]]:
    request_count = _request_count(args)
    if request_count == 1:
        return [request_completion(args)]

    responses: list[dict[str, Any] | None] = [None] * request_count
    failures: list[tuple[int, BaseException]] = []
    concurrency = (
        int(args.request_concurrency)
        if args.request_concurrency is not None
        else request_count
    )
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = {
            executor.submit(request_completion, args): request_idx
            for request_idx in range(request_count)
        }
        for future in as_completed(futures):
            request_idx = futures[future]
            try:
                responses[request_idx] = future.result()
            except Exception as exc:
                failures.append((request_idx, exc))
                print(
                    f"Completion request {request_idx} failed: {exc!r}",
                    file=sys.stderr,
                )

    if failures:
        details = ", ".join(
            f"{request_idx}: {exc!r}" for request_idx, exc in failures
        )
        raise RuntimeError(f"{len(failures)} completion request(s) failed: {details}")

    completed_responses = [response for response in responses if response is not None]
    if not completed_responses:
        raise RuntimeError("All completion requests failed")
    return completed_responses


def _request_count(args: argparse.Namespace) -> int:
    if args.num_requests is not None:
        return int(args.num_requests)
    if args.full_graph or args.enable_ubatching:
        return int(args.graph_capture_size)
    return max(int(args.num_attention_servers), 1)


def assert_log_expectations(
    args: argparse.Namespace,
    logs: dict[str, list[str]],
) -> None:
    ffn_log = "".join(logs["ffn"])
    attention_log = "".join(logs["attention"])
    if args.enable_ubatching:
        _require_log(
            attention_log,
            "AFD_NPU_E2E: Attention sending 2 ubatch DP metadata slices",
            "Attention did not report two-way ubatch metadata splitting",
        )
        _require_log(
            ffn_log,
            "AFD_NPU_E2E: FFN processing 2 ubatch stages",
            "FFN did not report processing two ubatch stages",
        )
    if args.full_graph:
        _require_log(
            ffn_log,
            "AFD_NPU_E2E: FFN ACL graph captured",
            "FFN ACL graph capture was not observed",
        )
        _require_log(
            ffn_log,
            "AFD_NPU_E2E: FFN ACL graph replayed",
            "FFN ACL graph replay was not observed",
        )


def _require_log(log: str, needle: str, message: str) -> None:
    if needle not in log:
        raise AssertionError(message)


def ensure_alive(process: subprocess.Popen[str], message: str) -> None:
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(f"{message} (returncode={returncode})")


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    for process in reversed(processes):
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def print_command(name: str, command: list[str], visible_devices: str) -> None:
    printable = " ".join(shell_quote(token) for token in command)
    print(f"\n=== Starting {name} (ASCEND_RT_VISIBLE_DEVICES={visible_devices}) ===")
    print(printable)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "@%_+=:,./-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
