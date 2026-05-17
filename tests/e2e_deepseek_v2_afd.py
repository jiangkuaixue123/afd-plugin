#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run manual DeepSeekV2 AFD end-to-end smoke tests.

This script is intentionally opt-in and is not collected by pytest. It starts
one FFN-side ``vllm serve`` process and one Attention-side ``vllm serve``
OpenAI-compatible API process. XAYF topologies are represented as native vLLM
data parallelism: Attention runs with ``DP=X, TP=1`` and FFN runs with
``DP=Y, TP=1``.

Current Phase 4 limitations are enforced by the command line: full weights are
loaded on all sides, and ``--enforce-eager`` is always passed because CUDA graph
support is not implemented yet.
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

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    attention_gpus = parse_csv(args.attention_gpus)
    ffn_gpus = parse_csv(args.ffn_gpus)
    validate_topology(args, attention_gpus, ffn_gpus)

    processes: list[subprocess.Popen[str]] = []
    log_threads: list[threading.Thread] = []

    try:
        ffn_cmd = build_vllm_command(args, role="ffn")
        ffn_cuda_visible_devices = ",".join(ffn_gpus)
        print_command("FFN", ffn_cmd, ffn_cuda_visible_devices)
        ffn_proc = start_process("ffn", ffn_cmd, build_env(ffn_cuda_visible_devices))
        processes.append(ffn_proc)
        log_threads.append(stream_output("ffn", ffn_proc))

        time.sleep(args.ffn_start_delay)
        ensure_alive(ffn_proc, "FFN process exited during startup")

        attention_cmd = build_vllm_command(args, role="attention")
        attention_cuda_visible_devices = ",".join(attention_gpus)
        print_command("ATTN", attention_cmd, attention_cuda_visible_devices)
        attention_proc = start_process(
            "attention",
            attention_cmd,
            build_env(attention_cuda_visible_devices),
        )
        processes.append(attention_proc)
        log_threads.append(stream_output("attention", attention_proc))

        ensure_alive(attention_proc, "Attention process exited during startup")
        wait_for_openai_api(args)

        responses = request_completions(args)
        for request_idx, response in enumerate(responses):
            print(f"\n=== Completion response: request {request_idx} ===")
            print(json.dumps(response, ensure_ascii=False, indent=2))

        return 0
    finally:
        terminate_processes(processes)
        for thread in log_threads:
            thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual DeepSeekV2 AFD E2E smoke test.",
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
        "--attention-gpus",
        default="0",
        help=(
            "Comma-separated CUDA_VISIBLE_DEVICES values for the Attention "
            "serve process. The number of GPUs must match Attention DP size."
        ),
    )
    parser.add_argument(
        "--ffn-gpus",
        default="1",
        help=(
            "Comma-separated CUDA_VISIBLE_DEVICES values for the FFN serve "
            "process. The number of GPUs must match FFN DP size."
        ),
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port-base", type=int, default=8000)
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument("--afd-port", type=int, default=1239)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--ffn-start-delay", type=float, default=8)
    parser.add_argument("--prompt", default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--served-model-name-prefix",
        default="deepseek-v2-lite-afd",
        help="Prefix used for role-specific served model names.",
    )
    parser.add_argument(
        "--ffn-headless",
        action="store_true",
        help="Run FFN servers with --headless. Not required by the current patch.",
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
        help="Extra single-token vLLM arg added only to Attention processes.",
    )
    parser.add_argument(
        "--ffn-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to FFN processes.",
    )
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_topology(
    args: argparse.Namespace,
    attention_gpus: list[str],
    ffn_gpus: list[str],
) -> None:
    if args.num_attention_servers != len(attention_gpus):
        raise ValueError(
            "--num-attention-servers must match the number of --attention-gpus",
        )
    if args.num_ffn_servers != len(ffn_gpus):
        raise ValueError("--num-ffn-servers must match the number of --ffn-gpus")
    if args.num_attention_servers < 1 or args.num_ffn_servers < 1:
        raise ValueError("AFD E2E requires at least one Attention and FFN server")


def build_vllm_command(
    args: argparse.Namespace,
    *,
    role: str,
) -> list[str]:
    role_dp_size = (
        args.num_attention_servers if role == "attention" else args.num_ffn_servers
    )
    afd_config = {
        "afd": {
            "enabled": True,
            "role": role,
            "connector": "p2pconnector",
            "host": args.afd_host,
            "port": args.afd_port,
            "num_attention_servers": args.num_attention_servers,
            "num_ffn_servers": args.num_ffn_servers,
            "extra_config": {
                "afd_size": f"{args.num_attention_servers}A{args.num_ffn_servers}F",
            },
        },
    }
    worker_cls = (
        "afd_plugin.runtime.AFDAttentionWorker"
        if role == "attention"
        else "afd_plugin.runtime.AFDFFNWorker"
    )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--worker-cls",
        worker_cls,
        "--enforce-eager",
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
    if role == "attention":
        cmd.extend(
            ["--host", args.api_host, "--port", str(attention_api_port(args))],
        )
        cmd.extend(args.attention_vllm_arg)
    else:
        if args.ffn_headless:
            cmd.append("--headless")
        else:
            cmd.extend(
                ["--host", args.api_host, "--port", str(ffn_api_port(args))],
            )
        cmd.extend(args.ffn_vllm_arg)
    cmd.extend(args.common_vllm_arg)
    return cmd


def served_model_name(args: argparse.Namespace, role: str) -> str:
    return f"{args.served_model_name_prefix}-{role}"


def attention_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base


def ffn_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base + 1


def build_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["VLLM_PLUGINS"] = "afd"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("AFD_PLUGIN_EARLY_ENGINE_PATCH", None)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not current_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{current_pythonpath}"
    )
    return env


def start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
) -> subprocess.Popen[str]:
    del name
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


def stream_output(name: str, process: subprocess.Popen[str]) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="")

    thread = threading.Thread(target=worker, name=f"{name}-log-stream", daemon=True)
    thread.start()
    return thread


def wait_for_openai_api(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.startup_timeout
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/models"
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
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
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def request_completions(args: argparse.Namespace) -> list[dict[str, Any]]:
    request_count = max(int(args.num_attention_servers), 1)
    if request_count == 1:
        return [request_completion(args)]

    responses: list[dict[str, Any] | None] = [None] * request_count
    with ThreadPoolExecutor(max_workers=request_count) as executor:
        futures = {
            executor.submit(request_completion, args): request_idx
            for request_idx in range(request_count)
        }
        for future in as_completed(futures):
            responses[futures[future]] = future.result()

    return [response for response in responses if response is not None]


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


def print_command(name: str, command: list[str], cuda_visible_devices: str) -> None:
    printable = " ".join(shell_quote(token) for token in command)
    print(f"\n=== Starting {name} (CUDA_VISIBLE_DEVICES={cuda_visible_devices}) ===")
    print(printable)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "@%_+=:,./-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
