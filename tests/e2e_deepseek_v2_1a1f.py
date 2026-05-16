#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run a manual 1A1F DeepSeekV2 AFD end-to-end smoke test.

This script is intentionally opt-in and is not collected by pytest. It starts:

1. one FFN-side ``vllm serve --headless`` process;
2. one Attention-side ``vllm serve`` OpenAI-compatible API process;
3. one completion request against the Attention-side API.

Current Phase 4 limitations are enforced by the command line: full weights are
loaded on both sides, and ``--enforce-eager`` is always passed because CUDA
graph support is not implemented yet.
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
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    processes: list[subprocess.Popen[str]] = []
    log_threads: list[threading.Thread] = []

    try:
        ffn_cmd = build_vllm_command(args, role="ffn")
        attention_cmd = build_vllm_command(args, role="attention")

        print_command("FFN", ffn_cmd, args.ffn_gpu)
        ffn_proc = start_process("ffn", ffn_cmd, build_env(args.ffn_gpu))
        processes.append(ffn_proc)
        log_threads.append(stream_output("ffn", ffn_proc))

        time.sleep(args.ffn_start_delay)
        ensure_alive(ffn_proc, "FFN process exited during startup")

        print_command("ATTN", attention_cmd, args.attention_gpu)
        attention_proc = start_process(
            "attention",
            attention_cmd,
            build_env(args.attention_gpu),
        )
        processes.append(attention_proc)
        log_threads.append(stream_output("attention", attention_proc))

        wait_for_openai_api(args)
        response = request_completion(args)
        print("\n=== Completion response ===")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    finally:
        terminate_processes(processes)
        for thread in log_threads:
            thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual 1A1F DeepSeekV2 AFD E2E smoke test.",
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
    parser.add_argument("--attention-gpu", default="0")
    parser.add_argument("--ffn-gpu", default="1")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument("--afd-port", type=int, default=1239)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--ffn-start-delay", type=float, default=8)
    parser.add_argument("--prompt", default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Optional served model name. Also used in the completion request.",
    )
    parser.add_argument(
        "--common-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added to both processes.",
    )
    parser.add_argument(
        "--attention-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to the Attention process.",
    )
    parser.add_argument(
        "--ffn-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to the FFN process.",
    )
    return parser.parse_args()


def build_vllm_command(args: argparse.Namespace, *, role: str) -> list[str]:
    afd_config = {
        "afd": {
            "enabled": True,
            "role": role,
            "connector": "p2pconnector",
            "host": args.afd_host,
            "port": args.afd_port,
            "num_attention_servers": 1,
            "num_ffn_servers": 1,
            "afd_server_rank": 0,
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
        "--additional-config",
        json.dumps(afd_config, separators=(",", ":")),
    ]
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if role == "attention":
        cmd.extend(["--host", args.api_host, "--port", str(args.api_port)])
    else:
        cmd.append("--headless")
    cmd.extend(args.common_vllm_arg)
    cmd.extend(args.attention_vllm_arg if role == "attention" else args.ffn_vllm_arg)
    return cmd


def build_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["VLLM_PLUGINS"] = "afd"
    env["PYTHONUNBUFFERED"] = "1"
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
    url = f"http://{args.api_host}:{args.api_port}/v1/models"
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
    url = f"http://{args.api_host}:{args.api_port}/v1/completions"
    payload = {
        "model": args.served_model_name or args.model,
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
