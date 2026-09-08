#!/usr/bin/env python3
"""Audit and summarize the DSV4 Flash prefill benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TOPOLOGIES = ("afd_dp6tp4", "afd_dp3tp8", "ep16")
ALL_CHUNK_SIZES = (4096, 8192, 16384, 32768, 65536)
FORMAL_CHUNKS = {
    "afd_dp6tp4": (16384, 32768, 65536),
    "afd_dp3tp8": (16384, 32768, 65536),
    "ep16": (4096, 8192, 16384, 32768, 65536),
}
TERMINAL_DEPLOYMENT_STATUSES = {
    "oom_undeployable",
    "capacity_undeployable",
}
OFFERED_RATES = (4.0, 6.0, 8.0)
REPEATS = (1, 2, 3)
NPU_DIES = {
    "afd_dp6tp4": 32,
    "afd_dp3tp8": 32,
    "ep16": 16,
}
EXPECTED_REQUESTS = 1536
EXPECTED_INPUT_TOKENS = 15_803_063
EXPECTED_OUTPUT_TOKENS = 1536
EXPECTED_DATASET_SHA256 = (
    "1ebccbd149bc8f28568d3d5eced3911d2a9473fb015bdb829b898a26abf63d08"
)
EXPECTED_RUNS_PER_CHUNK = len(OFFERED_RATES) * len(REPEATS)
DETAILED_FIELDS = (
    "input_lens",
    "output_lens",
    "ttfts",
    "itls",
    "start_times",
    "generated_texts",
    "errors",
)


@dataclass(frozen=True)
class RunRecord:
    topology: str
    chunk_size: int
    offered_rps: float
    repeat: int
    result_path: str
    duration_s: float
    completed: int
    failed: int
    achieved_qps: float
    input_tokens_per_s: float
    qps_per_npu: float
    input_tokens_per_s_per_npu: float
    mean_ttft_ms: float
    p25_ttft_ms: float
    p50_ttft_ms: float
    p90_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    mean_e2el_ms: float
    p25_e2el_ms: float
    p50_e2el_ms: float
    p90_e2el_ms: float
    p95_e2el_ms: float
    p99_e2el_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_root",
        type=Path,
        help="bench_results/dsv4-flash directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: result root)",
    )
    return parser.parse_args()


def rate_label(rate: float) -> str:
    return str(int(rate)) if rate.is_integer() else str(rate)


def require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing/non-numeric {key}: {value!r}")
    return float(value)


def validate_result(
    payload: dict[str, Any],
    topology: str,
    chunk_size: int,
    offered_rps: float,
    repeat: int,
    path: Path,
) -> None:
    expected_scalars = {
        "completed": EXPECTED_REQUESTS,
        "failed": 0,
        "total_input_tokens": EXPECTED_INPUT_TOKENS,
        "total_output_tokens": EXPECTED_OUTPUT_TOKENS,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"{path}: expected {key}={expected}, got {payload.get(key)!r}"
            )
    expected_metadata = {
        "topology": topology,
        "chunk_size": str(chunk_size),
        "offered_rps": rate_label(offered_rps),
        "repeat": str(repeat),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "prefix_cache": "false",
        "kv_connector": "false",
    }
    for key, expected in expected_metadata.items():
        if str(payload.get(key)) != expected:
            raise ValueError(
                f"{path}: expected {key}={expected!r}, got {payload.get(key)!r}"
            )
    for key in DETAILED_FIELDS:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != EXPECTED_REQUESTS:
            size = len(values) if isinstance(values, list) else None
            raise ValueError(f"{path}: expected {key} length 1536, got {size}")


def load_run(
    path: Path,
    topology: str,
    chunk_size: int,
    offered_rps: float,
    repeat: int,
) -> RunRecord:
    payload = json.loads(path.read_text())
    validate_result(payload, topology, chunk_size, offered_rps, repeat, path)
    duration = require_number(payload, "duration")
    achieved_qps = require_number(payload, "request_throughput")
    input_throughput = EXPECTED_INPUT_TOKENS / duration
    npu_dies = NPU_DIES[topology]
    return RunRecord(
        topology=topology,
        chunk_size=chunk_size,
        offered_rps=offered_rps,
        repeat=repeat,
        result_path=str(path),
        duration_s=duration,
        completed=int(payload["completed"]),
        failed=int(payload["failed"]),
        achieved_qps=achieved_qps,
        input_tokens_per_s=input_throughput,
        qps_per_npu=achieved_qps / npu_dies,
        input_tokens_per_s_per_npu=input_throughput / npu_dies,
        mean_ttft_ms=require_number(payload, "mean_ttft_ms"),
        p25_ttft_ms=require_number(payload, "p25_ttft_ms"),
        p50_ttft_ms=require_number(payload, "p50_ttft_ms"),
        p90_ttft_ms=require_number(payload, "p90_ttft_ms"),
        p95_ttft_ms=require_number(payload, "p95_ttft_ms"),
        p99_ttft_ms=require_number(payload, "p99_ttft_ms"),
        mean_e2el_ms=require_number(payload, "mean_e2el_ms"),
        p25_e2el_ms=require_number(payload, "p25_e2el_ms"),
        p50_e2el_ms=require_number(payload, "p50_e2el_ms"),
        p90_e2el_ms=require_number(payload, "p90_e2el_ms"),
        p95_e2el_ms=require_number(payload, "p95_e2el_ms"),
        p99_e2el_ms=require_number(payload, "p99_e2el_ms"),
    )


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_group(records: list[RunRecord]) -> dict[str, Any]:
    first = records[0]
    summary: dict[str, Any] = {
        "topology": first.topology,
        "chunk_size": first.chunk_size,
        "offered_rps": first.offered_rps,
        "valid_repeats": len(records),
    }
    metric_names = (
        "duration_s",
        "achieved_qps",
        "input_tokens_per_s",
        "qps_per_npu",
        "input_tokens_per_s_per_npu",
        "mean_ttft_ms",
        "p25_ttft_ms",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_e2el_ms",
        "p25_e2el_ms",
        "p50_e2el_ms",
        "p90_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
    )
    for metric in metric_names:
        values = [float(getattr(record, metric)) for record in records]
        metric_mean = mean(values)
        metric_std = sample_std(values)
        summary[f"{metric}_mean"] = metric_mean
        summary[f"{metric}_std"] = metric_std
        summary[f"{metric}_cv"] = (
            metric_std / metric_mean if not math.isclose(metric_mean, 0.0) else 0.0
        )
    summary["offered_load_sustained"] = (
        summary["achieved_qps_mean"] >= first.offered_rps * 0.95
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_deployment_status(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "missing", ""
    payload = json.loads(path.read_text())
    return str(payload.get("status", "unknown")), str(payload.get("reason", ""))


def build_report(
    coverage: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> str:
    expected_chunks = sum(len(chunks) for chunks in FORMAL_CHUNKS.values())
    expected_runs = expected_chunks * EXPECTED_RUNS_PER_CHUNK
    formal_coverage = [row for row in coverage if row["formal_required"]]
    complete_chunks = sum(row["coverage_complete"] for row in formal_coverage)
    actual_runs = sum(int(row["valid_results"]) for row in formal_coverage)
    terminal_chunks = sum(
        row["deployment_status"] in TERMINAL_DEPLOYMENT_STATUSES
        for row in formal_coverage
    )
    terminal_runs = terminal_chunks * EXPECTED_RUNS_PER_CHUNK
    lines = [
        "# DSV4 Flash Prefill 实验报告（自动汇总）",
        "",
        "## 覆盖率",
        "",
        f"- 完整 topology/chunk：{complete_chunks}/{expected_chunks}",
        f"- 有效 detailed 运行：{actual_runs}/{expected_runs}",
        f"- 部署终态覆盖：{terminal_runs}/{expected_runs}",
        f"- 已解释正式测量点：{actual_runs + terminal_runs}/{expected_runs}",
        f"- 正式 chunk 只有在拥有 {EXPECTED_RUNS_PER_CHUNK} 份有效结果，或明确记录有证据的不可部署终态时才算覆盖完成。",
        "- AFD 的 chunk=4096/8192 仅作探索性证据；EP16 的五档 chunk 均为正式矩阵。",
        "",
        "| topology | chunk | formal required | valid/expected | deployment status | complete |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in coverage:
        lines.append(
            f"| {row['topology']} | {row['chunk_size']} | "
            f"{'yes' if row['formal_required'] else 'no'} | "
            f"{row['valid_results']}/{EXPECTED_RUNS_PER_CHUNK} | {row['deployment_status']} | "
            f"{('yes' if row['coverage_complete'] else 'no') if row['formal_required'] else 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## 各 topology/chunk 最大稳定 offered load",
            "",
            "稳定定义：三次重复的平均 achieved QPS ≥ offered QPS × 95%。",
            "",
            "| topology | chunk | stable offered QPS | achieved QPS | input tok/s | P99 TTFT ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_chunk: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_chunk[(summary["topology"], summary["chunk_size"])].append(summary)
    for topology in TOPOLOGIES:
        for chunk_size in FORMAL_CHUNKS[topology]:
            candidates = [
                row
                for row in by_chunk.get((topology, chunk_size), [])
                if row["valid_repeats"] == len(REPEATS)
                and row["offered_load_sustained"]
            ]
            if not candidates:
                lines.append(f"| {topology} | {chunk_size} | - | - | - | - |")
                continue
            best = max(candidates, key=lambda row: float(row["offered_rps"]))
            lines.append(
                f"| {topology} | {chunk_size} | {best['offered_rps']:.0f} | "
                f"{best['achieved_qps_mean']:.3f} | "
                f"{best['input_tokens_per_s_mean']:.1f} | "
                f"{best['p99_ttft_ms_mean']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `runs.csv`：每次运行的原始汇总字段；",
            "- `summary.csv`：每个 topology/chunk/RPS 的三次重复统计；",
            f"- `coverage.csv`：{len(coverage)} 个 topology/chunk 的覆盖审计。",
            "",
            "> 本文件是机器汇总草稿。主实验报告还需结合 OOM、运行日志、版本冻结信息和跨拓扑对比进行解释。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or result_root).resolve()
    records: list[RunRecord] = []
    coverage: list[dict[str, Any]] = []
    validation_errors: list[str] = []

    for topology in TOPOLOGIES:
        for chunk_size in ALL_CHUNK_SIZES:
            chunk_dir = result_root / topology / f"chunk_{chunk_size}"
            status, reason = load_deployment_status(
                chunk_dir / "deployment_status.json"
            )
            valid_count = 0
            for offered_rps in OFFERED_RATES:
                for repeat in REPEATS:
                    path = (
                        chunk_dir
                        / f"rps_{rate_label(offered_rps)}"
                        / f"repeat_{repeat}"
                        / "result.json"
                    )
                    if not path.exists():
                        continue
                    try:
                        records.append(
                            load_run(
                                path,
                                topology,
                                chunk_size,
                                offered_rps,
                                repeat,
                            )
                        )
                        valid_count += 1
                    except (ValueError, TypeError, json.JSONDecodeError) as error:
                        validation_errors.append(str(error))
            coverage_complete = (
                valid_count == EXPECTED_RUNS_PER_CHUNK
                or status in TERMINAL_DEPLOYMENT_STATUSES
            )
            coverage.append(
                {
                    "topology": topology,
                    "chunk_size": chunk_size,
                    "valid_results": valid_count,
                    "expected_results": EXPECTED_RUNS_PER_CHUNK,
                    "deployment_status": status,
                    "deployment_reason": reason,
                    "formal_required": chunk_size in FORMAL_CHUNKS[topology],
                    "coverage_complete": coverage_complete,
                }
            )

    grouped: dict[tuple[str, int, float], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.topology, record.chunk_size, record.offered_rps)].append(
            record
        )
    summaries = [
        summarize_group(sorted(group, key=lambda record: record.repeat))
        for _, group in sorted(grouped.items())
    ]

    write_csv(output_dir / "runs.csv", [asdict(record) for record in records])
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "coverage.csv", coverage)
    (output_dir / "EXPERIMENT_REPORT.generated.md").write_text(
        build_report(coverage, summaries)
    )
    (output_dir / "validation_errors.txt").write_text(
        "\n".join(validation_errors) + ("\n" if validation_errors else "")
    )

    incomplete = [
        row
        for row in coverage
        if row["formal_required"] and not row["coverage_complete"]
    ]
    if validation_errors:
        print(f"Validation errors: {len(validation_errors)}")
    print(
        "Valid formal runs: "
        f"{sum(record.chunk_size in FORMAL_CHUNKS[record.topology] for record in records)}/"
        f"{sum(len(chunks) for chunks in FORMAL_CHUNKS.values()) * EXPECTED_RUNS_PER_CHUNK}"
    )
    print(
        "Complete formal topology/chunk entries: "
        f"{sum(len(chunks) for chunks in FORMAL_CHUNKS.values()) - len(incomplete)}/"
        f"{sum(len(chunks) for chunks in FORMAL_CHUNKS.values())}"
    )
    return 1 if validation_errors or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
