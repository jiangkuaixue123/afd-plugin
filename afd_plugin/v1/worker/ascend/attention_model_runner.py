# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side model runner for the first AFD runtime version."""

from __future__ import annotations

import inspect
import textwrap
from typing import Any

from afd_plugin.compat.ascend import (
    enable_npu_afd_ubatching_if_requested,
    ensure_vllm_config_has_afd_proxy,
    fail_if_unsupported_npu_afd_features,
    mirror_afd_metadata_on_forward_context,
    npu_afd_ubatching_requested,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import AFDConnectorFactory, AFDDPMetadata, AFDMetadata
from afd_plugin.v1.worker._optional import optional_class
from afd_plugin.v1.worker.attention_model_runner import (
    _batch_execution_values,
    _check_ubatch_thresholds,
    _forward_context_num_tokens,
    _full_cudagraph_padded_tokens,
    _has_enough_tokens_for_ubatches,
    _resolve_cudagraph_mode_none,
    _resolve_world_ranks,
    _with_dp_derived_afd_rank,
)
from afd_plugin.v1.worker.ascend.ubatch_wrapper import AFDNPUUBatchWrapper
from afd_plugin.v1.worker.ubatch_wrapper import build_ubatch_dp_metadata_list

_NPUModelRunner, _NPUModelRunner_IMPORT_ERROR = optional_class(
    "vllm_ascend.worker.model_runner_v1",
    "NPUModelRunner",
)

_PATCHED_ASCEND_BUILD_ATTENTION_METADATA: Any | None = None


class AFDNPUAttentionModelRunner(_NPUModelRunner):  # type: ignore[misc, valid-type]
    """NPU model runner that injects AFD metadata into Ascend forward context."""

    afd_expected_role = "attention"
    vllm_base_import_error = _NPUModelRunner_IMPORT_ERROR

    def __init__(self, vllm_config: object, device: object) -> None:
        if _NPUModelRunner_IMPORT_ERROR is not None:
            raise RuntimeError(
                "AFDNPUAttentionModelRunner requires an importable vLLM-Ascend runtime",
            ) from _NPUModelRunner_IMPORT_ERROR

        afd_config = self.parse_config(vllm_config)
        ensure_vllm_config_has_afd_proxy(vllm_config, afd_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        if not self.afd_config.enabled:
            raise ValueError("AFD NPU Attention runtime requires enabled=true")
        enable_npu_afd_ubatching_if_requested(vllm_config)
        fail_if_unsupported_npu_afd_features(vllm_config)
        self.afd_config = _with_dp_derived_afd_rank(vllm_config, self.afd_config)
        rank, local_rank = _resolve_world_ranks()
        self.afd_connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.afd_connector.init_afd_connector()
        self._is_warmup = False
        self._afd_is_graph_capturing = False
        self._afd_pending_metadata: AFDMetadata | None = None
        self._afd_pending_ubatch_slices: Any = None
        self._afd_transaction_counter = 0

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        result = super().load_model(*args, **kwargs)
        if npu_afd_ubatching_requested(self.vllm_config):
            self._install_afd_npu_ubatch_wrapper()
        return result

    def _install_afd_npu_ubatch_wrapper(self) -> None:
        if isinstance(self.model, AFDNPUUBatchWrapper):
            return

        runtime_mode = _resolve_cudagraph_mode_none()
        self.model = AFDNPUUBatchWrapper(
            self.model,
            self.vllm_config,
            runtime_mode,
            self.device,
        )
        print(
            "[AFDNPUAttentionModelRunner] installed AFDNPUUBatchWrapper "
            f"num_ubatches={self.vllm_config.parallel_config.num_ubatches}",
            flush=True,
        )

    def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        self._install_afd_metadata_on_forward_context(forward_context)
        return super()._model_forward(*args, **kwargs)

    def _build_attention_metadata(self, *args: Any, **kwargs: Any) -> Any:
        values = _attention_metadata_values(args, kwargs)
        ubatch_slices = values.get("ubatch_slices")
        self._afd_pending_ubatch_slices = ubatch_slices
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            int(values.get("num_tokens", 0)),
        )
        if ubatch_slices is not None:
            print(
                "[AFDNPUAttentionModelRunner] build attention metadata "
                "with per-ubatch split metadata",
                flush=True,
            )
            self._ensure_ubatch_metadata_builders()
            return _patched_ascend_build_attention_metadata()(self, *args, **kwargs)
        return super()._build_attention_metadata(*args, **kwargs)

    def _ensure_ubatch_metadata_builders(self) -> None:
        num_ubatches = int(self.vllm_config.parallel_config.num_ubatches)
        for kv_cache_group in self.attn_groups:
            for attn_group in kv_cache_group:
                if len(attn_group.metadata_builders) >= num_ubatches:
                    continue
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    num_metadata_builders=num_ubatches,
                )

    def _determine_batch_execution_and_padding(self, *args: Any, **kwargs: Any) -> Any:
        enable_npu_afd_ubatching_if_requested(self.vllm_config)
        result = super()._determine_batch_execution_and_padding(*args, **kwargs)
        (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = result
        values = _batch_execution_values(args, kwargs)
        num_tokens = int(values.get("num_tokens", 0))
        if should_ubatch and not _has_enough_tokens_for_ubatches(
            self.vllm_config,
            num_tokens,
        ):
            should_ubatch = False
        elif not should_ubatch:
            should_ubatch = self._should_ubatch_without_vllm_dp(*args, **kwargs)

        print(
            "[AFDNPUAttentionModelRunner] determine ubatch "
            f"base_should_ubatch={result[2]} final_should_ubatch={should_ubatch} "
            f"num_tokens={num_tokens} num_reqs={values.get('num_reqs')} "
            f"enable_dbo={getattr(self.vllm_config.parallel_config, 'enable_dbo', None)} "
            f"use_ubatching={getattr(self.vllm_config.parallel_config, 'use_ubatching', None)} "
            f"num_ubatches={getattr(self.vllm_config.parallel_config, 'num_ubatches', None)}",
            flush=True,
        )
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _should_ubatch_without_vllm_dp(self, *args: Any, **kwargs: Any) -> bool:
        parallel_config = self.vllm_config.parallel_config
        if int(parallel_config.data_parallel_size) > 1:
            return False
        if not bool(parallel_config.use_ubatching):
            return False
        if not bool(kwargs.get("allow_microbatching", True)):
            return False

        values = _batch_execution_values(args, kwargs)
        num_tokens = int(values["num_tokens"])
        if not _has_enough_tokens_for_ubatches(self.vllm_config, num_tokens):
            return False

        uniform_decode = _is_uniform_decode(
            speculative_config=getattr(self, "speculative_config", None),
            uniform_decode_query_len=int(self.uniform_decode_query_len),
            num_tokens=num_tokens,
            num_reqs=int(values["num_reqs"]),
            max_num_scheduled_tokens=int(values["max_num_scheduled_tokens"]),
            force_uniform_decode=values.get("force_uniform_decode"),
        )
        return _check_ubatch_thresholds(
            parallel_config,
            num_tokens,
            bool(uniform_decode),
        )

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        previous = self._afd_is_graph_capturing
        self._afd_is_graph_capturing = bool(
            kwargs.get("is_graph_capturing", previous),
        )
        try:
            return super()._dummy_run(*args, **kwargs)
        finally:
            self._afd_is_graph_capturing = previous
            self._afd_pending_metadata = None
            self._afd_pending_ubatch_slices = None

    def _build_afd_metadata(
        self,
        ubatch_slices: Any,
        num_tokens_unpadded: int,
    ) -> AFDMetadata:
        if ubatch_slices and len(ubatch_slices) > 1:
            afd_tokens_start_loc = [ub.token_slice.start for ub in ubatch_slices]
            afd_reqs_start_loc = [ub.request_slice.start for ub in ubatch_slices]
            afd_tokens_lens = [ub.num_tokens for ub in ubatch_slices]
            afd_tokens_unpadded_lens = [int(ub.num_tokens) for ub in ubatch_slices]
            num_of_stages = len(ubatch_slices)
        else:
            afd_tokens_start_loc = [0]
            afd_reqs_start_loc = [0]
            afd_tokens_lens = [num_tokens_unpadded]
            afd_tokens_unpadded_lens = [num_tokens_unpadded]
            num_of_stages = 1

        return AFDMetadata(
            afd_tokens_start_loc=afd_tokens_start_loc,
            afd_reqs_start_loc=afd_reqs_start_loc,
            afd_stage_idx=0,
            afd_connector=self.afd_connector,
            afd_tokens_lens=afd_tokens_lens,
            num_of_stages=num_of_stages,
            transaction_id=self._next_afd_transaction_id(),
            afd_tokens_unpadded_lens=afd_tokens_unpadded_lens,
        )

    def _install_afd_metadata_on_forward_context(
        self,
        forward_context: object,
    ) -> None:
        if (
            getattr(forward_context, "ubatch_slices", None) is None
            and self._afd_pending_ubatch_slices is not None
        ):
            forward_context.ubatch_slices = self._afd_pending_ubatch_slices

        if self._afd_pending_metadata is None:
            self._afd_pending_metadata = self._build_afd_metadata(
                forward_context.ubatch_slices,
                _forward_context_num_tokens(forward_context, self.vllm_config),
            )

        mirror_afd_metadata_on_forward_context(
            forward_context,
            self._afd_pending_metadata,
        )
        dp_metadata = forward_context.dp_metadata
        ubatch_slices = forward_context.ubatch_slices
        padded_graph_tokens = _full_cudagraph_padded_tokens(forward_context)
        if padded_graph_tokens is not None and not ubatch_slices:
            dp_metadata = self._build_capture_dp_metadata(padded_graph_tokens)
        self._send_dp_metadata(dp_metadata, ubatch_slices)

    def _send_dp_metadata(self, dp_metadata: Any, ubatch_slices: Any) -> None:
        if ubatch_slices and len(ubatch_slices) > 1:
            dp_metadata_list = {
                idx: metadata
                for idx, metadata in enumerate(
                    build_ubatch_dp_metadata_list(self.vllm_config, ubatch_slices),
                )
            }
        else:
            dp_metadata = self._ensure_dp_metadata(dp_metadata)
            dp_metadata_list = {0: dp_metadata}
        is_warmup = bool(self._is_warmup)
        is_graph_capturing = bool(self._afd_is_graph_capturing)
        self.afd_connector.update_state_from_dp_metadata(
            dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        should_send = self.afd_connector.is_attn_top_min_size_rank(
            self.afd_connector.world_rank,
        )
        if should_send:
            print(
                "[AFDNPUAttentionModelRunner] send dp metadata "
                f"stages={list(dp_metadata_list)} "
                f"ubatches={None if ubatch_slices is None else len(ubatch_slices)}",
                flush=True,
            )
            self.afd_connector.send_dp_metadata_list(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )
            print("[AFDNPUAttentionModelRunner] sent dp metadata", flush=True)

    def _ensure_dp_metadata(self, dp_metadata: Any) -> Any:
        if dp_metadata is not None:
            return dp_metadata

        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        if dp_size != 1:
            raise RuntimeError("AFD NPU Attention expected DPMetadata for DP > 1")
        if self._afd_pending_metadata is None:
            raise RuntimeError("AFD metadata is not available for DP fallback")

        num_tokens = int(self._afd_pending_metadata.afd_tokens_lens[0])
        return _make_uniform_dp_metadata(dp_size, num_tokens)

    def _build_capture_dp_metadata(self, num_tokens: int) -> Any:
        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        return _make_uniform_dp_metadata(dp_size, int(num_tokens))

    def shutdown(self) -> None:
        self.afd_connector.close()
        super().shutdown()

    def _next_afd_transaction_id(self) -> str:
        counter = self._afd_transaction_counter
        self._afd_transaction_counter = counter + 1
        return f"afd-npu-{counter}"


def _make_uniform_dp_metadata(dp_size: int, num_tokens: int) -> AFDDPMetadata:
    try:
        import torch
    except ModuleNotFoundError:
        num_tokens_across_dp_cpu = [int(num_tokens)] * int(dp_size)
    else:
        num_tokens_across_dp_cpu = torch.full(
            (int(dp_size),),
            int(num_tokens),
            dtype=torch.int32,
            device="cpu",
        )
    return AFDDPMetadata(num_tokens_across_dp_cpu=num_tokens_across_dp_cpu)


def _attention_metadata_values(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    names = [
        "num_tokens",
        "num_reqs",
        "max_num_scheduled_tokens",
        "ubatch_slices",
    ]
    values = dict(zip(names, args, strict=False))
    values.update(kwargs)
    return values


def _is_uniform_decode(
    *,
    speculative_config: object | None,
    uniform_decode_query_len: int,
    num_tokens: int,
    num_reqs: int,
    max_num_scheduled_tokens: int,
    force_uniform_decode: object | None,
) -> bool:
    if force_uniform_decode is not None:
        return bool(force_uniform_decode)
    return bool(
        (speculative_config is None)
        and (int(max_num_scheduled_tokens) == int(uniform_decode_query_len))
        and (int(num_tokens) == int(max_num_scheduled_tokens) * int(num_reqs)),
    )


def _patched_ascend_build_attention_metadata() -> Any:
    global _PATCHED_ASCEND_BUILD_ATTENTION_METADATA
    if _PATCHED_ASCEND_BUILD_ATTENTION_METADATA is not None:
        return _PATCHED_ASCEND_BUILD_ATTENTION_METADATA

    import vllm_ascend.worker.model_runner_v1 as ascend_model_runner

    source = textwrap.dedent(
        inspect.getsource(ascend_model_runner.NPUModelRunner._build_attention_metadata),
    )
    new = """\
{loop_indent}for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
{body_indent}if ubatch_slices is not None:
{body_indent}    for ubid, _cm in enumerate(
{body_indent}        split_attn_metadata(ubatch_slices, cm)
{body_indent}    ):
{body_indent}        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)
{body_indent}else:
{body_indent}    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
"""
    source = _replace_ascend_attention_metadata_split_loop(source, new)
    namespace = dict(vars(ascend_model_runner))
    namespace["split_attn_metadata"] = _split_ascend_attn_metadata
    exec(compile(source, "<afd_npu_attention_metadata_patch>", "exec"), namespace)
    _PATCHED_ASCEND_BUILD_ATTENTION_METADATA = namespace["_build_attention_metadata"]
    return _PATCHED_ASCEND_BUILD_ATTENTION_METADATA


def _split_ascend_attn_metadata(
    ubatch_slices: list[Any],
    common_attn_metadata: Any,
) -> list[Any]:
    return [
        _make_ascend_metadata_with_slice(ubatch_slice, common_attn_metadata)
        for ubatch_slice in ubatch_slices
    ]


def _make_ascend_metadata_with_slice(ubatch_slice: Any, attn_metadata: Any) -> Any:
    from dataclasses import replace

    import torch

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice
    start_locs = attn_metadata.query_start_loc_cpu
    first_req = int(request_slice.start)
    first_tok = int(token_slice.start)
    last_req = int(request_slice.stop) - 1
    last_tok = int(token_slice.stop) - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )

    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = _slice_query_start_locs(start_locs, request_slice)
    query_start_loc = _slice_query_start_locs(
        attn_metadata.query_start_loc,
        request_slice,
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped

    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = _slice_optional(attn_metadata.seq_lens_cpu, request_slice)
    optimistic_seq_lens_cpu = _slice_optional(
        getattr(attn_metadata, "_seq_lens_cpu", None),
        request_slice,
    )

    if splits_last_request:
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        if optimistic_seq_lens_cpu is not None:
            optimistic_seq_lens_cpu = optimistic_seq_lens_cpu.clone()
            optimistic_seq_lens_cpu[-1] -= tokens_skipped

    seq_lens_for_max = (
        seq_lens_cpu
        if seq_lens_cpu is not None
        else optimistic_seq_lens_cpu
        if optimistic_seq_lens_cpu is not None
        else seq_lens.cpu()
    )
    max_seq_len = int(seq_lens_for_max.max())
    max_query_len = int(
        torch.max(
            torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]),
        ).item(),
    )
    if max_query_len == 0:
        max_query_len = int(attn_metadata.max_query_len)

    actual_seq_lengths_q = []
    if getattr(attn_metadata, "actual_seq_lengths_q", None):
        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()

    return replace(
        attn_metadata,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        _seq_lens_cpu=optimistic_seq_lens_cpu,
        num_computed_tokens_cpu=_slice_optional(
            attn_metadata.num_computed_tokens_cpu,
            request_slice,
        ),
        _num_computed_tokens_cpu=_slice_optional(
            getattr(attn_metadata, "_num_computed_tokens_cpu", None),
            request_slice,
        ),
        num_reqs=int(request_slice.stop) - int(request_slice.start),
        num_actual_tokens=int(token_slice.stop) - int(token_slice.start),
        num_input_tokens=int(token_slice.stop) - int(token_slice.start),
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=attn_metadata.block_table_tensor[request_slice],
        slot_mapping=attn_metadata.slot_mapping[token_slice],
        positions=_slice_optional(attn_metadata.positions, token_slice),
        actual_seq_lengths_q=actual_seq_lengths_q,
    )


def _slice_query_start_locs(query_start_loc: Any, request_slice: slice) -> Any:
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _slice_optional(value: Any, value_slice: slice) -> Any:
    if value is None:
        return None
    return value[value_slice]


def _replace_ascend_attention_metadata_split_loop(source: str, template: str) -> str:
    lines = source.splitlines()
    loop_marker = "for attn_gid in range(len(self.attn_groups[kv_cache_gid])):"
    call_marker = "_build_attn_group_metadata(kv_cache_gid, attn_gid, cm)"
    for idx, line in enumerate(lines):
        if loop_marker not in line:
            continue
        next_idx = idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        if next_idx >= len(lines) or call_marker not in lines[next_idx]:
            continue

        loop_indent = line[: len(line) - len(line.lstrip())]
        body_line = lines[next_idx]
        body_indent = body_line[: len(body_line) - len(body_line.lstrip())]
        replacement = template.format(
            loop_indent=loop_indent,
            body_indent=body_indent,
        ).splitlines()
        lines[idx : next_idx + 1] = replacement
        return "\n".join(lines) + "\n"

    raise RuntimeError("Unable to patch vllm-ascend attention metadata split point")


__all__ = ["AFDNPUAttentionModelRunner"]
