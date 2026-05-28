# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side model runner for the first AFD runtime version."""

from __future__ import annotations

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
            kwargs = dict(kwargs)
            kwargs["ubatch_slices"] = None
            print(
                "[AFDNPUAttentionModelRunner] build attention metadata "
                "with single metadata dict for NPU ubatch",
                flush=True,
            )
        return super()._build_attention_metadata(*args, **kwargs)

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
            self.afd_connector.send_dp_metadata_list(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )

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


__all__ = ["AFDNPUAttentionModelRunner"]
