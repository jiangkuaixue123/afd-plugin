# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side model runner for the Phase 2 MVP."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDMetadata,
    AFDSingleDPMetadata,
)
from afd_plugin.runtime._optional import optional_class
from afd_plugin.runtime.ubatch_wrapper import (
    AFDUBatchWrapper,
    build_ubatch_dp_metadata_list,
    trace_ubatch_slices,
)
from afd_plugin.tracing import afd_trace, dp_metadata_summary

_GPUModelRunner, _GPUModelRunner_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner",
)


class AFDAttentionModelRunner(_GPUModelRunner):  # type: ignore[misc, valid-type]
    """Attention model runner that injects AFD metadata into forward context."""

    afd_expected_role = "attention"
    vllm_base_import_error = _GPUModelRunner_IMPORT_ERROR

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _GPUModelRunner_IMPORT_ERROR is not None:
            raise RuntimeError(
                "AFDAttentionModelRunner requires an importable vLLM runtime",
            ) from _GPUModelRunner_IMPORT_ERROR

        super().__init__(*args, **kwargs)
        self.afd_config = self.parse_config(self.vllm_config)
        if not self.afd_config.enabled:
            raise ValueError("AFD Attention runtime requires enabled=true")
        fail_if_unsupported_ubatching(self.vllm_config)
        fail_if_cuda_graph_enabled(self.vllm_config)
        self.afd_config = _with_dp_derived_afd_rank(
            self.vllm_config,
            self.afd_config,
        )
        rank, local_rank = _resolve_world_ranks()
        self.afd_connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            self.vllm_config,
            self.afd_config,
        )
        self.afd_connector.init_afd_connector()
        self._is_warmup = False
        self._afd_pending_metadata: AFDMetadata | None = None
        self._afd_transaction_counter = 0

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    def _build_afd_metadata(
        self,
        ubatch_slices: Any,
        num_tokens_unpadded: int,
    ) -> AFDMetadata:
        if ubatch_slices and len(ubatch_slices) > 1:
            trace_ubatch_slices(ubatch_slices, source="attention_metadata")
            afd_tokens_start_loc = [ub.token_slice.start for ub in ubatch_slices]
            afd_reqs_start_loc = [ub.request_slice.start for ub in ubatch_slices]
            afd_tokens_lens = [ub.num_tokens for ub in ubatch_slices]
            afd_tokens_unpadded_lens = [
                int(getattr(ub, "num_tokens_unpadded", ub.num_tokens))
                for ub in ubatch_slices
            ]
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
        update = getattr(self.afd_connector, "update_state_from_dp_metadata", None)
        if callable(update):
            update(dp_metadata_list, is_warmup=self._is_warmup)

        should_send = True
        rank = getattr(self.afd_connector, "world_rank", None)
        is_top_rank = getattr(self.afd_connector, "is_attn_top_min_size_rank", None)
        if callable(is_top_rank) and rank is not None:
            should_send = bool(is_top_rank(rank))

        send = getattr(self.afd_connector, "send_dp_metadata_list", None)
        afd_trace(
            "attn_send_dp_metadata_decision",
            rank=rank,
            should_send=should_send,
            stages=sorted(int(stage_idx) for stage_idx in dp_metadata_list),
            dp_metadata=dp_metadata_summary(dp_metadata_list),
            is_warmup=self._is_warmup,
        )
        if should_send and callable(send):
            afd_trace(
                "attn_send_dp_metadata_begin",
                rank=rank,
                dp_metadata=dp_metadata_summary(dp_metadata_list),
                is_warmup=self._is_warmup,
            )
            send(dp_metadata_list, is_warmup=self._is_warmup)
            afd_trace(
                "attn_send_dp_metadata_done",
                rank=rank,
                dp_metadata=dp_metadata_summary(dp_metadata_list),
                is_warmup=self._is_warmup,
            )

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        use_ubatching = _is_ubatching_enabled(self.vllm_config)
        with _use_afd_ubatch_wrapper_during_load(use_ubatching):
            result = super().load_model(*args, **kwargs)
        if use_ubatching:
            self._install_afd_ubatch_wrapper()
        return result

    def _install_afd_ubatch_wrapper(self) -> None:
        if isinstance(self.model, AFDUBatchWrapper):
            return

        runtime_mode = _resolve_cudagraph_mode_none()
        native_wrapper_cls = _resolve_native_ubatch_wrapper()
        model = self.model
        if native_wrapper_cls is not None and isinstance(model, native_wrapper_cls):
            model = model.unwrap()
        self.model = AFDUBatchWrapper(
            model,
            self.vllm_config,
            runtime_mode,
            self.device,
        )

    def _ensure_dp_metadata(self, dp_metadata: Any) -> Any:
        if dp_metadata is not None:
            return dp_metadata

        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        dp_size = int(getattr(parallel_config, "data_parallel_size", 1))
        if dp_size != 1:
            raise RuntimeError("AFD expected vLLM DPMetadata for attention DP > 1")

        if self._afd_pending_metadata is None:
            raise RuntimeError("AFD metadata is not available for DP metadata fallback")
        if len(self._afd_pending_metadata.afd_tokens_lens) != 1:
            raise RuntimeError("AFD DP=1 fallback only supports one stage")

        import torch

        num_tokens = int(self._afd_pending_metadata.afd_tokens_lens[0])
        num_tokens_across_dp_cpu = torch.tensor(
            [num_tokens],
            dtype=torch.int32,
            device="cpu",
        )
        return AFDSingleDPMetadata(
            num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
            max_tokens_across_dp_cpu=torch.max(num_tokens_across_dp_cpu),
        )

    def _install_afd_metadata_on_forward_context(
        self,
        forward_context: object,
    ) -> None:
        if self._afd_pending_metadata is not None:
            forward_context.additional_kwargs["afd_metadata"] = (
                self._afd_pending_metadata
            )
        self._send_dp_metadata(
            getattr(forward_context, "dp_metadata", None),
            getattr(forward_context, "ubatch_slices", None),
        )

    def _build_attention_metadata(self, *args: Any, **kwargs: Any) -> Any:
        num_tokens = kwargs.get("num_tokens", 0)
        ubatch_slices = kwargs.get("ubatch_slices")
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            int(num_tokens),
        )
        return super()._build_attention_metadata(*args, **kwargs)

    def _determine_batch_execution_and_padding(self, *args: Any, **kwargs: Any) -> Any:
        result = super()._determine_batch_execution_and_padding(*args, **kwargs)
        (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = result
        if should_ubatch:
            return result
        should_ubatch = self._should_ubatch_without_vllm_dp(*args, **kwargs)
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _should_ubatch_without_vllm_dp(self, *args: Any, **kwargs: Any) -> bool:
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        if parallel_config is None:
            return False
        if int(getattr(parallel_config, "data_parallel_size", 1)) > 1:
            return False
        if not bool(getattr(parallel_config, "use_ubatching", False)):
            return False
        if not bool(kwargs.get("allow_microbatching", True)):
            return False

        names = [
            "num_tokens",
            "num_reqs",
            "num_scheduled_tokens_np",
            "max_num_scheduled_tokens",
            "use_cascade_attn",
            "allow_microbatching",
            "force_eager",
            "force_uniform_decode",
        ]
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=values["max_num_scheduled_tokens"],
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=values["num_tokens"],
            num_reqs=values["num_reqs"],
            force_uniform_decode=values.get("force_uniform_decode"),
        )
        return _check_ubatch_thresholds(
            parallel_config,
            int(values["num_tokens"]),
            bool(uniform_decode),
        )

    def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        self._install_afd_metadata_on_forward_context(forward_context)
        return super()._model_forward(*args, **kwargs)

    def shutdown(self) -> None:
        close = getattr(getattr(self, "afd_connector", None), "close", None)
        if callable(close):
            close()
        parent_shutdown = getattr(super(), "shutdown", None)
        if callable(parent_shutdown):
            parent_shutdown()

    def _next_afd_transaction_id(self) -> str:
        counter = getattr(self, "_afd_transaction_counter", 0)
        self._afd_transaction_counter = counter + 1
        return f"afd-{counter}"


def fail_if_unsupported_ubatching(vllm_config: object) -> None:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return
    num_ubatches = int(getattr(parallel_config, "num_ubatches", 1))
    if _is_ubatching_enabled(vllm_config) and num_ubatches != 2:
        raise RuntimeError(
            "AFD Phase 5 currently supports exactly two ubatches; "
            f"got num_ubatches={num_ubatches}",
        )


fail_if_ubatching_enabled = fail_if_unsupported_ubatching


def fail_if_cuda_graph_enabled(vllm_config: object) -> None:
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "enforce_eager", True) is False:
        raise RuntimeError(
            "AFD CUDA graph support is deferred to Phase 6; pass --enforce-eager",
        )


def _resolve_world_ranks() -> tuple[int, int]:
    try:
        from vllm.distributed.parallel_state import get_world_group

        group = get_world_group()
        return int(group.rank), int(group.local_rank)
    except Exception:
        return 0, 0


def _with_dp_derived_afd_rank(
    vllm_config: object,
    afd_config: AFDConfig,
) -> AFDConfig:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return afd_config
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1))
    if dp_size <= 1:
        return afd_config
    dp_rank = int(getattr(parallel_config, "data_parallel_rank", 0))
    role_size = (
        afd_config.num_attention_servers
        if afd_config.role == "attention"
        else afd_config.num_ffn_servers
    )
    role_rank = afd_config.afd_server_rank + dp_rank
    if role_rank >= role_size:
        raise ValueError(
            "AFD role rank derived from data_parallel_rank is out of range: "
            f"base={afd_config.afd_server_rank}, dp_rank={dp_rank}, "
            f"role_size={role_size}",
        )
    return replace(afd_config, afd_server_rank=role_rank)


def _is_ubatching_enabled(vllm_config: object) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return False
    return bool(
        getattr(parallel_config, "use_ubatching", False)
        or getattr(parallel_config, "enable_dbo", False)
        or int(getattr(parallel_config, "ubatch_size", 1)) > 1
    )


def _resolve_native_ubatch_wrapper() -> type[Any] | None:
    try:
        from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

        return UBatchWrapper
    except Exception:
        return None


def _resolve_cudagraph_mode_none() -> Any:
    try:
        from vllm.config import CUDAGraphMode

        return CUDAGraphMode.NONE
    except Exception:
        return None


def _check_ubatch_thresholds(
    parallel_config: object,
    num_tokens: int,
    uniform_decode: bool,
) -> bool:
    try:
        from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds

        return bool(
            check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode),
        )
    except Exception:
        if not bool(getattr(parallel_config, "use_ubatching", False)):
            return False
        if uniform_decode:
            threshold = int(getattr(parallel_config, "dbo_decode_token_threshold", 32))
        else:
            threshold = int(
                getattr(parallel_config, "dbo_prefill_token_threshold", 512),
            )
        return num_tokens >= threshold


@contextmanager
def _use_afd_ubatch_wrapper_during_load(enabled: bool):
    if not enabled:
        yield
        return
    try:
        import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    except Exception:
        yield
        return

    original = getattr(gpu_model_runner, "UBatchWrapper", None)
    gpu_model_runner.UBatchWrapper = AFDUBatchWrapper
    try:
        yield
    finally:
        if original is None:
            delattr(gpu_model_runner, "UBatchWrapper")
        else:
            gpu_model_runner.UBatchWrapper = original


__all__ = [
    "AFDAttentionModelRunner",
    "fail_if_cuda_graph_enabled",
    "fail_if_ubatching_enabled",
    "fail_if_unsupported_ubatching",
]
