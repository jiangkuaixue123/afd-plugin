# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side model runner for the Phase 2 MVP."""

from __future__ import annotations

from typing import Any

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDMetadata,
    AFDSingleDPMetadata,
)
from afd_plugin.runtime._optional import optional_class

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
        fail_if_ubatching_enabled(self.vllm_config)
        fail_if_cuda_graph_enabled(self.vllm_config)
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

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    def _build_afd_metadata(
        self,
        ubatch_slices: Any,
        num_tokens_unpadded: int,
    ) -> AFDMetadata:
        if ubatch_slices and len(ubatch_slices) > 1:
            afd_tokens_start_loc = [ub.token_slice.start for ub in ubatch_slices]
            afd_reqs_start_loc = [ub.request_slice.start for ub in ubatch_slices]
            afd_tokens_lens = [ub.num_tokens for ub in ubatch_slices]
            num_of_stages = len(ubatch_slices)
        else:
            afd_tokens_start_loc = [0]
            afd_reqs_start_loc = [0]
            afd_tokens_lens = [num_tokens_unpadded]
            num_of_stages = 1

        return AFDMetadata(
            afd_tokens_start_loc=afd_tokens_start_loc,
            afd_reqs_start_loc=afd_reqs_start_loc,
            afd_stage_idx=0,
            afd_connector=self.afd_connector,
            afd_tokens_lens=afd_tokens_lens,
            num_of_stages=num_of_stages,
        )

    def _send_dp_metadata(self, dp_metadata: Any, ubatch_slices: Any) -> None:
        if ubatch_slices and len(ubatch_slices) > 1:
            raise RuntimeError("AFD + ubatching is deferred to Phase 5")

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
        if should_send and callable(send):
            send(dp_metadata_list, is_warmup=self._is_warmup)

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


def fail_if_ubatching_enabled(vllm_config: object) -> None:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return
    if getattr(parallel_config, "use_ubatching", False) or getattr(
        parallel_config,
        "enable_dbo",
        False,
    ):
        raise RuntimeError("AFD + ubatching/DBO is deferred to Phase 5")


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


__all__ = [
    "AFDAttentionModelRunner",
    "fail_if_cuda_graph_enabled",
    "fail_if_ubatching_enabled",
]
