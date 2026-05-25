# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""In-process NPU dummy connector for first-version runtime validation."""

from __future__ import annotations

import copy
import queue
import time
from dataclasses import dataclass
from typing import Any

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDConnectorMetadata

_CHANNELS: dict[tuple[str, int], _DummyChannel] = {}


@dataclass
class AFDRecvOutput:
    hidden_states: Any
    metadata: AFDConnectorMetadata
    group_list: Any = None
    topk_weights: Any = None
    topk_ids: Any = None
    router_logits: Any = None
    row_idx: Any = None
    x_active_mask: Any = None
    dynamic_scales: Any = None
    cam_p2p_ep_name: str | None = None


class _DummyChannel:
    def __init__(self) -> None:
        self.dp_metadata_queue: queue.Queue[tuple[dict[int, Any], bool, bool]] = (
            queue.Queue()
        )
        self.attn_output_queue: queue.Queue[tuple[Any, AFDConnectorMetadata]] = (
            queue.Queue()
        )
        self.ffn_output_queue: queue.Queue[tuple[Any, AFDConnectorMetadata]] = (
            queue.Queue()
        )


class NPUDummyAFDConnector(AFDConnectorBase):
    """A CPU/NPU-safe in-process connector for Attention/FFN lifecycle smoke runs."""

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config)
        self._initialized = False
        self.world_rank = rank
        self.attn_size = int(afd_config.num_attention_servers)
        self.ffn_size = int(afd_config.num_ffn_servers)
        self.dp_metadata_list: dict[int, Any] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self._channel_key = (afd_config.host, int(afd_config.port))

    @property
    def _channel(self) -> _DummyChannel:
        channel = _CHANNELS.get(self._channel_key)
        if channel is None:
            channel = _DummyChannel()
            _CHANNELS[self._channel_key] = channel
        return channel

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        self._initialized = True
        _ = self._channel

    def close(self) -> None:
        self._initialized = False

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        self.dp_metadata_list = dict(dp_metadata_list)
        self.is_graph_capturing = bool(is_graph_capturing)
        self.is_warmup = bool(is_warmup)

    def is_attn_top_min_size_rank(self, world_rank: int) -> bool:
        del world_rank
        return self.afd_config.role == "attention"

    def send_dp_metadata_list(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        self._channel.dp_metadata_queue.put(
            (copy.copy(dp_metadata_list), bool(is_graph_capturing), bool(is_warmup)),
        )

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[dict[int, Any], bool, bool]:
        try:
            return self._channel.dp_metadata_queue.get(
                timeout=_timeout_seconds(timeout_ms),
            )
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for AFD dummy DP metadata") from exc

    def send_attn_output(
        self,
        hidden_states: Any,
        metadata: AFDConnectorMetadata,
    ) -> None:
        self._channel.attn_output_queue.put((hidden_states, metadata))

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
        **kwargs: Any,
    ) -> tuple[Any, AFDConnectorMetadata] | AFDRecvOutput:
        del kwargs
        hidden_states, metadata = self._get_stage_item(
            self._channel.attn_output_queue,
            ubatch_idx,
            timeout_ms,
        )
        if self.afd_config.extra_config.get("recv_output_object"):
            return AFDRecvOutput(hidden_states=hidden_states, metadata=metadata)
        return hidden_states, metadata

    def send_ffn_output(
        self,
        ffn_output: Any,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._channel.ffn_output_queue.put((ffn_output, metadata))

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        del handle
        ubatch_idx = kwargs.get("ubatch_idx")
        ref_tensor = kwargs.get("ref_tensor")
        timeout_ms = kwargs.get("timeout_ms", 100)
        try:
            output, _metadata = self._get_stage_item(
                self._channel.ffn_output_queue,
                ubatch_idx,
                timeout_ms,
            )
        except TimeoutError:
            if ref_tensor is not None and self.afd_config.extra_config.get(
                "dummy_passthrough_without_peer",
                True,
            ):
                return ref_tensor
            raise
        return output

    def create_recv_metadata(self, **kwargs: Any) -> AFDConnectorMetadata:
        dp_metadata_list = kwargs.get("dp_metadata_list") or self.dp_metadata_list
        ubatch_idx = int(kwargs.get("ubatch_idx", 0))
        layer_idx = int(kwargs.get("layer_idx", 0))
        seq_len = _num_tokens_for_stage(dp_metadata_list, ubatch_idx)
        return AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=[seq_len],
        )

    def configure_metadata(
        self,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        del metadata, kwargs

    def update_metadata(
        self,
        metadata: AFDConnectorMetadata,
        recv_output: Any,
    ) -> None:
        payload_metadata = getattr(recv_output, "metadata", None)
        if payload_metadata is not None:
            metadata.seq_lens = list(getattr(payload_metadata, "seq_lens", []))

    @staticmethod
    def _get_stage_item(
        item_queue: queue.Queue[tuple[Any, AFDConnectorMetadata]],
        stage_idx: int | None,
        timeout_ms: int | None,
    ) -> tuple[Any, AFDConnectorMetadata]:
        deadline = None
        if timeout_ms is not None:
            deadline = time.monotonic() + _timeout_seconds(timeout_ms)
        held: list[tuple[Any, AFDConnectorMetadata]] = []
        try:
            while True:
                timeout = None
                if deadline is not None:
                    timeout = max(0.0, deadline - time.monotonic())
                try:
                    item = item_queue.get(timeout=timeout)
                except queue.Empty as exc:
                    raise TimeoutError(
                        "timed out waiting for AFD dummy tensor payload",
                    ) from exc
                metadata = item[1]
                if stage_idx is None or int(metadata.stage_idx) == int(stage_idx):
                    return item
                held.append(item)
        finally:
            for item in held:
                item_queue.put(item)


def _timeout_seconds(timeout_ms: int | None) -> float | None:
    if timeout_ms is None:
        return None
    return max(float(timeout_ms) / 1000.0, 0.0)


def _num_tokens_for_stage(dp_metadata_list: dict[int, Any], stage_idx: int) -> int:
    dp_metadata = dp_metadata_list.get(int(stage_idx))
    token_counts = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
    if token_counts is None:
        return 1
    item = token_counts[0]
    item_fn = getattr(item, "item", None)
    return max(1, int(item_fn() if callable(item_fn) else item))


__all__ = ["AFDRecvOutput", "NPUDummyAFDConnector"]
