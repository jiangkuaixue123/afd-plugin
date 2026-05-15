# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dummy AFD connector used by the Phase 2 Attention runtime MVP."""

from __future__ import annotations

from collections import deque
from typing import Any

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDConnectorMetadata


class DummyAFDConnector(AFDConnectorBase):
    """No-op connector that records metadata and returns zero-like tensors."""

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config)
        self._is_initialized = False
        self.events: deque[tuple[Any, AFDConnectorMetadata]] = deque(
            maxlen=afd_config.num_afd_stages,
        )
        self.dp_metadata_updates: list[dict[int, Any]] = []
        self.sent_dp_metadata_lists: list[dict[int, Any]] = []
        self.world_rank = rank
        self.init_afd_connector()

    def init_afd_connector(self) -> None:
        self._is_initialized = True

    def close(self) -> None:
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, Any],
        is_warmup: bool,
    ) -> None:
        del is_warmup
        self.dp_metadata_updates.append(dict(dp_metadata_list))

    def is_attn_top_min_size_rank(self, world_rank: int) -> bool:
        return world_rank == self.world_rank

    def send_dp_metadata_list(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_warmup: bool = False,
    ) -> None:
        del is_warmup
        self.sent_dp_metadata_lists.append(dict(dp_metadata_list))

    def send_attn_output(
        self,
        hidden_states: Any,
        metadata: AFDConnectorMetadata,
    ) -> None:
        if hasattr(hidden_states, "shape") and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"AFD metadata token count {metadata.total_tokens}",
            )
        if not metadata.is_single_sequence:
            raise ValueError("attention side metadata must describe one sequence")
        self.events.append((hidden_states, metadata))
        return None

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        del handle
        ref_tensor = kwargs.get("ref_tensor")
        if ref_tensor is not None:
            return _zeros_like(ref_tensor)
        hidden_states, _ = self.events.popleft()
        return _zeros_like(hidden_states)

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
    ) -> tuple[Any, AFDConnectorMetadata]:
        raise NotImplementedError(
            "DummyAFDConnector.recv_attn_output is implemented in Phase 3",
        )

    def send_ffn_output(
        self,
        ffn_output: Any,
        metadata: AFDConnectorMetadata,
    ) -> None:
        if hasattr(ffn_output, "shape") and not metadata.validate_tensor_shape(
            tuple(ffn_output.shape),
        ):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match metadata",
            )


def _zeros_like(value: Any) -> Any:
    if hasattr(value, "new_zeros") and hasattr(value, "shape"):
        return value.new_zeros(value.shape)
    return value


__all__ = ["DummyAFDConnector"]
