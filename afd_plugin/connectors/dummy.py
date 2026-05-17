# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""In-process dummy AFD connector for CPU-safe AFD runtime smoke tests."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDConnectorMetadata


@dataclass
class _DummyState:
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.Lock()),
    )
    attn_to_ffn: deque[tuple[Any, AFDConnectorMetadata]] = field(
        default_factory=deque,
    )
    ffn_to_attn: deque[tuple[Any, AFDConnectorMetadata]] = field(
        default_factory=deque,
    )
    dp_metadata: deque[tuple[dict[int, Any], bool, bool]] = field(
        default_factory=deque,
    )


_DUMMY_STATE = _DummyState()


class DummyAFDConnector(AFDConnectorBase):
    """In-process connector that links Attention and FFN dummy runtimes.

    The dummy backend deliberately stays CPU-safe. It is not a cross-process
    transport; Phase 4 owns real P2P communication. For Phase 3 it gives tests
    and local development a real queue-backed Attention -> FFN -> Attention
    round trip without importing CUDA-heavy modules.
    """

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
        self._state = _DUMMY_STATE
        self.init_afd_connector()

    def init_afd_connector(self) -> None:
        self._is_initialized = True

    def close(self) -> None:
        self._is_initialized = False
        with self._state.condition:
            self._state.condition.notify_all()

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
        is_attn_graph_capturing: bool = False,
    ) -> None:
        self.sent_dp_metadata_lists.append(dict(dp_metadata_list))
        with self._state.condition:
            self._state.dp_metadata.append(
                (dict(dp_metadata_list), bool(is_attn_graph_capturing), is_warmup),
            )
            self._state.condition.notify_all()

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[dict[int, Any], bool, bool]:
        deadline = _deadline(timeout_ms)
        with self._state.condition:
            while not self._state.dp_metadata:
                _wait_for_item(
                    self._state.condition,
                    deadline,
                    "timed out waiting for dummy AFD DP metadata",
                )
            return self._state.dp_metadata.popleft()

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
        metadata.direction = "attention_to_ffn"
        self.events.append((hidden_states, metadata))
        with self._state.condition:
            self._state.attn_to_ffn.append((hidden_states, metadata))
            self._state.condition.notify_all()
        return None

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        del handle
        timeout_ms = kwargs.get("timeout_ms")
        ubatch_idx = kwargs.get("ubatch_idx")
        if timeout_ms is not None:
            deadline = _deadline(timeout_ms)
            with self._state.condition:
                while not _has_matching_ubatch(self._state.ffn_to_attn, ubatch_idx):
                    _wait_for_item(
                        self._state.condition,
                        deadline,
                        "timed out waiting for dummy AFD FFN output",
                    )
                ffn_output, _ = _pop_matching_ubatch(
                    self._state.ffn_to_attn,
                    ubatch_idx,
                )
                return ffn_output

        with self._state.condition:
            if _has_matching_ubatch(self._state.ffn_to_attn, ubatch_idx):
                ffn_output, _ = _pop_matching_ubatch(
                    self._state.ffn_to_attn,
                    ubatch_idx,
                )
                return ffn_output

        ref_tensor = kwargs.get("ref_tensor")
        if ref_tensor is not None:
            return _zeros_like(ref_tensor)
        hidden_states, _ = self.events.popleft()
        with self._state.condition:
            _discard_queued_attention_event(self._state.attn_to_ffn, hidden_states)
        return _zeros_like(hidden_states)

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
    ) -> tuple[Any, AFDConnectorMetadata]:
        deadline = _deadline(timeout_ms)
        with self._state.condition:
            while not _has_matching_ubatch(self._state.attn_to_ffn, ubatch_idx):
                _wait_for_item(
                    self._state.condition,
                    deadline,
                    "timed out waiting for dummy AFD attention output",
                )
            return _pop_matching_ubatch(self._state.attn_to_ffn, ubatch_idx)

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
        metadata.direction = "ffn_to_attention"
        with self._state.condition:
            self._state.ffn_to_attn.append((ffn_output, metadata))
            self._state.condition.notify_all()


def _zeros_like(value: Any) -> Any:
    if hasattr(value, "new_zeros") and hasattr(value, "shape"):
        return value.new_zeros(value.shape)
    return value


def _deadline(timeout_ms: int | None) -> float | None:
    if timeout_ms is None:
        return None
    return time.monotonic() + timeout_ms / 1000


def _wait_for_item(
    condition: threading.Condition,
    deadline: float | None,
    timeout_message: str,
) -> None:
    if deadline is None:
        condition.wait()
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    condition.wait(remaining)


def _discard_queued_attention_event(
    queue: deque[tuple[Any, AFDConnectorMetadata]],
    hidden_states: Any,
) -> None:
    for item in tuple(queue):
        if item[0] is hidden_states:
            queue.remove(item)
            break


def _has_matching_ubatch(
    queue: deque[tuple[Any, AFDConnectorMetadata]],
    ubatch_idx: int | None,
) -> bool:
    if ubatch_idx is None:
        return bool(queue)
    return any(
        _metadata_ubatch_idx(metadata) == int(ubatch_idx) for _, metadata in queue
    )


def _pop_matching_ubatch(
    queue: deque[tuple[Any, AFDConnectorMetadata]],
    ubatch_idx: int | None,
) -> tuple[Any, AFDConnectorMetadata]:
    if ubatch_idx is None:
        return queue.popleft()
    expected = int(ubatch_idx)
    for item in tuple(queue):
        if _metadata_ubatch_idx(item[1]) == expected:
            queue.remove(item)
            return item
    raise IndexError(f"no dummy AFD message for ubatch_idx={expected}")


def _metadata_ubatch_idx(metadata: AFDConnectorMetadata) -> int:
    value = getattr(metadata, "ubatch_idx", None)
    if value is None:
        value = getattr(metadata, "stage_idx", 0)
    return int(value)


__all__ = ["DummyAFDConnector"]
