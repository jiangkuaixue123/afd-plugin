# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD metadata objects shared by runtime classes and model wrappers."""

from __future__ import annotations

import copy
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FFNNeedForwardData:
    """Small payload carried from Attention side to FFN side."""

    moe_comm_method: Any
    num_input_tokens: int
    with_prefill: bool
    total_num_scheduled_tokens: int | None
    is_dummy_run: bool = False


@dataclass(frozen=True, slots=True)
class AFDMessageKey:
    """Stable identity for one AFD connector message."""

    transaction_id: str
    ubatch_idx: int
    afd_stage_idx: int
    direction: str
    layer_idx: int


@dataclass(slots=True)
class AFDSingleDPMetadata:
    """Minimal DPMetadata-compatible payload for attention DP=1."""

    num_tokens_across_dp_cpu: Any
    max_tokens_across_dp_cpu: Any | None = None
    local_sizes: list[int] | None = None

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int) -> Iterator[list[int]]:
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    @contextmanager
    def chunked_sizes(
        self,
        sequence_parallel_size: int,
        max_chunk_size_per_rank: int,
        chunk_idx: int,
    ) -> Iterator[list[int]]:
        sp_tokens = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
        )
        self.local_sizes = [
            max(
                1,
                min(
                    max_chunk_size_per_rank,
                    size - max_chunk_size_per_rank * chunk_idx,
                ),
            )
            for size in sp_tokens
        ]
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: Any,
    sequence_parallel_size: int,
) -> list[int]:
    if hasattr(num_tokens_across_dp_cpu, "repeat_interleave"):
        sp_tokens = (
            num_tokens_across_dp_cpu + sequence_parallel_size - 1
        ) // sequence_parallel_size
        return sp_tokens.repeat_interleave(sequence_parallel_size).tolist()

    if isinstance(num_tokens_across_dp_cpu, (int, float)):
        values = [int(num_tokens_across_dp_cpu)]
    else:
        values = [int(value) for value in num_tokens_across_dp_cpu]
    local_sizes: list[int] = []
    for value in values:
        local_sizes.extend(
            [max(1, (value + sequence_parallel_size - 1) // sequence_parallel_size)]
            * sequence_parallel_size,
        )
    return local_sizes


@dataclass(slots=True)
class AFDConnectorMetadata:
    """Communication metadata for one AFD Attention/FFN exchange."""

    layer_idx: int
    stage_idx: int
    seq_lens: list[int]
    dtype: Any
    device: Any
    topk_idx: Any = None
    topk_weights: Any = None
    moe_expert_num: int | None = None
    shared_expert_num: int | None = None
    scale: Any = None
    expertTokenNumsOut: Any = None
    recv_handle_list: list[Any] | None = None
    request_id: str | None = None
    timestamp: float | None = None
    ffn_need_forward_data: FFNNeedForwardData | None = None
    num_of_stages: int = 1
    afd_tokens_lens: list[int] = field(default_factory=list)
    ubatch_idx: int | None = None
    transaction_id: str | None = None
    direction: str = "attention_to_ffn"
    afd_tokens_unpadded_lens: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("all sequence lengths must be positive")
        if self.ubatch_idx is None:
            self.ubatch_idx = self.stage_idx

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)

    @property
    def is_single_sequence(self) -> bool:
        return self.num_sequences == 1

    @property
    def is_multi_sequence(self) -> bool:
        return self.num_sequences > 1

    @property
    def message_key(self) -> AFDMessageKey:
        return AFDMessageKey(
            transaction_id=self.transaction_id or self.request_id or "default",
            ubatch_idx=int(self.ubatch_idx or 0),
            afd_stage_idx=int(self.stage_idx),
            direction=self.direction,
            layer_idx=int(self.layer_idx),
        )

    @classmethod
    def create_attention_metadata(
        cls,
        *,
        layer_idx: int,
        stage_idx: int,
        seq_len: int,
        dtype: Any,
        device: Any,
        request_id: str | None = None,
        ffn_need_forward_data: FFNNeedForwardData | None = None,
        num_of_stages: int = 1,
        afd_tokens_lens: list[int] | None = None,
        ubatch_idx: int | None = None,
        transaction_id: str | None = None,
        afd_tokens_unpadded_lens: list[int] | None = None,
    ) -> AFDConnectorMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[seq_len],
            dtype=dtype,
            device=device,
            request_id=request_id,
            ffn_need_forward_data=ffn_need_forward_data,
            timestamp=_compile_safe_timestamp(),
            num_of_stages=num_of_stages,
            afd_tokens_lens=list(afd_tokens_lens or ()),
            ubatch_idx=ubatch_idx,
            transaction_id=transaction_id,
            afd_tokens_unpadded_lens=list(afd_tokens_unpadded_lens or ()),
        )

    @classmethod
    def create_ffn_metadata(
        cls,
        *,
        layer_idx: int,
        stage_idx: int,
        seq_lens: list[int],
        dtype: Any,
        device: Any,
        request_id: str | None = None,
        ubatch_idx: int | None = None,
        transaction_id: str | None = None,
    ) -> AFDConnectorMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=list(seq_lens),
            dtype=dtype,
            device=device,
            request_id=request_id,
            timestamp=_compile_safe_timestamp(),
            ubatch_idx=ubatch_idx,
            transaction_id=transaction_id,
            direction="ffn_to_attention",
        )

    def get_split_indices(self) -> list[int]:
        indices: list[int] = []
        cumsum = 0
        for length in self.seq_lens[:-1]:
            cumsum += length
            indices.append(cumsum)
        return indices

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        return len(tensor_shape) > 0 and tensor_shape[0] == self.total_tokens


def _compile_safe_timestamp() -> float:
    if _torch_is_compiling():
        return 0.0
    return time.time()


def _torch_is_compiling() -> bool:
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


@dataclass(slots=True)
class AFDMetadata:
    """Forward-context metadata visible to plugin-owned model wrappers."""

    afd_tokens_start_loc: list[int]
    afd_reqs_start_loc: list[int]
    afd_stage_idx: int
    afd_connector: Any
    afd_tokens_lens: list[int]
    num_of_stages: int
    ubatch_idx: int = 0
    transaction_id: str | None = None
    afd_tokens_unpadded_lens: list[int] = field(default_factory=list)

    def clone(self) -> AFDMetadata:
        cloned = copy.copy(self)
        cloned.afd_tokens_start_loc = list(self.afd_tokens_start_loc)
        cloned.afd_reqs_start_loc = list(self.afd_reqs_start_loc)
        cloned.afd_tokens_lens = list(self.afd_tokens_lens)
        cloned.afd_tokens_unpadded_lens = list(self.afd_tokens_unpadded_lens)
        return cloned


__all__ = [
    "AFDConnectorMetadata",
    "AFDMessageKey",
    "AFDMetadata",
    "AFDSingleDPMetadata",
    "FFNNeedForwardData",
]
