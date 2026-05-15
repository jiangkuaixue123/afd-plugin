# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD metadata objects shared by runtime classes and model wrappers."""

from __future__ import annotations

import copy
import time
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

    def __post_init__(self) -> None:
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("all sequence lengths must be positive")

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
    ) -> AFDConnectorMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[seq_len],
            dtype=dtype,
            device=device,
            request_id=request_id,
            ffn_need_forward_data=ffn_need_forward_data,
            timestamp=time.time(),
            num_of_stages=num_of_stages,
            afd_tokens_lens=list(afd_tokens_lens or ()),
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
    ) -> AFDConnectorMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=list(seq_lens),
            dtype=dtype,
            device=device,
            request_id=request_id,
            timestamp=time.time(),
        )

    def get_split_indices(self) -> list[int]:
        indices: list[int] = []
        cumsum = 0
        for length in self.seq_lens[:-1]:
            cumsum += length
            indices.append(cumsum)
        return indices

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        return bool(tensor_shape) and tensor_shape[0] == self.total_tokens


@dataclass(slots=True)
class AFDMetadata:
    """Forward-context metadata visible to plugin-owned model wrappers."""

    afd_tokens_start_loc: list[int]
    afd_reqs_start_loc: list[int]
    afd_stage_idx: int
    afd_connector: Any
    afd_tokens_lens: list[int]
    num_of_stages: int

    def clone(self) -> AFDMetadata:
        return copy.copy(self)


__all__ = [
    "AFDConnectorMetadata",
    "AFDMetadata",
    "FFNNeedForwardData",
]
