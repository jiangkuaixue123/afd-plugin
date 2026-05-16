# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD topology helpers shared by P2P connector and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from afd_plugin.config import AFDConfig


@dataclass(frozen=True, slots=True)
class AFDRankMapping:
    """Rank mapping for the Phase 4 P2P connector.

    The P2P world always places FFN ranks first, followed by Attention ranks:
    ``[F0, F1, ..., A0, A1, ...]``.  Each FFN rank owns one subgroup containing
    itself at subgroup rank 0 and one or more consecutive Attention ranks.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    ratio: int
    subgroup_index: int
    rank_in_subgroup: int
    subgroup_ranks: tuple[int, ...]
    dp_metadata_destinations: tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_attention_top_min_size_rank(self) -> bool:
        return (
            self.ffn_size
            <= self.world_rank
            < self.ffn_size + self.min_size
        )

    @property
    def participates_in_dp_metadata_group(self) -> bool:
        return self.world_rank < self.ffn_size or self.is_attention_top_min_size_rank


def topology_from_config(config: AFDConfig) -> tuple[int, int]:
    """Return ``(attention_size, ffn_size)`` for an AFD config.

    ``extra_config["afd_size"]`` is kept for compatibility with the original
    in-tree AFD branch and accepts values such as ``"4A2F"`` or ``"4:2"``.
    Canonical plugin fields remain ``num_attention_servers`` and
    ``num_ffn_servers``.
    """

    afd_size = config.extra_config.get("afd_size")
    if afd_size is None:
        return config.num_attention_servers, config.num_ffn_servers

    match = re.fullmatch(r"\s*(\d+)\D+(\d+)\D*\s*", str(afd_size))
    if match is None:
        raise ValueError(
            "extra_config['afd_size'] must contain two positive integers, "
            f"got {afd_size!r}",
        )
    attention_size, ffn_size = (int(match.group(1)), int(match.group(2)))
    if attention_size <= 0 or ffn_size <= 0:
        raise ValueError("extra_config['afd_size'] values must be positive")
    return attention_size, ffn_size


def validate_p2p_topology(config: AFDConfig) -> None:
    attention_size, ffn_size = topology_from_config(config)
    if attention_size < ffn_size:
        raise ValueError(
            "p2pconnector currently requires num_attention_servers >= "
            f"num_ffn_servers, got {attention_size} < {ffn_size}",
        )
    if attention_size % ffn_size != 0:
        raise ValueError(
            "p2pconnector currently requires num_attention_servers to be a "
            "multiple of num_ffn_servers, got "
            f"{attention_size} and {ffn_size}",
        )


def build_rank_mapping(
    config: AFDConfig,
    role_rank: int | None = None,
) -> AFDRankMapping:
    """Build the P2P rank mapping for one Attention or FFN process."""

    validate_p2p_topology(config)
    attention_size, ffn_size = topology_from_config(config)
    role_rank = config.afd_server_rank if role_rank is None else role_rank
    if role_rank < 0:
        raise ValueError(f"AFD role rank must be non-negative, got {role_rank}")

    if config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        subgroup_index = role_rank // (attention_size // ffn_size)
    elif config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        subgroup_index = role_rank
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    ratio = attention_size // ffn_size
    min_size = min(ffn_size, attention_size)
    ffn_ranks = list(range(ffn_size))
    attention_ranks = list(range(ffn_size, ffn_size + attention_size))
    subgroup_ranks = tuple(
        [ffn_ranks[subgroup_index]]
        + [
            attention_ranks[subgroup_index * ratio + offset]
            for offset in range(ratio)
        ],
    )
    rank_in_subgroup = subgroup_ranks.index(world_rank)
    p2p_rank = role_rank + min_size if config.role == "attention" else role_rank

    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        destination = local_attention_rank
        while destination < ffn_size:
            destinations.append(destination)
            destination += min_size

    return AFDRankMapping(
        role=config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        ratio=ratio,
        subgroup_index=subgroup_index,
        rank_in_subgroup=rank_in_subgroup,
        subgroup_ranks=subgroup_ranks,
        dp_metadata_destinations=tuple(destinations),
    )


def resolve_hidden_size(vllm_config: object) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(hf_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("p2pconnector requires model_config.hf_config.hidden_size")
    return int(hidden_size)


def resolve_num_hidden_layers(vllm_config: object) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", None)
    value: Any = getattr(text_config, "num_hidden_layers", None)
    if value is None:
        value = getattr(hf_config, "num_hidden_layers", None)
    if value is None:
        return 1
    return int(value)


__all__ = [
    "AFDRankMapping",
    "build_rank_mapping",
    "resolve_hidden_size",
    "resolve_num_hidden_layers",
    "topology_from_config",
    "validate_p2p_topology",
]
