# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector contract for AFD Attention/FFN communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.metadata import AFDConnectorMetadata


class AFDConnectorBase(ABC):
    """Abstract base class for plugin-owned AFD connectors."""

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.vllm_config = vllm_config
        self.afd_config = afd_config

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_afd_connector(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: Any,
        metadata: AFDConnectorMetadata,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
    ) -> tuple[Any, AFDConnectorMetadata]:
        raise NotImplementedError

    @abstractmethod
    def send_ffn_output(
        self,
        ffn_output: Any,
        metadata: AFDConnectorMetadata,
    ) -> None:
        raise NotImplementedError


__all__ = ["AFDConnectorBase"]
