# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Placeholder for the future CAMP2P Ascend connector migration."""

from __future__ import annotations

from typing import Any

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase


class CAMP2PAFDConnector(AFDConnectorBase):
    """Import-safe class path for the production NPU connector.

    The first NPU runtime version uses ``npudummyconnector``.  This class exists
    so config/class-path validation can name ``camp2pconnector`` before the real
    HCCL/CAM-backed implementation lands.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config)
        raise NotImplementedError(
            "camp2pconnector is reserved for the production NPU connector; "
            "use npudummyconnector for the first NPU runtime version",
        )

    @property
    def is_initialized(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def init_afd_connector(self) -> None:
        raise NotImplementedError("camp2pconnector is not implemented yet")

    def send_attn_output(self, hidden_states: Any, metadata: Any) -> Any:
        raise NotImplementedError("camp2pconnector is not implemented yet")

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError("camp2pconnector is not implemented yet")

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
    ) -> tuple[Any, Any]:
        raise NotImplementedError("camp2pconnector is not implemented yet")

    def send_ffn_output(self, ffn_output: Any, metadata: Any) -> None:
        raise NotImplementedError("camp2pconnector is not implemented yet")


__all__ = ["CAMP2PAFDConnector"]
