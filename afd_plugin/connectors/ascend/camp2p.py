# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CAMP2P Ascend connector for the first real NPU AFD data path.

The module stays import-safe on CPU/GPU machines.  torch-npu, HCCL process
groups, and the plugin-owned ``torch.ops.afd_ascend`` custom ops are imported
only when the connector is initialized or used.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from afd_plugin.compat.ascend import ensure_afd_ascend_ops_loaded
from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDConnectorMetadata, AFDRecvOutput
from afd_plugin.distributed import init_afd_process_group, topology_from_config

_CAMP2P_CUSTOM_OPS_REGISTERED = False
_CAMP2P_STUB_IO_ENABLED = False
_CAMP2P_STUB_IO_NOTICE_PRINTED = False
_CAMP2P_STUB_IO_ENV = "AFD_CAMP2P_STUB_IO"
_CAMP2P_STUB_IO_EXTRA_KEYS = (
    "stub_io",
    "camp2p_stub_io",
    "enable_camp2p_stub_io",
)


@dataclass(slots=True)
class CAMP2PAFDConnectorMetadata:
    """CAMP2P payload metadata carried between recv and send phases.

    This mirrors ``vllm_ascend.distributed.metadata.CAMP2PAFDConnectorMetadata``
    while keeping the class plugin-owned and CPU-safe at import time.
    """

    moe_expert_num: int = 0
    shared_expert_num: int = 0
    scale: Any = None
    handle: Any = None
    quant_mode: int = 0
    aiv_num: int = 8
    batch_size: int = 0
    h: int = 0
    k: int = 1
    atten_batch_size: Any = None
    x_active_mask: Any = None
    group_ep: str = ""
    ffn_group_ep: str = ""


@dataclass(frozen=True, slots=True)
class _CAMP2PTopology:
    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    dp_metadata_destinations: tuple[int, ...]

    @property
    def p2p_world_size(self) -> int:
        return self.ffn_size + self.min_size

    @property
    def participates_in_p2p_group(self) -> bool:
        return self.world_rank < self.ffn_size or self.is_attn_top_min_size_rank

    @property
    def is_attn_top_min_size_rank(self) -> bool:
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size


class CAMP2PAFDConnector(AFDConnectorBase):
    """HCCL/CAMP2P-backed Attention <-> FFN connector for NPU.

    Phase 3B intentionally supports only eager single-stream execution.  ACL
    graph, ubatching/DBO, communication multistream, quantization modes, and
    compute-gate-on-attention remain rejected by NPU runtime validation.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config)
        self._initialized = False
        self._role_rank = _resolve_role_rank(vllm_config, afd_config)
        self.topology = build_camp2p_topology(afd_config, self._role_rank)
        self.world_rank = self.topology.world_rank
        self.p2p_rank = self.topology.p2p_rank
        self.attn_size = self.topology.attention_size
        self.ffn_size = self.topology.ffn_size
        self.min_size = self.topology.min_size
        self.ratio = self.attn_size // self.ffn_size
        self.dst_list = list(self.topology.dp_metadata_destinations)
        self.dp_metadata_list: dict[int, Any] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = _resolve_max_num_reqs(vllm_config)
        self.afd_pg: Any | None = None
        self.afd_ubatch_pgs: list[Any] = []
        self.p2p_pg: Any | None = None
        self.ffn_pg: Any | None = None
        self.hccl_comm_name = ""
        self.hccl_comm_name1 = ""
        self.hccl_comm_name_list: list[str] = []
        self.aiv_num = _resolve_aiv_num(afd_config)
        self.hidden_size = _resolve_hidden_size(vllm_config)
        self.num_experts_per_tok = _resolve_int_attr(
            vllm_config,
            "num_experts_per_tok",
            default=1,
        )
        self.num_routed_experts = _resolve_int_attr(
            vllm_config,
            "n_routed_experts",
            default=0,
        )
        self.num_shared_experts = _resolve_int_attr(
            vllm_config,
            "n_shared_experts",
            default=0,
        )
        self.mix_placement = bool(
            afd_config.extra_config.get("mix_placement", False),
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        if self._initialized:
            return
        ensure_afd_ascend_ops_loaded()
        import torch_npu  # noqa: F401

        _configure_camp2p_stub_io(self.afd_config)
        _register_camp2p_custom_ops()

        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.ffn_size + self.attn_size,
            rank=self.world_rank,
            group_name="afd",
            timeout=timedelta(minutes=30),
        )
        self.hccl_comm_name = _hccl_comm_name(self.afd_pg, self.world_rank)
        self.hccl_comm_name_list = [self.hccl_comm_name]
        self.afd_ubatch_pgs = [self.afd_pg]

        for ubatch_idx in range(
            1,
            _resolve_num_ubatches(self.vllm_config, self.afd_config),
        ):
            ubatch_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.world_rank,
                group_name=f"afd_ubatch_{ubatch_idx}",
                timeout=timedelta(minutes=30),
            )
            self.afd_ubatch_pgs.append(ubatch_pg)
            self.hccl_comm_name_list.append(
                _hccl_comm_name(ubatch_pg, self.world_rank),
            )

        if self.afd_config.role == "ffn":
            self.ffn_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size,
                rank=self.world_rank,
                group_name="afd_moe",
                timeout=timedelta(minutes=30),
            )
            self.hccl_comm_name1 = _hccl_comm_name(self.ffn_pg, self.world_rank)

        if self.topology.participates_in_p2p_group:
            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.topology.p2p_world_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        self._initialized = True

    def close(self) -> None:
        import torch.distributed as dist

        for group in (self.p2p_pg, self.ffn_pg, self.afd_pg):
            if group is not None:
                dist.destroy_process_group(group)
        self.p2p_pg = None
        self.ffn_pg = None
        self.afd_pg = None
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
        return self.ffn_size <= int(world_rank) < self.ffn_size + self.min_size

    def send_dp_metadata_list(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        if self.p2p_pg is None:
            return
        payload = (dp_metadata_list, bool(is_graph_capturing), bool(is_warmup))
        for dst in self.dst_list:
            _send_object(payload, dst=dst, group=self.p2p_pg)

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[dict[int, Any], bool, bool]:
        del timeout_ms
        if self.p2p_pg is None:
            raise RuntimeError("CAMP2P metadata process group is not initialized")
        src = self.p2p_rank % self.min_size + self.ffn_size
        payload = _recv_object(src=src, group=self.p2p_pg)
        if len(payload) == 3:
            dp_metadata_list, is_graph_capturing, is_warmup = payload
        else:
            dp_metadata_list, is_graph_capturing = payload
            is_warmup = False
        return dp_metadata_list, bool(is_graph_capturing), bool(is_warmup)

    def configure_metadata(
        self,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        metadata.connector_data = self._make_connector_data(
            batch_size=int(kwargs.get("batch_size", metadata.total_tokens)),
            layer_idx=int(metadata.layer_idx),
        )

    def create_recv_metadata(self, **kwargs: Any) -> AFDConnectorMetadata:
        dp_metadata_list = kwargs.get("dp_metadata_list") or self.dp_metadata_list
        ubatch_idx = int(kwargs.get("ubatch_idx", 0))
        layer_idx = int(kwargs.get("layer_idx", 0))
        batch_size = _num_tokens_for_ffn_rank(
            dp_metadata_list,
            ubatch_idx,
            ffn_rank=self._role_rank,
            attention_size=self.attn_size,
            ffn_size=self.ffn_size,
            fallback=int(kwargs.get("max_num_tokens", 1) or 1),
        )
        metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=[max(1, int(batch_size))],
        )
        metadata.connector_data = self._make_connector_data(
            batch_size=max(1, int(batch_size)),
            layer_idx=layer_idx,
        )
        return metadata

    def update_metadata(
        self,
        metadata: AFDConnectorMetadata,
        recv_output: AFDRecvOutput,
    ) -> None:
        connector_data = _ensure_connector_data(metadata)
        connector_data.atten_batch_size = recv_output.atten_batch_size
        connector_data.x_active_mask = recv_output.x_active_mask
        connector_data.handle = [
            recv_output.topk_ids,
            recv_output.topk_weights,
            recv_output.expand_idx,
            recv_output.ep_recv_counts,
            connector_data.atten_batch_size,
        ]

    def send_attn_output(
        self,
        hidden_states: Any,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> Any:
        self._require_initialized()
        if not _is_torch_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"CAMP2P metadata token count {metadata.total_tokens}",
            )
        connector_data = self._metadata_data_or_default(metadata, hidden_states)
        _set_forward_context_connector_data(connector_data)
        topk_ids = kwargs.get("topk_ids")
        topk_weights = kwargs.get("topk_weights")
        if _camp2p_stub_io_enabled():
            connector_data.group_ep = self._group_ep(int(metadata.stage_idx))
            connector_data.atten_batch_size = _stub_atten_batch_size(
                connector_data.batch_size,
            )
            connector_data.x_active_mask = _stub_x_active_mask(
                connector_data.batch_size,
            )
            _set_forward_context_connector_data(connector_data)
            return hidden_states, None
        torch = _torch()
        hidden_states = torch.ops.vllm.afd_camp2p_send_attn_output(
            hidden_states,
            topk_weights,
            topk_ids,
            connector_data.batch_size,
            connector_data.h,
            connector_data.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            self._group_ep(int(metadata.stage_idx)),
            connector_data.aiv_num,
            0,
        )
        return hidden_states, None

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        del handle
        self._require_initialized()
        ref_tensor = kwargs.get("ref_tensor")
        if ref_tensor is None:
            raise RuntimeError("CAMP2P recv_ffn_output requires ref_tensor")
        connector_data = _get_forward_context_connector_data()
        if connector_data is None:
            raise RuntimeError("CAMP2P Attention side is missing connector data")
        if _camp2p_stub_io_enabled():
            return ref_tensor
        torch = _torch()
        ubatch_idx = int(kwargs.get("ubatch_idx", 0) or 0)
        return torch.ops.vllm.afd_camp2p_recv_ffn_output(
            ref_tensor,
            connector_data.batch_size,
            connector_data.h,
            connector_data.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            self._group_ep(ubatch_idx),
            connector_data.aiv_num,
        )

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
        **kwargs: Any,
    ) -> AFDRecvOutput:
        del timeout_ms
        self._require_initialized()
        ubatch_idx = 0 if ubatch_idx is None else int(ubatch_idx)
        metadata = kwargs.get("metadata")
        if metadata is None:
            metadata = self.create_recv_metadata(
                ubatch_idx=ubatch_idx,
                layer_idx=int(kwargs.get("layer_idx", 0)),
            )
        connector_data = _ensure_connector_data(metadata)
        if _camp2p_stub_io_enabled():
            outputs = _stub_recv_attn_outputs(connector_data)
        else:
            outputs = _afd_ascend_ops().a2e(
                _empty_npu_tensor(dtype_name="bfloat16"),
                _empty_npu_tensor(dtype_name="int32"),
                _empty_npu_tensor(dtype_name="float32"),
                connector_data.batch_size,
                connector_data.h,
                connector_data.k,
                self.ffn_size,
                self.attn_size,
                self.world_rank,
                self._group_ep(ubatch_idx),
                connector_data.aiv_num,
                0,
            )
        connector_data.handle = list(outputs[:5])
        connector_data.atten_batch_size = outputs[3]
        connector_data.x_active_mask = outputs[4]
        connector_data.group_ep = self._group_ep(ubatch_idx)
        connector_data.ffn_group_ep = self.hccl_comm_name1
        return AFDRecvOutput(
            hidden_states=outputs[0],
            metadata=metadata,
            topk_ids=None,
            topk_weights=None,
            x_active_mask=outputs[4],
            cam_p2p_ep_name=self.hccl_comm_name1,
            atten_batch_size=outputs[3],
        )

    def send_ffn_output(
        self,
        ffn_output: Any,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        connector_data = _ensure_connector_data(metadata)
        if connector_data.atten_batch_size is None:
            raise RuntimeError("CAMP2P FFN side is missing A2E atten_batch_size")
        ubatch_idx = int(kwargs.get("ubatch_idx", metadata.stage_idx) or 0)
        if _camp2p_stub_io_enabled():
            return None
        _afd_ascend_ops().e2a(
            ffn_output,
            connector_data.atten_batch_size,
            connector_data.batch_size,
            connector_data.h,
            connector_data.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            self._group_ep(ubatch_idx),
            connector_data.aiv_num,
        )
        return None

    def _metadata_data_or_default(
        self,
        metadata: AFDConnectorMetadata,
        hidden_states: Any,
    ) -> CAMP2PAFDConnectorMetadata:
        data = metadata.connector_data
        if data is None:
            data = self._make_connector_data(
                batch_size=self.max_num_reqs,
                layer_idx=int(metadata.layer_idx),
            )
            metadata.connector_data = data
        return data

    def _make_connector_data(
        self,
        *,
        batch_size: int,
        layer_idx: int,
    ) -> CAMP2PAFDConnectorMetadata:
        del layer_idx
        k = self.num_experts_per_tok
        moe_experts = self.num_routed_experts
        shared_experts = 0
        if self.mix_placement:
            k += self.num_shared_experts
            moe_experts += self.num_shared_experts
            shared_experts = self.num_shared_experts
        return CAMP2PAFDConnectorMetadata(
            moe_expert_num=moe_experts,
            shared_expert_num=shared_experts,
            quant_mode=0,
            aiv_num=self.aiv_num,
            batch_size=max(1, int(batch_size)),
            h=self.hidden_size,
            k=max(1, int(k)),
        )

    def _group_ep(self, ubatch_idx: int) -> str:
        if not self.hccl_comm_name_list:
            return self.hccl_comm_name
        idx = min(max(int(ubatch_idx), 0), len(self.hccl_comm_name_list) - 1)
        return self.hccl_comm_name_list[idx]

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")


def build_camp2p_topology(
    afd_config: AFDConfig,
    role_rank: int | None = None,
) -> _CAMP2PTopology:
    attention_size, ffn_size = topology_from_config(afd_config)
    role_rank = afd_config.afd_server_rank if role_rank is None else int(role_rank)
    if attention_size <= 0 or ffn_size <= 0:
        raise ValueError("CAMP2P topology sizes must be positive")
    if attention_size < ffn_size:
        raise ValueError(
            "CAMP2P Phase 3B requires attention_size >= ffn_size, got "
            f"{attention_size} < {ffn_size}",
        )
    if role_rank < 0:
        raise ValueError(f"CAMP2P role rank must be non-negative, got {role_rank}")

    if afd_config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        p2p_rank = role_rank + min(ffn_size, attention_size)
    elif afd_config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        p2p_rank = role_rank
    else:
        raise ValueError(f"unknown AFD role {afd_config.role!r}")

    min_size = min(attention_size, ffn_size)
    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        dst = local_attention_rank
        while dst < ffn_size:
            destinations.append(dst)
            dst += min_size

    return _CAMP2PTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        dp_metadata_destinations=tuple(destinations),
    )


def _send_object(obj: Any, *, dst: int, group: Any) -> None:
    torch = _torch()
    object_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
    size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long, device="cpu")
    torch.distributed.send(size_tensor, dst=dst, group=group)
    torch.distributed.send(object_tensor, dst=dst, group=group)


def _recv_object(*, src: int, group: Any) -> Any:
    torch = _torch()
    size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
    rank_size = torch.distributed.recv(size_tensor, src=src, group=group)
    object_tensor = torch.empty(
        int(size_tensor.item()),
        dtype=torch.uint8,
        device="cpu",
    )
    rank_object = torch.distributed.recv(object_tensor, src=src, group=group)
    if rank_object != rank_size:
        raise RuntimeError("received CAMP2P object fragments from different ranks")
    return pickle.loads(object_tensor.cpu().numpy().tobytes())


def _num_tokens_for_ffn_rank(
    dp_metadata_list: dict[int, Any],
    stage_idx: int,
    *,
    ffn_rank: int,
    attention_size: int,
    ffn_size: int,
    fallback: int,
) -> int:
    dp_metadata = dp_metadata_list.get(int(stage_idx))
    if dp_metadata is None:
        return max(1, int(fallback))
    token_counts = dp_metadata.num_tokens_across_dp_cpu
    counts = _to_int_list(token_counts)
    if len(counts) < attention_size:
        return max(1, int(fallback))
    if attention_size >= ffn_size and attention_size % ffn_size == 0:
        group_size = attention_size // ffn_size
        start_idx = int(ffn_rank) * group_size
        end_idx = start_idx + group_size
        return max(1, sum(counts[start_idx:end_idx]))
    return max(1, int(fallback))


def _resolve_role_rank(vllm_config: object, afd_config: AFDConfig) -> int:
    del vllm_config
    return int(afd_config.afd_server_rank)


def _resolve_aiv_num(afd_config: AFDConfig) -> int:
    extra = afd_config.extra_config
    key = "attn_core_num" if afd_config.role == "attention" else "ffn_core_num"
    value = extra.get(key, extra.get("core_num", 8))
    return max(1, int(value or 8))


def _resolve_hidden_size(vllm_config: object) -> int:
    return _resolve_int_attr(vllm_config, "hidden_size", default=1)


def _resolve_num_ubatches(vllm_config: object, afd_config: AFDConfig) -> int:
    if "num_ubatches" in afd_config.extra_config:
        return max(1, int(afd_config.extra_config["num_ubatches"]))
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return max(1, int(getattr(parallel_config, "num_ubatches", 1)))


def _resolve_max_num_reqs(vllm_config: object) -> int:
    scheduler_config = vllm_config.scheduler_config
    return int(scheduler_config.max_num_seqs)


def _resolve_int_attr(vllm_config: object, name: str, *, default: int) -> int:
    del default
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    if name == "hidden_size":
        return int(hf_config.hidden_size)
    if name == "num_experts_per_tok":
        return int(hf_config.num_experts_per_tok)
    if name == "n_routed_experts":
        return int(hf_config.n_routed_experts)
    if name == "n_shared_experts":
        return int(hf_config.n_shared_experts)
    raise KeyError(name)


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, (int, float)):
        value = [value]
    elif isinstance(value, (list, tuple)):
        pass
    else:
        value = value.tolist()
    return [int(item) for item in value]


def _ensure_connector_data(
    metadata: AFDConnectorMetadata,
) -> CAMP2PAFDConnectorMetadata:
    data = metadata.connector_data
    if not isinstance(data, CAMP2PAFDConnectorMetadata):
        raise RuntimeError("CAMP2P metadata is missing connector_data")
    return data


def _set_forward_context_connector_data(data: CAMP2PAFDConnectorMetadata) -> None:
    from vllm.forward_context import get_forward_context

    get_forward_context().cam_afdconnector_data = data


def _get_forward_context_connector_data() -> CAMP2PAFDConnectorMetadata | None:
    from vllm.forward_context import get_forward_context

    data = getattr(get_forward_context(), "cam_afdconnector_data", None)
    if data is None:
        return None
    if not isinstance(data, CAMP2PAFDConnectorMetadata):
        raise TypeError("forward_context.cam_afdconnector_data has wrong type")
    return data


def _register_camp2p_custom_ops() -> None:
    global _CAMP2P_CUSTOM_OPS_REGISTERED
    if _CAMP2P_CUSTOM_OPS_REGISTERED:
        return

    import torch
    from typing import Optional
    from vllm.utils.torch_utils import direct_register_custom_op

    def send_attn_output_impl(
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        group_ep: str,
        aiv_num: int,
        compute_gate: int,
    ) -> torch.Tensor:
        connector_data = _get_forward_context_connector_data()
        if connector_data is None:
            connector_data = CAMP2PAFDConnectorMetadata()
        connector_data.batch_size = int(batch_size)
        connector_data.h = int(hidden_size)
        connector_data.k = int(topk)
        connector_data.aiv_num = int(aiv_num)
        connector_data.group_ep = group_ep

        if _camp2p_stub_io_enabled():
            connector_data.atten_batch_size = _stub_atten_batch_size(
                connector_data.batch_size,
            )
            connector_data.x_active_mask = _stub_x_active_mask(
                connector_data.batch_size,
            )
            _set_forward_context_connector_data(connector_data)
            return hidden_states

        outputs = _afd_ascend_ops().a2e(
            hidden_states,
            topk_ids,
            topk_weights,
            connector_data.batch_size,
            connector_data.h,
            connector_data.k,
            int(ffn_size),
            int(attn_size),
            int(world_rank),
            group_ep,
            connector_data.aiv_num,
            int(compute_gate),
        )
        connector_data.handle = list(outputs[:5])
        connector_data.atten_batch_size = outputs[3]
        connector_data.x_active_mask = outputs[4]
        _set_forward_context_connector_data(connector_data)
        return hidden_states

    def send_attn_output_fake_impl(
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        group_ep: str,
        aiv_num: int,
        compute_gate: int,
    ) -> torch.Tensor:
        del (
            topk_weights,
            topk_ids,
            batch_size,
            hidden_size,
            topk,
            ffn_size,
            attn_size,
            world_rank,
            group_ep,
            aiv_num,
            compute_gate,
        )
        return hidden_states

    def recv_ffn_output_impl(
        ref_tensor: torch.Tensor,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        group_ep: str,
        aiv_num: int,
    ) -> torch.Tensor:
        connector_data = _get_forward_context_connector_data()
        if connector_data is None or connector_data.atten_batch_size is None:
            raise RuntimeError("CAMP2P Attention side is missing A2E handle data")
        connector_data.batch_size = int(batch_size)
        connector_data.h = int(hidden_size)
        connector_data.k = int(topk)
        connector_data.aiv_num = int(aiv_num)
        connector_data.group_ep = group_ep
        if _camp2p_stub_io_enabled():
            return ref_tensor
        return _afd_ascend_ops().e2a(
            ref_tensor,
            connector_data.atten_batch_size,
            connector_data.batch_size,
            connector_data.h,
            connector_data.k,
            int(ffn_size),
            int(attn_size),
            int(world_rank),
            group_ep,
            connector_data.aiv_num,
        )

    def recv_ffn_output_fake_impl(
        ref_tensor: torch.Tensor,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        group_ep: str,
        aiv_num: int,
    ) -> torch.Tensor:
        del (
            batch_size,
            hidden_size,
            topk,
            ffn_size,
            attn_size,
            world_rank,
            group_ep,
            aiv_num,
        )
        return ref_tensor

    send_annotations = {
        "hidden_states": torch.Tensor,
        "topk_weights": Optional[torch.Tensor],
        "topk_ids": Optional[torch.Tensor],
        "batch_size": int,
        "hidden_size": int,
        "topk": int,
        "ffn_size": int,
        "attn_size": int,
        "world_rank": int,
        "group_ep": str,
        "aiv_num": int,
        "compute_gate": int,
        "return": torch.Tensor,
    }
    recv_annotations = {
        "ref_tensor": torch.Tensor,
        "batch_size": int,
        "hidden_size": int,
        "topk": int,
        "ffn_size": int,
        "attn_size": int,
        "world_rank": int,
        "group_ep": str,
        "aiv_num": int,
        "return": torch.Tensor,
    }
    send_attn_output_impl.__annotations__ = send_annotations
    send_attn_output_fake_impl.__annotations__ = send_annotations
    recv_ffn_output_impl.__annotations__ = recv_annotations
    recv_ffn_output_fake_impl.__annotations__ = recv_annotations

    try:
        direct_register_custom_op(
            op_name="afd_camp2p_send_attn_output",
            op_func=send_attn_output_impl,
            mutates_args=[],
            fake_impl=send_attn_output_fake_impl,
            dispatch_key="PrivateUse1",
        )
        direct_register_custom_op(
            op_name="afd_camp2p_recv_ffn_output",
            op_func=recv_ffn_output_impl,
            mutates_args=[],
            fake_impl=recv_ffn_output_fake_impl,
            dispatch_key="PrivateUse1",
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        duplicate = any(
            marker in message
            for marker in ("already", "duplicate", "same name", "defined")
        )
        if not duplicate:
            raise
    _CAMP2P_CUSTOM_OPS_REGISTERED = True


def _is_torch_compiling() -> bool:
    torch = _torch()
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and dynamo.is_compiling())


def _configure_camp2p_stub_io(afd_config: AFDConfig) -> None:
    global _CAMP2P_STUB_IO_ENABLED
    extra = afd_config.extra_config or {}
    _CAMP2P_STUB_IO_ENABLED = any(
        _truthy(extra.get(key)) for key in _CAMP2P_STUB_IO_EXTRA_KEYS
    )
    if _camp2p_stub_io_enabled():
        _print_camp2p_stub_io_notice()


def _camp2p_stub_io_enabled() -> bool:
    return _CAMP2P_STUB_IO_ENABLED or _truthy(os.getenv(_CAMP2P_STUB_IO_ENV))


def _print_camp2p_stub_io_notice() -> None:
    global _CAMP2P_STUB_IO_NOTICE_PRINTED
    if _CAMP2P_STUB_IO_NOTICE_PRINTED:
        return
    print(
        "AFD_CAMP2P_STUB_IO=1: CAMP2P tensor data-path send/recv "
        "calls are stubbed for torch.compile diagnostics.",
        flush=True,
    )
    _CAMP2P_STUB_IO_NOTICE_PRINTED = True


def _stub_atten_batch_size(batch_size: int) -> Any:
    torch = _torch()
    return torch.tensor([max(1, int(batch_size))], dtype=torch.int32, device="npu")


def _stub_x_active_mask(batch_size: int) -> Any:
    torch = _torch()
    return torch.ones(max(1, int(batch_size)), dtype=torch.bool, device="npu")


def _stub_recv_attn_outputs(
    connector_data: CAMP2PAFDConnectorMetadata,
) -> tuple[Any, Any, Any, Any, Any]:
    torch = _torch()
    batch_size = max(1, int(connector_data.batch_size))
    hidden_size = max(1, int(connector_data.h))
    topk = max(1, int(connector_data.k))
    hidden_states = torch.zeros(
        (batch_size, hidden_size),
        dtype=torch.bfloat16,
        device="npu",
    )
    topk_ids = torch.zeros((batch_size, topk), dtype=torch.int32, device="npu")
    topk_weights = torch.zeros(
        (batch_size, topk),
        dtype=torch.float32,
        device="npu",
    )
    atten_batch_size = _stub_atten_batch_size(batch_size)
    x_active_mask = _stub_x_active_mask(batch_size)
    return hidden_states, topk_ids, topk_weights, atten_batch_size, x_active_mask


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _empty_npu_tensor(*, dtype_name: str) -> Any:
    torch = _torch()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }[dtype_name]
    return torch.tensor([], dtype=dtype, device="npu")


def _afd_ascend_ops() -> Any:
    torch = _torch()
    return torch.ops.afd_ascend


def _hccl_comm_name(group: Any, rank: int) -> str:
    torch = _torch()
    backend = group._get_backend(torch.device("npu"))
    return str(backend.get_hccl_comm_name(int(rank)))


def _torch() -> Any:
    import torch

    return torch


__all__ = [
    "CAMP2PAFDConnector",
    "CAMP2PAFDConnectorData",
    "CAMP2PAFDConnectorMetadata",
    "build_camp2p_topology",
]

CAMP2PAFDConnectorData = CAMP2PAFDConnectorMetadata
