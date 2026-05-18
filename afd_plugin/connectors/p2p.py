# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""P2P AFD connector migrated from the original in-tree AFD branch.

The module is intentionally CPU-safe at import time. CUDA, torch.distributed,
PyNCCL, and vLLM runtime imports are delayed until connector initialization or
actual send/recv calls.
"""

from __future__ import annotations

import pickle
from datetime import timedelta
from typing import Any, NamedTuple

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDConnectorMetadata
from afd_plugin.distributed import (
    DefaultProcessGroupSwitcher,
    build_rank_mapping,
    init_afd_process_group,
)
from afd_plugin.tracing import afd_trace, dp_metadata_summary, tensor_summary


class _TensorMetadata(NamedTuple):
    device: Any
    dtype: Any
    size: Any


class P2PAFDConnector(AFDConnectorBase):
    """NCCL-backed Attention <-> FFN connector for Phase 4.

    The first supported topology matches the original AFD branch: FFN ranks are
    placed before Attention ranks in the AFD world, and each FFN rank owns a
    subgroup with one or more consecutive Attention ranks.
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
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            role_rank = int(parallel_config.data_parallel_rank)
        else:
            role_rank = int(afd_config.afd_server_rank)
        self.mapping = build_rank_mapping(
            afd_config,
            role_rank=role_rank,
        )
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.num_hidden_layers = int(
            vllm_config.model_config.hf_config.num_hidden_layers,
        )
        self.hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.dp_metadata_list: dict[int, Any] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self._tensor_metadata_list: dict[int, _TensorMetadata] = {}
        self._recv_attn_buffers: dict[tuple[int, int, tuple[int, ...]], Any] = {}
        self.a2e_group: Any | None = None
        self.e2a_group: Any | None = None
        self.p2p_pg: Any | None = None
        self.a2e_pynccl: Any | None = None
        self.e2a_pynccl: Any | None = None
        afd_trace(
            "p2p_init",
            role=afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            local_rank=self.local_rank,
            attn_size=self.attn_size,
            ffn_size=self.ffn_size,
            group_size=len(self.mapping.subgroup_ranks),
            group_rank=self.mapping.rank_in_subgroup,
            dp_dsts=self.dst_list,
        )

    def close(self) -> None:
        for communicator_name in ("a2e_pynccl", "e2a_pynccl"):
            communicator = getattr(self, communicator_name, None)
            shutdown = getattr(communicator, "shutdown", None)
            if callable(shutdown):
                shutdown()
            setattr(self, communicator_name, None)
        self._initialized = False

    def init_afd_connector(self) -> None:
        if self._initialized:
            return

        from torch.distributed.distributed_c10d import _get_default_group
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.ffn_size + self.attn_size,
            rank=self.world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )

        with DefaultProcessGroupSwitcher(_get_default_group(), afd_pg):
            base_port = self.afd_config.port
            self.a2e_group = StatelessProcessGroup.create(
                host=self.afd_config.host,
                port=base_port + self.mapping.subgroup_index + 1,
                rank=self.mapping.rank_in_subgroup,
                world_size=len(self.mapping.subgroup_ranks),
            )
            self.e2a_group = self.a2e_group
            self.rank_in_group = self.mapping.rank_in_subgroup
            self.group_size = len(self.mapping.subgroup_ranks)
            self.a2e_pynccl = PyNcclCommunicator(
                group=self.a2e_group,
                device=self.local_rank,
            )
            self.e2a_pynccl = PyNcclCommunicator(
                group=self.e2a_group,
                device=self.local_rank,
            )

        if self.mapping.participates_in_dp_metadata_group:
            self.p2p_pg = init_afd_process_group(
                backend="nccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        import torch

        self.dp_metadata_list = dp_metadata_list
        self.is_graph_capturing = is_graph_capturing
        self.is_warmup = is_warmup
        self._tensor_metadata_list = {}
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.vllm_config.model_config.dtype
        dp_rank = int(self.vllm_config.parallel_config.data_parallel_rank)
        for stage_idx, dp_metadata in dp_metadata_list.items():
            num_tokens = _num_tokens_for_dp_rank(dp_metadata, dp_rank)
            self._tensor_metadata_list[int(stage_idx)] = _TensorMetadata(
                device,
                dtype,
                torch.Size([num_tokens, self.hidden_size]),
            )
        afd_trace(
            "p2p_update_state_from_dp_metadata",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            dp_rank=dp_rank,
            stages=sorted(int(stage_idx) for stage_idx in dp_metadata_list),
            dp_metadata=dp_metadata_summary(dp_metadata_list),
            tensor_metadata={
                stage_idx: tuple(metadata.size)
                for stage_idx, metadata in self._tensor_metadata_list.items()
            },
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )

        if (
            self.afd_config.role == "ffn"
            and not self.vllm_config.model_config.enforce_eager
        ):
            for stage_idx, tensor_metadata in self._tensor_metadata_list.items():
                for src_rank in range(1, self.group_size):
                    buffer_key = (stage_idx, src_rank, tuple(tensor_metadata.size))
                    existing = self._recv_attn_buffers.get(buffer_key)
                    if _matches_tensor_metadata(existing, tensor_metadata):
                        continue
                    self._recv_attn_buffers[buffer_key] = torch.empty(
                        tuple(tensor_metadata.size),
                        dtype=tensor_metadata.dtype,
                        device=tensor_metadata.device,
                    )

    def is_attn_top_min_size_rank(self, world_rank: int) -> bool:
        return self.ffn_size <= world_rank < self.ffn_size + self.min_size

    def send_dp_metadata_list(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        import torch

        if self.p2p_pg is None:
            afd_trace(
                "p2p_send_dp_metadata_skip_no_group",
                role=self.afd_config.role,
                world_rank=self.world_rank,
                p2p_rank=self.p2p_rank,
                dp_metadata=dp_metadata_summary(dp_metadata_list),
            )
            return
        device = torch.device(f"cuda:{self.local_rank}")
        send_data = (dp_metadata_list, is_graph_capturing, is_warmup)
        object_bytes = pickle.dumps(send_data)
        object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
        object_tensor = object_tensor_cpu.to(device)
        size_tensor = torch.tensor(
            [object_tensor.numel()],
            dtype=torch.long,
            device=device,
        )

        for dst in self.dst_list:
            afd_trace(
                "p2p_send_dp_metadata_begin",
                role=self.afd_config.role,
                world_rank=self.world_rank,
                p2p_rank=self.p2p_rank,
                dst=dst,
                bytes=int(size_tensor.item()),
                dp_metadata=dp_metadata_summary(dp_metadata_list),
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )
            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor, dst=dst, group=self.p2p_pg)
            afd_trace(
                "p2p_send_dp_metadata_done",
                role=self.afd_config.role,
                world_rank=self.world_rank,
                p2p_rank=self.p2p_rank,
                dst=dst,
            )

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[dict[int, Any], bool, bool]:
        del timeout_ms
        import torch

        if self.p2p_pg is None:
            raise RuntimeError("P2P DP metadata process group is not initialized")

        src = self.p2p_rank % self.min_size + self.ffn_size
        device = torch.device(f"cuda:{self.local_rank}")
        size_tensor = torch.empty(1, dtype=torch.long, device=device)
        afd_trace(
            "p2p_recv_dp_metadata_size_begin",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            src=src,
        )
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        object_tensor = torch.empty(
            int(size_tensor.item()),
            dtype=torch.uint8,
            device=device,
        )
        afd_trace(
            "p2p_recv_dp_metadata_body_begin",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            src=src,
            bytes=int(size_tensor.item()),
        )
        rank_object = torch.distributed.recv(object_tensor, src=src, group=self.p2p_pg)
        if rank_object != rank_size:
            raise RuntimeError("received AFD metadata fragments from different ranks")

        obj = pickle.loads(object_tensor.cpu().numpy().tobytes())
        if len(obj) == 3:
            data, is_graph_capturing, is_warmup = obj
        else:
            data, is_graph_capturing = obj
            is_warmup = False
        afd_trace(
            "p2p_recv_dp_metadata_done",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            src=src,
            dp_metadata=dp_metadata_summary(data),
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        return data, is_graph_capturing, is_warmup

    def send_attn_output(
        self,
        hidden_states: Any,
        metadata: AFDConnectorMetadata,
    ) -> None:
        if not metadata.validate_tensor_shape(tuple(hidden_states.shape)):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"AFD metadata token count {metadata.total_tokens}",
            )
        metadata.direction = "attention_to_ffn"
        afd_trace(
            "p2p_send_attn_output",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            layer_idx=metadata.layer_idx,
            stage_idx=metadata.stage_idx,
            ubatch_idx=metadata.ubatch_idx,
            tensor=tensor_summary(hidden_states),
        )
        self._send_hidden_states(hidden_states, 0, self.a2e_group, self.a2e_pynccl)

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        del handle
        ref_tensor = kwargs.get("ref_tensor")
        ubatch_idx = kwargs.get("ubatch_idx")
        if ubatch_idx is None:
            ubatch_idx = self._current_ubatch_idx()
        afd_trace(
            "p2p_recv_ffn_output_begin",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            ubatch_idx=ubatch_idx,
            expected_shape=tuple(self._tensor_metadata_list[int(ubatch_idx)].size),
        )
        output = self._recv_hidden_states(
            0,
            self.e2a_group,
            self.e2a_pynccl,
            self._tensor_metadata_list[int(ubatch_idx)],
            ref_tensor=ref_tensor,
        )
        afd_trace(
            "p2p_recv_ffn_output_done",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            ubatch_idx=ubatch_idx,
            tensor=tensor_summary(output),
        )
        return output

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
    ) -> tuple[Any, AFDConnectorMetadata]:
        del timeout_ms
        import torch

        ubatch_idx = 0 if ubatch_idx is None else int(ubatch_idx)
        tensor_metadata = self._tensor_metadata_list[ubatch_idx]
        hidden_states_list: list[Any] = []
        afd_trace(
            "p2p_recv_attn_output_begin",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            ubatch_idx=ubatch_idx,
            expected_shape=tuple(tensor_metadata.size),
            peer_count=max(self.group_size - 1, 0),
        )

        for src in range(1, self.group_size):
            ref_tensor = None
            if not self.vllm_config.model_config.enforce_eager:
                ref_tensor = self._recv_attn_buffers.get(
                    (ubatch_idx, src, tuple(tensor_metadata.size)),
                )
            hidden_states_list.append(
                self._recv_hidden_states(
                    src,
                    self.a2e_group,
                    self.a2e_pynccl,
                    tensor_metadata,
                    ref_tensor=ref_tensor,
                ),
            )
            afd_trace(
                "p2p_recv_attn_output_part_done",
                role=self.afd_config.role,
                world_rank=self.world_rank,
                p2p_rank=self.p2p_rank,
                ubatch_idx=ubatch_idx,
                src=src,
                tensor=tensor_summary(hidden_states_list[-1]),
            )

        if not hidden_states_list:
            raise RuntimeError("P2P FFN rank has no Attention peers")
        hidden_states = (
            torch.cat(hidden_states_list, dim=0)
            if len(hidden_states_list) > 1
            else hidden_states_list[0]
        )
        metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=ubatch_idx,
            seq_lens=[int(tensor.shape[0]) for tensor in hidden_states_list],
            dtype=tensor_metadata.dtype,
            device=tensor_metadata.device,
            ubatch_idx=ubatch_idx,
        )
        afd_trace(
            "p2p_recv_attn_output_done",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            ubatch_idx=ubatch_idx,
            tensor=tensor_summary(hidden_states),
            seq_lens=metadata.seq_lens,
        )
        return hidden_states, metadata

    def send_ffn_output(
        self,
        ffn_output: Any,
        metadata: AFDConnectorMetadata,
    ) -> None:
        if not metadata.validate_tensor_shape(tuple(ffn_output.shape)):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match metadata",
            )
        metadata.direction = "ffn_to_attention"
        afd_trace(
            "p2p_send_ffn_output",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            layer_idx=metadata.layer_idx,
            stage_idx=metadata.stage_idx,
            ubatch_idx=metadata.ubatch_idx,
            ratio=self.ratio,
            tensor=tensor_summary(ffn_output),
        )
        if self.ratio == 1:
            self._send_hidden_states(ffn_output, 1, self.e2a_group, self.e2a_pynccl)
            return

        split_sizes = metadata.seq_lens
        if len(split_sizes) != self.ratio:
            total_tokens = int(ffn_output.shape[0])
            if total_tokens % self.ratio != 0:
                raise ValueError(
                    "cannot evenly split FFN output across Attention peers: "
                    f"tokens={total_tokens}, ratio={self.ratio}",
                )
            tokens_per_attention = total_tokens // self.ratio
            split_sizes = [tokens_per_attention] * self.ratio

        start = 0
        for dst, token_count in zip(
            range(1, self.group_size),
            split_sizes,
            strict=False,
        ):
            end = start + token_count
            self._send_hidden_states(
                ffn_output[start:end],
                dst,
                self.e2a_group,
                self.e2a_pynccl,
            )
            start = end

    def _send_hidden_states(
        self,
        hidden_states: Any,
        dst: int,
        process_group: Any,
        communicator: Any,
    ) -> None:
        if process_group is None or communicator is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
            return
        if dst >= process_group.world_size:
            raise ValueError(f"invalid P2P destination rank {dst}")
        if getattr(hidden_states, "is_cpu", False):
            raise ValueError("P2P hidden states must be on GPU")

        import torch

        afd_trace(
            "p2p_send_hidden_states_begin",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            group_rank=getattr(process_group, "rank", None),
            group_world_size=getattr(process_group, "world_size", None),
            dst=dst,
            tensor=tensor_summary(hidden_states),
        )
        communicator.send(
            hidden_states,
            dst,
            stream=torch.cuda.current_stream(hidden_states.device),
        )
        afd_trace(
            "p2p_send_hidden_states_done",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            dst=dst,
        )

    def _recv_hidden_states(
        self,
        src: int,
        process_group: Any,
        communicator: Any,
        tensor_metadata: _TensorMetadata,
        *,
        ref_tensor: Any | None = None,
    ) -> Any:
        if process_group is None or communicator is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
            return ref_tensor
        if src >= process_group.world_size:
            raise ValueError(f"invalid P2P source rank {src}")

        import torch

        if _matches_tensor_metadata(ref_tensor, tensor_metadata):
            hidden_states = ref_tensor
        else:
            hidden_states = torch.empty(
                tuple(tensor_metadata.size),
                dtype=tensor_metadata.dtype,
                device=tensor_metadata.device,
            )
        afd_trace(
            "p2p_recv_hidden_states_begin",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            group_rank=getattr(process_group, "rank", None),
            group_world_size=getattr(process_group, "world_size", None),
            src=src,
            tensor=tensor_summary(hidden_states),
        )
        communicator.recv(
            hidden_states,
            src,
            stream=torch.cuda.current_stream(hidden_states.device),
        )
        afd_trace(
            "p2p_recv_hidden_states_done",
            role=self.afd_config.role,
            world_rank=self.world_rank,
            p2p_rank=self.p2p_rank,
            src=src,
            tensor=tensor_summary(hidden_states),
        )
        return hidden_states

    @staticmethod
    def _current_ubatch_idx() -> int:
        try:
            from vllm.forward_context import get_forward_context

            forward_context = get_forward_context()
            afd_metadata = forward_context.additional_kwargs["afd_metadata"]
            return int(afd_metadata.ubatch_idx)
        except Exception:
            return 0


def _matches_tensor_metadata(value: Any, tensor_metadata: _TensorMetadata) -> bool:
    if value is None:
        return False
    return (
        getattr(value, "shape", None) == tensor_metadata.size
        and getattr(value, "dtype", None) == tensor_metadata.dtype
        and getattr(value, "device", None) == tensor_metadata.device
    )


def _num_tokens_for_dp_rank(dp_metadata: Any, dp_rank: int) -> int:
    token_count = dp_metadata.num_tokens_across_dp_cpu[dp_rank]
    item = getattr(token_count, "item", None)
    return int(item() if callable(item) else token_count)


__all__ = ["P2PAFDConnector"]
