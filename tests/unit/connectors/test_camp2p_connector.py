from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDConnectorFactory, AFDRecvOutput
from afd_plugin.connectors.ascend.camp2p import (
    CAMP2PAFDConnector,
    CAMP2PAFDConnectorMetadata,
    build_camp2p_topology,
)


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=0,
            ),
        ),
    )


def _afd_config(*, role: str, rank: int = 0):
    return AFDConfig(
        enabled=True,
        connector="camp2pconnector",
        role=role,
        afd_server_rank=rank,
        extra_config={"afd_size": "4A2F"},
    )


def test_camp2p_factory_creates_import_safe_connector():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, CAMP2PAFDConnector)
    assert not connector.is_initialized
    assert connector.max_num_reqs == 8


def test_camp2p_topology_matches_original_rank_layout():
    attn0 = build_camp2p_topology(_afd_config(role="attention", rank=0), 0)
    attn1 = build_camp2p_topology(_afd_config(role="attention", rank=1), 1)
    attn2 = build_camp2p_topology(_afd_config(role="attention", rank=2), 2)
    ffn1 = build_camp2p_topology(_afd_config(role="ffn", rank=1), 1)

    assert (attn0.world_rank, attn0.p2p_rank, attn0.dp_metadata_destinations) == (
        2,
        2,
        (0,),
    )
    assert (attn1.world_rank, attn1.p2p_rank, attn1.dp_metadata_destinations) == (
        3,
        3,
        (1,),
    )
    assert not attn2.participates_in_p2p_group
    assert (ffn1.world_rank, ffn1.p2p_rank) == (1, 1)


def test_camp2p_create_recv_metadata_uses_original_contiguous_af_grouping():
    rank0 = CAMP2PAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn", rank=0),
    )
    rank1 = CAMP2PAFDConnector(
        1,
        1,
        _vllm_config(),
        _afd_config(role="ffn", rank=1),
    )
    dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    metadata0 = rank0.create_recv_metadata(
        dp_metadata_list=dp_metadata_list,
        ubatch_idx=0,
        layer_idx=3,
    )
    metadata1 = rank1.create_recv_metadata(
        dp_metadata_list=dp_metadata_list,
        ubatch_idx=0,
        layer_idx=3,
    )

    assert metadata0.seq_lens == [5]
    assert metadata1.seq_lens == [12]
    assert isinstance(metadata0.connector_data, CAMP2PAFDConnectorMetadata)
    assert metadata0.connector_data.batch_size == 5
    assert metadata0.connector_data.h == 16
    assert metadata0.connector_data.k == 2


def test_camp2p_update_metadata_keeps_original_handle_shape():
    connector = CAMP2PAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn", rank=0),
    )
    metadata = connector.create_recv_metadata(
        dp_metadata_list={0: _FakeDPMetadata([2, 3, 5, 7])},
        ubatch_idx=0,
        layer_idx=0,
    )
    recv_output = AFDRecvOutput(
        hidden_states="hidden",
        metadata=metadata,
        topk_ids="ids",
        topk_weights="weights",
        expand_idx="expand",
        ep_recv_counts="counts",
        atten_batch_size="atten",
    )

    connector.update_metadata(metadata, recv_output)

    assert metadata.connector_data.handle == [
        "ids",
        "weights",
        "expand",
        "counts",
        "atten",
    ]


def test_camp2p_init_fails_cleanly_without_ascend_runtime():
    connector = CAMP2PAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention", rank=0),
    )

    with pytest.raises(RuntimeError, match="AFD Ascend custom ops|torch-npu"):
        connector.init_afd_connector()
