from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig, AFDConfig_from_mapping
from afd_plugin.connectors import AFDConnectorFactory
from afd_plugin.distributed import build_rank_mapping, topology_from_config


def _fake_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype="bf16",
            enforce_eager=True,
            hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )


def test_p2p_connector_is_registered_and_import_is_cpu_safe():
    sys.modules.pop("afd_plugin.connectors.p2p", None)
    sys.modules.pop("vllm.distributed.device_communicators.pynccl", None)

    cls = AFDConnectorFactory.get_connector_class("p2pconnector")

    assert cls.__name__ == "P2PAFDConnector"
    assert "vllm.distributed.device_communicators.pynccl" not in sys.modules


def test_p2p_connector_can_be_constructed_without_runtime_initialization():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="p2pconnector",
            num_attention_servers=2,
            num_ffn_servers=1,
        ),
    )

    assert connector.is_initialized is False
    assert connector.world_rank == 1
    assert connector.dst_list == [0]


def test_p2p_connector_uses_dp_rank_as_role_rank_for_native_dp():
    connector = AFDConnectorFactory.create_connector(
        1,
        1,
        SimpleNamespace(
            model_config=SimpleNamespace(
                dtype="bf16",
                enforce_eager=True,
                hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_rank=1,
            ),
        ),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="p2pconnector",
            num_attention_servers=2,
            num_ffn_servers=2,
        ),
    )

    assert connector.mapping.role_rank == 1
    assert connector.world_rank == 3
    assert connector.p2p_rank == 3


def test_p2p_topology_supports_afd_size_alias():
    config = AFDConfig_from_mapping(
        {
            "enabled": True,
            "role": "ffn",
            "connector": "p2pconnector",
            "extra_config": {"afd_size": "4A2F"},
            "afd_server_rank": 1,
        },
    )

    assert topology_from_config(config) == (4, 2)
    mapping = build_rank_mapping(config)
    assert mapping.world_rank == 1
    assert mapping.subgroup_ranks == (1, 4, 5)
    assert mapping.rank_in_subgroup == 0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "connector": "p2pconnector",
                "num_attention_servers": 1,
                "num_ffn_servers": 2,
            },
            "num_attention_servers >= num_ffn_servers",
        ),
        (
            {
                "connector": "p2pconnector",
                "num_attention_servers": 3,
                "num_ffn_servers": 2,
            },
            "multiple of num_ffn_servers",
        ),
        (
            {
                "connector": "p2pconnector",
                "extra_config": {"afd_size": "not-a-topology"},
            },
            "extra_config\\['afd_size'\\]",
        ),
    ],
)
def test_p2p_topology_validation_errors_are_clear(raw, message):
    with pytest.raises(ValueError, match=message):
        AFDConfig_from_mapping(raw)


def test_p2p_module_exports_connector_class():
    module = importlib.import_module("afd_plugin.connectors.p2p")

    assert module.P2PAFDConnector.__module__ == "afd_plugin.connectors.p2p"
