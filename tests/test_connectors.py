from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDConnectorFactory, AFDConnectorMetadata


class _FakeTensor:
    shape = (3, 8)

    def new_zeros(self, shape):
        return ("zeros", shape)


def test_dummy_connector_records_attention_events_and_returns_zero_like():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(),
        AFDConfig(enabled=True, role="attention"),
    )
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=3,
        dtype="bf16",
        device="cuda:0",
    )

    connector.send_attn_output(_FakeTensor(), metadata)

    assert connector.recv_ffn_output() == ("zeros", (3, 8))


def test_dummy_connector_round_trips_attention_and_ffn_outputs():
    afd_config = AFDConfig(enabled=True, role="attention")
    attention_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(),
        afd_config,
    )
    ffn_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(),
        AFDConfig(
            enabled=True,
            role="ffn",
        ),
    )
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=3,
        dtype="bf16",
        device="cpu",
    )

    attention_connector.send_dp_metadata_list({0: "dp"})
    attention_connector.send_attn_output(_FakeTensor(), metadata)
    dp_metadata_list, is_attn_graph_capturing, is_warmup = (
        ffn_connector.recv_dp_metadata_list(timeout_ms=100)
    )
    hidden_states, recv_metadata = ffn_connector.recv_attn_output(timeout_ms=100)
    ffn_connector.send_ffn_output(("ffn", hidden_states.shape), recv_metadata)

    assert dp_metadata_list == {0: "dp"}
    assert is_attn_graph_capturing is False
    assert is_warmup is False
    assert recv_metadata is metadata
    assert attention_connector.recv_ffn_output(timeout_ms=100) == ("ffn", (3, 8))


def test_connector_metadata_validates_sequence_lengths():
    with pytest.raises(ValueError, match="sequence lengths"):
        AFDConnectorMetadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[0],
            dtype="bf16",
            device="cuda:0",
        )


def test_dummy_connector_matches_out_of_order_ubatch_responses():
    attention_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(),
        AFDConfig(enabled=True, role="attention"),
    )
    ffn_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(),
        AFDConfig(enabled=True, role="ffn"),
    )
    metadata_0 = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=3,
        dtype="bf16",
        device="cpu",
        ubatch_idx=0,
        transaction_id="batch-1",
    )
    metadata_1 = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=1,
        seq_len=3,
        dtype="bf16",
        device="cpu",
        ubatch_idx=1,
        transaction_id="batch-1",
    )

    attention_connector.send_attn_output(_FakeTensor(), metadata_0)
    attention_connector.send_attn_output(_FakeTensor(), metadata_1)
    hidden_1, recv_metadata_1 = ffn_connector.recv_attn_output(
        timeout_ms=100,
        ubatch_idx=1,
    )
    hidden_0, recv_metadata_0 = ffn_connector.recv_attn_output(
        timeout_ms=100,
        ubatch_idx=0,
    )
    ffn_connector.send_ffn_output(("ffn-1", hidden_1.shape), recv_metadata_1)
    ffn_connector.send_ffn_output(("ffn-0", hidden_0.shape), recv_metadata_0)

    assert recv_metadata_1.message_key.ubatch_idx == 1
    assert recv_metadata_0.message_key.ubatch_idx == 0
    assert attention_connector.recv_ffn_output(timeout_ms=100, ubatch_idx=0) == (
        "ffn-0",
        (3, 8),
    )
    assert attention_connector.recv_ffn_output(timeout_ms=100, ubatch_idx=1) == (
        "ffn-1",
        (3, 8),
    )
