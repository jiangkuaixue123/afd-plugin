from __future__ import annotations

import pytest

from afd_plugin.connectors import AFDConnectorFactory, AFDConnectorMetadata


def test_dummy_connector_is_not_registered():
    with pytest.raises(ValueError, match="unsupported AFD connector type"):
        AFDConnectorFactory.get_connector_class("dummy")


def test_connector_metadata_validates_sequence_lengths():
    with pytest.raises(ValueError, match="sequence lengths"):
        AFDConnectorMetadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[0],
            dtype="bf16",
            device="cuda:0",
        )

