from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from afd_plugin.connectors import AFDConnectorMetadata
from afd_plugin.runtime.ffn_model_runner import (
    GPUFFNModelRunner,
    _set_moe_layer_index,
)
from afd_plugin.runtime.ffn_worker import AFDFFNWorker


class _FakeConnector:
    def __init__(self):
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.dp_metadata_updates = []

    def update_state_from_dp_metadata(self, dp_metadata_list, is_graph_capturing):
        self.dp_metadata_updates.append((dict(dp_metadata_list), is_graph_capturing))

    def recv_attn_output(self, ubatch_idx=None):
        if ubatch_idx is None:
            return self.attn_outputs.popleft()
        for item in tuple(self.attn_outputs):
            if getattr(item[1], "ubatch_idx", item[1].stage_idx) == ubatch_idx:
                self.attn_outputs.remove(item)
                return item
        raise IndexError(ubatch_idx)

    def send_ffn_output(self, ffn_output, metadata):
        self.ffn_outputs.append((ffn_output, metadata))


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx):
        return f"ffn({hidden_states}, layer={layer_idx})"


def _metadata():
    return AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
        dtype="bf16",
        device="cpu",
    )


def _metadata_for_stage(stage_idx):
    return AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=stage_idx,
        seq_len=1,
        dtype="bf16",
        device="cpu",
        ubatch_idx=stage_idx,
    )


def _runner_with_connector_and_model(model, *, num_layers=1):
    runner = object.__new__(GPUFFNModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.connector = _FakeConnector()
    runner.model = model
    runner.num_layers = num_layers
    return runner


def test_ffn_runner_executes_model_compute_ffn_output():
    runner = _runner_with_connector_and_model(_FakeModel())
    metadata = _metadata()
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: "dp"})

    assert runner.connector.dp_metadata_updates == [({0: "dp"}, False)]
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]
    assert metadata.layer_idx == 0


def test_ffn_runner_passthrough_without_model_compute_hook():
    runner = _runner_with_connector_and_model(SimpleNamespace())
    metadata = _metadata()
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: "dp"})

    assert runner.connector.ffn_outputs == [("hidden", metadata)]


def test_ffn_runner_processes_each_ubatch_for_each_layer():
    runner = _runner_with_connector_and_model(_FakeModel(), num_layers=2)
    metadata_0_layer_0 = _metadata_for_stage(0)
    metadata_1_layer_0 = _metadata_for_stage(1)
    metadata_0_layer_1 = _metadata_for_stage(0)
    metadata_1_layer_1 = _metadata_for_stage(1)
    runner.connector.attn_outputs.extend(
        [
            ("hidden-1-l0", metadata_1_layer_0),
            ("hidden-0-l0", metadata_0_layer_0),
            ("hidden-1-l1", metadata_1_layer_1),
            ("hidden-0-l1", metadata_0_layer_1),
        ],
    )

    runner.execute_model(dp_metadata_list={0: "dp0", 1: "dp1"})

    assert runner.connector.ffn_outputs == [
        ("ffn(hidden-0-l0, layer=0)", metadata_0_layer_0),
        ("ffn(hidden-1-l0, layer=0)", metadata_1_layer_0),
        ("ffn(hidden-0-l1, layer=1)", metadata_0_layer_1),
        ("ffn(hidden-1-l1, layer=1)", metadata_1_layer_1),
    ]


def test_ffn_runner_requires_dp_metadata_list():
    runner = object.__new__(GPUFFNModelRunner)

    with pytest.raises(RuntimeError, match="requires dp_metadata_list"):
        runner.execute_model()


def test_set_moe_layer_index_resets_for_current_layer():
    forward_context = SimpleNamespace(
        all_moe_layers=[
            "model.layers.1.mlp.experts",
            "model.layers.2.mlp.experts",
            "model.layers.3.mlp.experts",
        ],
        moe_layer_index=99,
    )

    _set_moe_layer_index(forward_context, 2)

    assert forward_context.moe_layer_index == 1


def test_ffn_worker_scheduler_execute_model_fails_fast():
    worker = object.__new__(AFDFFNWorker)

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())
