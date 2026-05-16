from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from afd_plugin.connectors import AFDConnectorMetadata
from afd_plugin.runtime.ffn_model_runner import GPUFFNModelRunner
from afd_plugin.runtime.ffn_worker import AFDFFNWorker


class _FakeConnector:
    def __init__(self):
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.dp_metadata_updates = []

    def update_state_from_dp_metadata(self, dp_metadata_list, is_graph_capturing):
        self.dp_metadata_updates.append((dict(dp_metadata_list), is_graph_capturing))

    def recv_attn_output(self):
        return self.attn_outputs.popleft()

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


def test_ffn_runner_executes_model_compute_ffn_output():
    runner = object.__new__(GPUFFNModelRunner)
    runner.connector = _FakeConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    metadata = _metadata()
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: "dp"})

    assert runner.connector.dp_metadata_updates == [({0: "dp"}, False)]
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]
    assert metadata.layer_idx == 0


def test_ffn_runner_passthrough_without_model_compute_hook():
    runner = object.__new__(GPUFFNModelRunner)
    runner.connector = _FakeConnector()
    runner.model = SimpleNamespace()
    runner.num_layers = 1
    metadata = _metadata()
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: "dp"})

    assert runner.connector.ffn_outputs == [("hidden", metadata)]


def test_ffn_runner_requires_dp_metadata_list():
    runner = object.__new__(GPUFFNModelRunner)

    with pytest.raises(RuntimeError, match="requires dp_metadata_list"):
        runner.execute_model()


def test_ffn_worker_scheduler_execute_model_fails_fast():
    worker = object.__new__(AFDFFNWorker)

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())
