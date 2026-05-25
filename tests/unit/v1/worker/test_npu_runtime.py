from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from afd_plugin.compat.ascend import fail_if_unsupported_npu_afd_features
from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDConnectorFactory, AFDConnectorMetadata
from afd_plugin.v1.worker.ascend.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.v1.worker.ascend.ffn_model_runner import AFDNPUFFNModelRunner
from afd_plugin.v1.worker.ascend.ffn_worker import AFDNPUFFNWorker


class _RecordingConnector:
    world_rank = 0

    def __init__(self):
        self.dp_metadata_updates = []
        self.sent_dp_metadata_lists = []

    def is_attn_top_min_size_rank(self, world_rank):
        return world_rank == self.world_rank

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        self.dp_metadata_updates.append(
            (dp_metadata_list, is_graph_capturing, is_warmup),
        )

    def send_dp_metadata_list(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        self.sent_dp_metadata_lists.append(
            (dp_metadata_list, is_graph_capturing, is_warmup),
        )


class _FakeFFNConnector:
    def __init__(self):
        self.dp_metadata_list = {}
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.updates = []

    def update_state_from_dp_metadata(self, dp_metadata_list, **kwargs):
        self.dp_metadata_list = dict(dp_metadata_list)
        self.updates.append((dict(dp_metadata_list), kwargs))

    def recv_attn_output(self, metadata=None, ubatch_idx=None):
        del metadata
        for item in tuple(self.attn_outputs):
            if item[1].stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return item
        raise IndexError(ubatch_idx)

    def send_ffn_output(self, ffn_output, metadata, **kwargs):
        self.ffn_outputs.append((ffn_output, metadata, kwargs))

    def close(self):
        return None


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx):
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "use_ubatching": False,
        "num_ubatches": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(*, role="attention", extra_config=None, **parallel_overrides):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "role": role,
                "connector": "npudummyconnector",
                "extra_config": extra_config or {},
            },
        },
        parallel_config=_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(enforce_eager=True),
    )


def test_npu_attention_runner_builds_and_mirrors_metadata():
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.vllm_config = _vllm_config(role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=5),
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert forward_context.afd_metadata is metadata
    assert metadata.afd_tokens_lens == [1]
    assert len(runner.afd_connector.dp_metadata_updates) == 1
    assert len(runner.afd_connector.sent_dp_metadata_lists) == 1


def test_npu_attention_runner_builds_dp_fallback():
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.vllm_config = _vllm_config(role="attention")
    runner.afd_connector = object()
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 7)

    dp_metadata = runner._ensure_dp_metadata(None)

    tokens = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [7]


def test_npu_ffn_runner_executes_eager_ffn_step():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.updates == [
        ({0: runner.connector.dp_metadata_list[0]}, {"is_graph_capturing": False}),
    ]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_passthrough_without_compute_hook():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = SimpleNamespace()
    runner.num_layers = 1
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs[0][0] == "hidden"


def test_npu_ffn_worker_scheduler_execute_model_fails_fast():
    worker = object.__new__(AFDNPUFFNWorker)

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_npu_feature_validation_rejects_unsupported_switches():
    for extra_config, message in [
        ({"compute_gate_on_attention": True}, "compute_gate_on_attention"),
        ({"quant_mode": 1}, "quant_mode=0"),
        ({"is_attn_multistream": True}, "multistream"),
        ({"multistream_info": {"attn_enable": "True"}}, "multistream_info"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            fail_if_unsupported_npu_afd_features(
                _vllm_config(extra_config=extra_config),
            )


def test_npu_feature_validation_rejects_ubatching_and_graph():
    with pytest.raises(RuntimeError, match="ubatching"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(use_ubatching=True, num_ubatches=2),
        )

    config = _vllm_config()
    config.model_config.enforce_eager = False
    with pytest.raises(RuntimeError, match="enforce_eager"):
        fail_if_unsupported_npu_afd_features(config)


def test_npudummyconnector_round_trips_control_and_payload():
    attn = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(role="attention"),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="npudummyconnector",
            port=22345,
        ),
    )
    ffn = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(role="ffn"),
        AFDConfig(
            enabled=True,
            role="ffn",
            connector="npudummyconnector",
            port=22345,
        ),
    )
    attn.init_afd_connector()
    ffn.init_afd_connector()
    dp_metadata = {0: _FakeDPMetadata([2])}

    attn.send_dp_metadata_list(dp_metadata, is_warmup=True)
    received, is_graph_capturing, is_warmup = ffn.recv_dp_metadata_list(
        timeout_ms=10,
    )

    assert received == dp_metadata
    assert not is_graph_capturing
    assert is_warmup

    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    attn.send_attn_output("hidden", metadata)
    assert ffn.recv_attn_output(timeout_ms=10) == ("hidden", metadata)

    ffn.send_ffn_output("ffn", metadata)
    assert attn.recv_ffn_output(timeout_ms=10) == "ffn"
