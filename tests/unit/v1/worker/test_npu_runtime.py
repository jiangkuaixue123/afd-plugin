from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from afd_plugin.compat.ascend import (
    enable_npu_afd_ubatching_if_requested,
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from afd_plugin.connectors import (
    AFDConnectorMetadata,
    AFDRecvOutput,
)
from afd_plugin.v1.worker.ascend.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.v1.worker.ascend.attention_worker import AFDNPUAttentionWorker
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
        self.metadata_updates = []

    def update_state_from_dp_metadata(self, dp_metadata_list, **kwargs):
        self.dp_metadata_list = dict(dp_metadata_list)
        self.updates.append((dict(dp_metadata_list), kwargs))

    def recv_attn_output(self, metadata=None, ubatch_idx=None):
        for item in tuple(self.attn_outputs):
            if item[1].stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return AFDRecvOutput(hidden_states=item[0], metadata=item[1])
        raise IndexError(ubatch_idx)

    def create_recv_metadata(self, **kwargs):
        return AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=kwargs["layer_idx"],
            stage_idx=kwargs["ubatch_idx"],
            seq_lens=[1],
        )

    def send_ffn_output(self, ffn_output, metadata, **kwargs):
        self.ffn_outputs.append((ffn_output, metadata, kwargs))

    def update_metadata(self, metadata, recv_output):
        self.metadata_updates.append((metadata, recv_output))

    def close(self):
        return None


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        del kwargs
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


class _Slice:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class _UbatchSlice:
    def __init__(self, token_start, token_stop, request_start, request_stop):
        self.token_slice = _Slice(token_start, token_stop)
        self.request_slice = _Slice(request_start, request_stop)
        self.num_tokens = token_stop - token_start


class _RuntimeParallelConfig:
    data_parallel_size = 1
    data_parallel_rank = 0
    worker_cls = "unused"

    def __init__(self):
        self.enable_dbo = False
        self.ubatch_size = 0

    @property
    def use_ubatching(self):
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self):
        return 2 if self.enable_dbo else self.ubatch_size


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
                "connector": "camp2pconnector",
                "extra_config": extra_config or {},
            },
        },
        parallel_config=_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
        ),
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
    if not isinstance(tokens, list):
        tokens = tokens.tolist()
    assert tokens == [7]


def test_npu_attention_runner_sends_graph_flags():
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.vllm_config = _vllm_config(role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 3)

    runner._send_dp_metadata(SimpleNamespace(num_tokens_across_dp_cpu=[3]), None)

    assert runner.afd_connector.dp_metadata_updates[0][1:] == (True, True)
    assert runner.afd_connector.sent_dp_metadata_lists[0][1:] == (True, True)


def test_npu_attention_runner_sends_per_ubatch_dp_metadata():
    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner.vllm_config = _vllm_config(
        role="attention",
        use_ubatching=True,
        num_ubatches=2,
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None
    ubatch_slices = [
        _UbatchSlice(0, 3, 0, 1),
        _UbatchSlice(3, 8, 1, 2),
    ]

    runner._send_dp_metadata(None, ubatch_slices)

    sent_metadata = runner.afd_connector.sent_dp_metadata_lists[0][0]
    assert set(sent_metadata) == {0, 1}
    assert _metadata_tokens(sent_metadata[0]) == [3]
    assert _metadata_tokens(sent_metadata[1]) == [5]
    assert runner.afd_connector.dp_metadata_updates[0][0] == sent_metadata


def test_npu_ffn_runner_executes_eager_ffn_step():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
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
    assert runner.connector.metadata_updates == [
        (metadata, AFDRecvOutput(hidden_states="hidden", metadata=metadata)),
    ]


def test_npu_ffn_runner_processes_each_ubatch_stage():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(
        role="ffn",
        use_ubatching=True,
        num_ubatches=2,
    )
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 5
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata0 = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=3,
    )
    metadata1 = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=1,
        seq_len=5,
    )
    runner.connector.attn_outputs.append(("hidden0", metadata0))
    runner.connector.attn_outputs.append(("hidden1", metadata1))

    runner.execute_model(
        dp_metadata_list={
            0: _FakeDPMetadata([3]),
            1: _FakeDPMetadata([5]),
        },
    )

    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden0, layer=0)", metadata0, {"ubatch_idx": 0}),
        ("npu-ffn(hidden1, layer=0)", metadata1, {"ubatch_idx": 1}),
    ]
    assert [item[0].stage_idx for item in runner.connector.metadata_updates] == [
        0,
        1,
    ]


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


def test_npu_ffn_runner_replays_acl_graph_when_key_exists():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([1])}
    graph = _FakeGraph()
    runner._acl_graphs = {runner._make_graph_key(dp_metadata): {"graph": graph}}

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_npu_ffn_runner_warmup_uses_eager_forward_without_graph():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    runner._graph_capture_context = lambda: nullcontext()
    runner._set_cudagraph_capturing_enabled = lambda enabled: None
    runner._npu_free_memory = lambda: 0
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_warmup=True,
    )

    assert runner._acl_graphs == {}
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_capture_stores_acl_graph_and_skips_duplicate_state_update(
    monkeypatch,
):
    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.ffn_model_runner._full_aclgraph_runtime_mode",
        lambda: "FULL",
    )
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    runner.graph_pool = None
    runner._graph_capture_context = lambda: nullcontext()
    runner._npu_graph_context = lambda graph: nullcontext()
    runner._new_npu_graph = _FakeGraph
    runner._set_cudagraph_capturing_enabled = lambda enabled: None
    runner._npu_free_memory = lambda: 0
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list=dp_metadata,
        is_graph_capturing=True,
    )

    assert runner._make_graph_key(dp_metadata) in runner._acl_graphs
    assert runner.connector.updates == [
        (dp_metadata, {"is_graph_capturing": True}),
    ]


def test_npu_ffn_runner_requires_compute_hook():
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = SimpleNamespace()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    with pytest.raises(AttributeError, match="compute_ffn_output"):
        runner.execute_ffn_step(dp_metadata_list={0: _FakeDPMetadata([1])})


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


def test_npu_feature_validation_allows_two_way_ubatching_and_acl_graph_config():
    config = _vllm_config(use_ubatching=True, num_ubatches=2)
    fail_if_unsupported_npu_afd_features(config)
    assert npu_afd_num_ubatches(config) == 2

    with pytest.raises(RuntimeError, match="ubatching"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(use_ubatching=True, num_ubatches=4),
        )

    config = _vllm_config()
    config.model_config.enforce_eager = False
    fail_if_unsupported_npu_afd_features(config)


def test_npu_ubatching_request_forces_runtime_ubatching_flags():
    config = _vllm_config(extra_config={"enable_ubatching": True, "num_ubatches": 2})
    config.parallel_config = _RuntimeParallelConfig()

    enable_npu_afd_ubatching_if_requested(config)

    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config.use_ubatching is True
    assert config.parallel_config.num_ubatches == 2
    assert config.parallel_config.ubatch_size == 2


def test_npu_workers_initialize_workspace_with_configured_ubatches(monkeypatch):
    calls = []

    def record_workspace(device, *, num_ubatches):
        calls.append((device, num_ubatches))

    attention_runner = object()
    ffn_runner = SimpleNamespace(initialize_afd_connector=lambda: None)

    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.attention_worker.assert_compatible_afd_stack",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.attention_worker.init_ascend_workspace_for_afd",
        record_workspace,
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.attention_worker.AFDNPUAttentionModelRunner",
        lambda config, device: attention_runner,
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.ffn_worker.assert_compatible_afd_stack",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.ffn_worker.init_ascend_workspace_for_afd",
        record_workspace,
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.ascend.ffn_worker.AFDNPUFFNModelRunner",
        lambda config, device: ffn_runner,
    )

    device = SimpleNamespace(type="npu")
    attention_worker = object.__new__(AFDNPUAttentionWorker)
    attention_worker.vllm_config = _vllm_config(
        role="attention",
        use_ubatching=True,
        num_ubatches=2,
    )
    attention_worker.use_v2_model_runner = False
    attention_worker._init_device = lambda: device
    ffn_worker = object.__new__(AFDNPUFFNWorker)
    ffn_worker.vllm_config = _vllm_config(
        role="ffn",
        use_ubatching=True,
        num_ubatches=2,
    )
    ffn_worker.use_v2_model_runner = False
    ffn_worker._init_device = lambda: device

    attention_worker.init_device()
    ffn_worker.init_device()

    assert calls == [(device, 2), (device, 2)]
    assert attention_worker.model_runner is attention_runner
    assert ffn_worker.model_runner is ffn_runner


def _metadata_tokens(metadata):
    tokens = metadata.num_tokens_across_dp_cpu
    if isinstance(tokens, list):
        return tokens
    return tokens.tolist()
