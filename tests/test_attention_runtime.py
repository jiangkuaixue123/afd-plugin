from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.models.forward_context import get_afd_metadata_from_forward_context
from afd_plugin.runtime.attention_model_runner import (
    AFDAttentionModelRunner,
    _has_enough_tokens_for_ubatches,
    _with_dp_derived_afd_rank,
    fail_if_cuda_graph_enabled,
    fail_if_unsupported_ubatching,
)
from afd_plugin.runtime.ubatch_wrapper import (
    build_ubatch_additional_kwargs,
    build_ubatch_afd_metadata,
)


class _UbatchSlice:
    def __init__(self, token_start, token_stop, request_start, request_stop):
        self.token_slice = slice(token_start, token_stop)
        self.request_slice = slice(request_start, request_stop)

    @property
    def num_tokens(self):
        return self.token_slice.stop - self.token_slice.start


class _RecordingConnector:
    world_rank = 1

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
        del is_graph_capturing, is_warmup
        self.dp_metadata_updates.append(dp_metadata_list)

    def send_dp_metadata_list(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        del is_graph_capturing, is_warmup
        self.sent_dp_metadata_lists.append(dp_metadata_list)


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "use_ubatching": False,
        "num_ubatches": 1,
        "dbo_decode_token_threshold": 32,
        "dbo_prefill_token_threshold": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_attention_runner_builds_single_stage_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_connector = object()
    runner._afd_transaction_counter = 0

    metadata = runner._build_afd_metadata(None, 7)

    assert metadata.afd_tokens_start_loc == [0]
    assert metadata.afd_reqs_start_loc == [0]
    assert metadata.afd_tokens_lens == [7]
    assert metadata.afd_tokens_unpadded_lens == [7]
    assert metadata.num_of_stages == 1
    assert metadata.afd_connector is runner.afd_connector


def test_attention_runner_installs_afd_metadata_on_forward_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs={"platform_key": "platform_value"},
        dp_metadata="dp",
        ubatch_slices=None,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["platform_key"] == "platform_value"
    assert forward_context.additional_kwargs["afd_metadata"].afd_tokens_lens == [5]
    assert runner.afd_connector.dp_metadata_updates == [{0: "dp"}]
    assert runner.afd_connector.sent_dp_metadata_lists == [{0: "dp"}]


def test_attention_runner_sends_per_ubatch_dp_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None
    ubatch_slices = [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)]

    runner._send_dp_metadata(None, ubatch_slices)

    assert set(runner.afd_connector.dp_metadata_updates[0]) == {0, 1}
    assert set(runner.afd_connector.sent_dp_metadata_lists[0]) == {0, 1}


def test_ubatch_metadata_clones_parent_and_preserves_additional_kwargs():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_connector = object()
    runner._afd_transaction_counter = 0
    parent = runner._build_afd_metadata(
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        8,
    )
    parent.afd_tokens_unpadded_lens = [3, 4]

    first = build_ubatch_afd_metadata(
        parent, [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)], 0
    )
    second = build_ubatch_afd_metadata(
        parent, [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)], 1
    )
    child_kwargs = build_ubatch_additional_kwargs(
        {"platform_key": "platform_value", "afd_metadata": parent},
        second,
    )

    assert first is not parent
    assert second is not parent
    assert first is not second
    assert first.ubatch_idx == 0
    assert first.afd_stage_idx == 0
    assert first.afd_tokens_lens == [3]
    assert second.ubatch_idx == 1
    assert second.afd_stage_idx == 1
    assert second.afd_tokens_start_loc == [3]
    assert second.afd_reqs_start_loc == [1]
    assert second.afd_tokens_lens == [5]
    assert second.afd_tokens_unpadded_lens == [4]
    assert child_kwargs["platform_key"] == "platform_value"
    assert child_kwargs["afd_metadata"] is second


def test_phase5_allows_two_way_ubatching_but_rejects_other_counts():
    fail_if_unsupported_ubatching(
        SimpleNamespace(
            parallel_config=_parallel_config(use_ubatching=True, num_ubatches=2),
        ),
    )

    with pytest.raises(RuntimeError, match="exactly two"):
        fail_if_unsupported_ubatching(
            SimpleNamespace(
                parallel_config=_parallel_config(use_ubatching=True, num_ubatches=4),
            ),
        )


def test_attention_runner_enables_ubatching_for_afd_dp1_thresholds():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            use_ubatching=True,
            num_ubatches=2,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=4,
        ),
    )
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **_kwargs: False

    assert runner._should_ubatch_without_vllm_dp(
        num_tokens=4,
        num_reqs=1,
        num_scheduled_tokens_np=[4],
        max_num_scheduled_tokens=4,
        use_cascade_attn=False,
    )

    assert not runner._should_ubatch_without_vllm_dp(
        num_tokens=3,
        num_reqs=1,
        num_scheduled_tokens_np=[3],
        max_num_scheduled_tokens=3,
        use_cascade_attn=False,
    )


def test_attention_runner_enables_decode_ubatching_for_afd_dp1_thresholds():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            use_ubatching=True,
            num_ubatches=2,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=4,
        ),
    )
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **_kwargs: True

    assert runner._should_ubatch_without_vllm_dp(
        num_tokens=2,
        num_reqs=2,
        num_scheduled_tokens_np=[1, 1],
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )

    assert not runner._should_ubatch_without_vllm_dp(
        num_tokens=1,
        num_reqs=1,
        num_scheduled_tokens_np=[1],
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )


def test_attention_runner_rejects_empty_native_ubatches():
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(num_ubatches=2),
    )

    assert not _has_enough_tokens_for_ubatches(vllm_config, 1)
    assert _has_enough_tokens_for_ubatches(vllm_config, 2)


def test_attention_runner_inherits_native_dummy_run_microbatching():
    assert "_dummy_run" in AFDAttentionModelRunner.__dict__


def test_forward_context_provider_installs_missing_afd_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=1),
    )

    from afd_plugin.models.forward_context import use_afd_metadata_provider

    with use_afd_metadata_provider(runner):
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata is not None
    assert metadata.afd_tokens_lens == [1]
    assert forward_context.additional_kwargs["afd_metadata"] is metadata
    assert runner.afd_connector.sent_dp_metadata_lists


def test_attention_runtime_rejects_cuda_graph_until_phase6():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
    )

    with pytest.raises(RuntimeError, match="CUDA graph"):
        fail_if_cuda_graph_enabled(vllm_config)


def test_afd_rank_derives_from_data_parallel_rank():
    config = AFDConfig(
        enabled=True,
        role="attention",
        connector="p2pconnector",
        num_attention_servers=2,
        num_ffn_servers=2,
        afd_server_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(data_parallel_size=2, data_parallel_rank=1),
    )

    ranked = _with_dp_derived_afd_rank(vllm_config, config)

    assert ranked.afd_server_rank == 1
    assert config.afd_server_rank == 0
