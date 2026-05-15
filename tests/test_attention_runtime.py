from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDConnectorFactory
from afd_plugin.runtime.attention_model_runner import AFDAttentionModelRunner


def test_attention_runner_builds_single_stage_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_connector = object()

    metadata = runner._build_afd_metadata(None, 7)

    assert metadata.afd_tokens_start_loc == [0]
    assert metadata.afd_reqs_start_loc == [0]
    assert metadata.afd_tokens_lens == [7]
    assert metadata.num_of_stages == 1
    assert metadata.afd_connector is runner.afd_connector


def test_attention_runner_installs_afd_metadata_on_forward_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.afd_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(),
        runner.afd_config,
    )
    runner._is_warmup = False
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


def test_attention_runner_rejects_ubatching_in_phase2():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_connector = object()
    runner._is_warmup = False

    with pytest.raises(RuntimeError, match="ubatching"):
        runner._send_dp_metadata(None, [object(), object()])
