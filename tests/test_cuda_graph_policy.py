from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.runtime.cuda_graph import (
    FULL_DECODE_ONLY,
    cudagraph_mode_name,
    make_ffn_graph_key,
    validate_cuda_graph_mode,
)


def _config(*, enforce_eager, cudagraph_mode=None, use_ubatching=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        parallel_config=SimpleNamespace(use_ubatching=use_ubatching),
    )


def test_cuda_graph_policy_allows_eager():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=True, cudagraph_mode="FULL"),
        role="attention",
    )

    assert policy.enabled is False
    assert policy.allow_attention_full_decode_only is False


def test_cuda_graph_policy_allows_full_decode_only_for_attention():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY),
        role="attention",
    )

    assert policy.enabled is True
    assert policy.mode_name == FULL_DECODE_ONLY
    assert policy.allow_attention_full_decode_only is True
    assert policy.enable_ffn_graph_cache is False


def test_cuda_graph_policy_allows_full_decode_only_for_ffn():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY),
        role="ffn",
    )

    assert policy.enabled is True
    assert policy.allow_attention_full_decode_only is False
    assert policy.enable_ffn_graph_cache is True


@pytest.mark.parametrize(
    "mode",
    [None, "NONE", "FULL", "PIECEWISE", "FULL_AND_PIECEWISE"],
)
def test_cuda_graph_policy_rejects_non_full_decode_only_graph_modes(mode):
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        validate_cuda_graph_mode(
            _config(enforce_eager=False, cudagraph_mode=mode),
            role="attention",
        )


def test_cuda_graph_policy_rejects_ubatching_with_graph():
    with pytest.raises(RuntimeError, match="ubatching"):
        validate_cuda_graph_mode(
            _config(
                enforce_eager=False,
                cudagraph_mode=FULL_DECODE_ONLY,
                use_ubatching=True,
            ),
            role="attention",
        )


def test_cudagraph_mode_name_handles_enum_like_values():
    mode = SimpleNamespace(name=FULL_DECODE_ONLY)

    assert cudagraph_mode_name(_config(enforce_eager=False, cudagraph_mode=mode)) == (
        FULL_DECODE_ONLY
    )


def test_make_ffn_graph_key_matches_original_shape():
    metadata_0 = SimpleNamespace(num_tokens_across_dp_cpu=[3, 5])
    metadata_1 = SimpleNamespace(num_tokens_across_dp_cpu=[7, 11])

    assert make_ffn_graph_key({1: metadata_1, 0: metadata_0}) == (
        (0, (3, 5)),
        (1, (7, 11)),
    )
