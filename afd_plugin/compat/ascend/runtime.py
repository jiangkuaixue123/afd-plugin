# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small Ascend runtime shims kept out of worker/model-runner modules."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from afd_plugin.config import AFDConfig, parse_afd_config

_PATCHES_APPLIED = False


def ensure_ascend_runtime_available() -> None:
    try:
        import vllm_ascend  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "AFD NPU runtime requires an importable vllm-ascend runtime",
        ) from exc


def apply_afd_ascend_patches_if_needed() -> None:
    """Apply plugin-owned Ascend patches.

    The first NPU runtime version does not need a monkey patch.  The function is
    intentionally present and idempotent so future patches have one guarded
    entry point.
    """

    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True


def init_ascend_workspace_for_afd(device: object, *, num_ubatches: int = 1) -> None:
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(device, int(num_ubatches))


def npu_afd_num_ubatches(vllm_config: object) -> int:
    if not npu_afd_ubatching_requested(vllm_config):
        return 1
    afd_config = parse_afd_config(vllm_config, validate=False)
    requested = afd_config.extra_config.get("num_ubatches")
    if requested is not None:
        return int(requested)
    parallel_config = vllm_config.parallel_config
    num_ubatches = int(getattr(parallel_config, "num_ubatches", 0))
    return 2 if num_ubatches <= 1 else num_ubatches


def npu_afd_ubatching_requested(vllm_config: object) -> bool:
    parallel_config = vllm_config.parallel_config
    if bool(getattr(parallel_config, "use_ubatching", False)):
        return True
    if _truthy(os.getenv("AFD_NPU_ENABLE_UBATCHING")):
        return True
    afd_config = parse_afd_config(vllm_config, validate=False)
    extra = afd_config.extra_config or {}
    return any(
        _truthy(extra.get(key))
        for key in (
            "enable_ubatching",
            "enable_dbo",
            "force_enable_ubatching",
        )
    )


def enable_npu_afd_ubatching_if_requested(vllm_config: object) -> None:
    if not npu_afd_ubatching_requested(vllm_config):
        return
    parallel_config = vllm_config.parallel_config
    parallel_config.enable_dbo = True


def fail_if_unsupported_npu_afd_features(vllm_config: object) -> None:
    """Fail fast for NPU AFD features intentionally outside the first version."""

    afd_config = parse_afd_config(vllm_config)
    extra = afd_config.extra_config or {}

    if _truthy(extra.get("compute_gate_on_attention")):
        raise RuntimeError(
            "AFD NPU runtime does not support compute_gate_on_attention=true yet",
        )

    quant_mode = extra.get("quant_mode", 0)
    if quant_mode not in (None, "", 0, "0"):
        raise RuntimeError("AFD NPU runtime currently supports only quant_mode=0")

    if _truthy(extra.get("is_multistream")):
        raise RuntimeError("AFD NPU runtime does not support multistream yet")
    if _truthy(extra.get("is_attn_multistream")):
        raise RuntimeError(
            "AFD NPU runtime does not support attention multistream yet",
        )
    if _truthy(extra.get("is_ffn_multistream")):
        raise RuntimeError("AFD NPU runtime does not support FFN multistream yet")

    multistream_info = extra.get("multistream_info")
    if isinstance(multistream_info, Mapping):
        for key in ("enable", "enabled", "attn_enable", "ffn_enable"):
            if _truthy(multistream_info.get(key)):
                raise RuntimeError(
                    "AFD NPU runtime does not support multistream_info enabled",
                )

    if npu_afd_ubatching_requested(vllm_config):
        num_ubatches = npu_afd_num_ubatches(vllm_config)
        if num_ubatches != 2:
            raise RuntimeError(
                "AFD NPU runtime currently supports ubatching/DBO only with "
                "exactly two ubatches; "
                f"got num_ubatches={num_ubatches}",
            )

    if not bool(vllm_config.model_config.enforce_eager):
        _npu_aclgraph_mode_name(vllm_config)


def _npu_aclgraph_mode_name(vllm_config: object) -> str:
    return vllm_config.compilation_config.cudagraph_mode.name


def mirror_afd_metadata_on_forward_context(
    forward_context: object,
    afd_metadata: object,
) -> None:
    """Store AFD metadata in canonical kwargs and Ascend's mirrored attribute."""

    if forward_context.additional_kwargs is None:
        forward_context.additional_kwargs = {}
    forward_context.additional_kwargs["afd_metadata"] = afd_metadata
    forward_context.afd_metadata = afd_metadata


@contextmanager
def ascend_forward_context(
    *,
    vllm_config: object,
    afd_metadata: object | None = None,
    model_instance: object | None = None,
    num_tokens: int = 0,
    num_tokens_across_dp: object | None = None,
    aclgraph_runtime_mode: object | None = None,
) -> Iterator[object | None]:
    """Create the minimal forward context needed by connector-driven FFN steps."""

    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    except Exception:
        yield None
        return

    if aclgraph_runtime_mode is None:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    context_kwargs = {
        "attn_metadata": None,
        "vllm_config": vllm_config,
        "batch_descriptor": None,
        "aclgraph_runtime_mode": aclgraph_runtime_mode,
        "model_instance": model_instance,
        "afd_metadata": afd_metadata,
        "num_tokens": int(num_tokens),
        "num_tokens_across_dp": num_tokens_across_dp,
    }
    signature = inspect.signature(set_ascend_forward_context)
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        context_kwargs = {
            key: value
            for key, value in context_kwargs.items()
            if key in signature.parameters
        }

    with set_ascend_forward_context(**context_kwargs):
        forward_context = get_forward_context()
        if afd_metadata is not None:
            mirror_afd_metadata_on_forward_context(forward_context, afd_metadata)
        yield forward_context


def ensure_vllm_config_has_afd_proxy(
    vllm_config: object,
    afd_config: AFDConfig | None = None,
) -> object | None:
    """Install an instance-local AFD proxy for vLLM-Ascend builds that read it.

    The plugin's public config channel remains ``additional_config["afd"]``.
    This shim only gives vLLM-Ascend code that still does ``vllm_config.afd_config``
    a read-only compatibility view.
    """

    config = afd_config or parse_afd_config(vllm_config, validate=False)
    if not config.enabled:
        return None
    proxy = _AscendAFDConfigProxy(config)
    vllm_config.afd_config = proxy
    return proxy


@dataclass(frozen=True)
class _AscendAFDConfigProxy:
    _config: AFDConfig

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def afd_extra_config(self) -> dict[str, Any]:
        return self._config.extra_config

    @property
    def afd_connector(self) -> str:
        return self._config.connector

    @property
    def afd_role(self) -> str:
        return self._config.role

    @property
    def afd_port(self) -> int:
        return self._config.port

    @property
    def afd_host(self) -> str:
        return self._config.host

    @property
    def is_attention_server(self) -> bool:
        return self._config.role == "attention"

    @property
    def is_ffn_server(self) -> bool:
        return self._config.role == "ffn"

    @property
    def compute_gate_on_attention(self) -> bool:
        return bool(self._config.extra_config.get("compute_gate_on_attention", False))

    @property
    def quant_mode(self) -> int:
        return int(self._config.extra_config.get("quant_mode", 0) or 0)

    @property
    def is_multistream(self) -> bool:
        return bool(self._config.extra_config.get("is_multistream", False))

    @property
    def is_attn_multistream(self) -> bool:
        return bool(self._config.extra_config.get("is_attn_multistream", False))

    @property
    def is_ffn_multistream(self) -> bool:
        return bool(self._config.extra_config.get("is_ffn_multistream", False))

    @property
    def multistream_info(self) -> dict[str, Any]:
        value = self._config.extra_config.get("multistream_info", {})
        return value if isinstance(value, dict) else {}

    def compute_hash(self) -> str:
        return self._config.compute_hash()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = [
    "apply_afd_ascend_patches_if_needed",
    "ascend_forward_context",
    "ensure_ascend_runtime_available",
    "ensure_vllm_config_has_afd_proxy",
    "fail_if_unsupported_npu_afd_features",
    "init_ascend_workspace_for_afd",
    "mirror_afd_metadata_on_forward_context",
]
