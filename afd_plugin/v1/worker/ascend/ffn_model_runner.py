# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side model runner for the first AFD runtime version."""

from __future__ import annotations

from typing import Any

from afd_plugin.compat.ascend import (
    ascend_forward_context,
    ensure_vllm_config_has_afd_proxy,
    fail_if_unsupported_npu_afd_features,
    mirror_afd_metadata_on_forward_context,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import AFDConnectorFactory, AFDMetadata
from afd_plugin.v1.worker._optional import optional_class
from afd_plugin.v1.worker.attention_model_runner import (
    _resolve_world_ranks,
    _with_dp_derived_afd_rank,
)
from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key
from afd_plugin.v1.worker.ffn_model_runner import _set_moe_layer_index

_NPUModelRunner, _NPUModelRunner_IMPORT_ERROR = optional_class(
    "vllm_ascend.worker.model_runner_v1",
    "NPUModelRunner",
)


class AFDNPUFFNModelRunner(_NPUModelRunner):  # type: ignore[misc, valid-type]
    """Connector-driven NPU FFN runner.

    This first version supports eager single-stream execution and keeps ACL graph
    and ubatching out of the data path.
    """

    afd_expected_role = "ffn"
    vllm_base_import_error = _NPUModelRunner_IMPORT_ERROR

    def __init__(self, vllm_config: object, device: object) -> None:
        if _NPUModelRunner_IMPORT_ERROR is not None:
            raise RuntimeError(
                "AFDNPUFFNModelRunner requires an importable vLLM-Ascend runtime",
            ) from _NPUModelRunner_IMPORT_ERROR

        afd_config = self.parse_config(vllm_config)
        ensure_vllm_config_has_afd_proxy(vllm_config, afd_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        if not self.afd_config.enabled:
            raise ValueError("AFD NPU FFN runtime requires enabled=true")
        fail_if_unsupported_npu_afd_features(vllm_config)
        self.afd_config = _with_dp_derived_afd_rank(vllm_config, self.afd_config)
        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.num_layers = _resolve_num_hidden_layers(self.model_config)
        self.use_aclgraph = False
        self._acl_graphs: dict[tuple, dict[str, Any]] = {}

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        del kv_cache_config
        return None

    def profile_run(self) -> None:
        return None

    def execute_ffn_step(
        self,
        *,
        dp_metadata_list: dict[int, Any],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        del is_warmup
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN requires dp_metadata_list")
        if is_graph_capturing:
            raise RuntimeError("AFD NPU FFN ACL graph capture is not supported yet")
        self._ffn_forward(dp_metadata_list=dp_metadata_list)
        return None

    def execute_model(
        self,
        scheduler_output: Any = None,
        intermediate_tensors: Any = None,
        *,
        dp_metadata_list: dict[int, Any] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        del scheduler_output, intermediate_tensors
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN is connector-driven")
        return self.execute_ffn_step(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )

    @staticmethod
    def _make_graph_key(dp_metadata_list: dict[int, Any]) -> tuple:
        return make_ffn_graph_key(dp_metadata_list)

    def _ffn_forward(self, *, dp_metadata_list: dict[int, Any]) -> Any:
        self.connector.update_state_from_dp_metadata(
            dp_metadata_list,
            is_graph_capturing=False,
        )
        num_stages = max(len(dp_metadata_list), 1)
        afd_metadata = AFDMetadata(
            afd_tokens_start_loc=[],
            afd_reqs_start_loc=[],
            afd_stage_idx=0,
            afd_connector=self.connector,
            afd_tokens_lens=[],
            num_of_stages=num_stages,
        )
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        num_tokens_across_dp = _first_dp_token_counts(dp_metadata_list)
        num_tokens = _first_token_count(num_tokens_across_dp)
        rank_ffn_output = None

        with ascend_forward_context(
            vllm_config=self.vllm_config,
            afd_metadata=afd_metadata,
            model_instance=getattr(self, "model", None),
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        ) as forward_context:
            for layer_idx in range(max(int(self.num_layers or 0), 1)):
                for stage_idx in stage_ids:
                    recv_output = self._recv_attn_output(stage_idx, layer_idx)
                    hidden_states, metadata, payload = _normalize_recv_output(
                        recv_output,
                        stage_idx=stage_idx,
                        layer_idx=layer_idx,
                    )
                    metadata.layer_idx = layer_idx
                    metadata.stage_idx = stage_idx
                    if forward_context is not None:
                        forward_context.dp_metadata = dp_metadata_list.get(stage_idx)
                        mirror_afd_metadata_on_forward_context(
                            forward_context,
                            metadata,
                        )
                        _set_moe_layer_index(forward_context, layer_idx)

                    recv_handle_list = getattr(metadata, "recv_handle_list", None)
                    if recv_handle_list is not None:
                        for work in recv_handle_list:
                            work.wait()
                        metadata.recv_handle_list = None

                    rank_ffn_output = self._run_ffn_computation(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                        group_list=getattr(payload, "group_list", None),
                        dynamic_scales=getattr(payload, "dynamic_scales", None),
                        topk_weights=getattr(payload, "topk_weights", None),
                        topk_ids=getattr(payload, "topk_ids", None),
                        router_logits=getattr(payload, "router_logits", None),
                        row_idx=getattr(payload, "row_idx", None),
                        x_active_mask=getattr(payload, "x_active_mask", None),
                        cam_p2p_ep_name=getattr(payload, "cam_p2p_ep_name", "") or "",
                    )
                    self.connector.send_ffn_output(
                        rank_ffn_output,
                        metadata,
                        ubatch_idx=stage_idx,
                    )
        return rank_ffn_output

    def _recv_attn_output(self, stage_idx: int, layer_idx: int) -> Any:
        metadata = None
        create_recv_metadata = getattr(self.connector, "create_recv_metadata", None)
        if callable(create_recv_metadata):
            metadata = create_recv_metadata(
                dp_metadata_list=self.connector.dp_metadata_list,
                ubatch_idx=stage_idx,
                layer_idx=layer_idx,
                max_num_tokens=getattr(self, "max_num_tokens", 0),
            )
        try:
            return self.connector.recv_attn_output(
                metadata=metadata,
                ubatch_idx=stage_idx,
            )
        except TypeError:
            return self.connector.recv_attn_output(ubatch_idx=stage_idx)

    def _run_ffn_computation(
        self,
        *,
        hidden_states: Any,
        layer_idx: int,
        group_list: Any = None,
        dynamic_scales: Any = None,
        topk_weights: Any = None,
        topk_ids: Any = None,
        router_logits: Any = None,
        row_idx: Any = None,
        x_active_mask: Any = None,
        cam_p2p_ep_name: str = "",
    ) -> Any:
        model = getattr(self, "model", None)
        compute = getattr(model, "compute_ffn_output", None)
        if not callable(compute):
            return hidden_states
        try:
            return compute(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                row_idx=row_idx,
                x_active_mask=x_active_mask,
                cam_p2p_ep_name=cam_p2p_ep_name,
            )
        except TypeError:
            return compute(hidden_states, layer_idx)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        del grammar_output
        raise RuntimeError("AFD NPU FFN runners do not sample tokens")

    def shutdown(self) -> None:
        self.connector.close()
        super().shutdown()


def _normalize_recv_output(
    recv_output: Any,
    *,
    stage_idx: int,
    layer_idx: int,
) -> tuple[Any, Any, Any]:
    if isinstance(recv_output, tuple):
        hidden_states, metadata = recv_output
        return hidden_states, metadata, recv_output
    hidden_states = recv_output.hidden_states
    metadata = getattr(recv_output, "metadata", None)
    if metadata is None:
        from afd_plugin.connectors import AFDConnectorMetadata

        metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[_tensor_tokens(hidden_states)],
        )
    return hidden_states, metadata, recv_output


def _resolve_num_hidden_layers(model_config: object) -> int:
    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    return int(hf_config.num_hidden_layers)


def _first_dp_token_counts(dp_metadata_list: dict[int, Any]) -> Any:
    if not dp_metadata_list:
        return None
    first_key = sorted(int(key) for key in dp_metadata_list)[0]
    return getattr(dp_metadata_list[first_key], "num_tokens_across_dp_cpu", None)


def _first_token_count(num_tokens_across_dp: Any) -> int:
    if num_tokens_across_dp is None:
        return 1
    first = num_tokens_across_dp[0]
    item = getattr(first, "item", None)
    return max(1, int(item() if callable(item) else first))


def _tensor_tokens(hidden_states: Any) -> int:
    shape = getattr(hidden_states, "shape", None)
    if shape is None:
        return 1
    return max(1, int(shape[0]))


__all__ = ["AFDNPUFFNModelRunner"]
