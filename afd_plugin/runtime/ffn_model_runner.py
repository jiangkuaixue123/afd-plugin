# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side model runner for the Phase 3 MVP."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import AFDConnectorFactory
from afd_plugin.runtime._optional import optional_class
from afd_plugin.runtime.attention_model_runner import (
    fail_if_cuda_graph_enabled,
    fail_if_ubatching_enabled,
)

_LoRAModelRunnerMixin, _LoRAModelRunnerMixin_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.lora_model_runner_mixin",
    "LoRAModelRunnerMixin",
)


class GPUFFNModelRunner(_LoRAModelRunnerMixin):  # type: ignore[misc, valid-type]
    """Minimal FFN model runner for connector-driven Phase 3 execution."""

    afd_expected_role = "ffn"
    vllm_base_import_error = _LoRAModelRunnerMixin_IMPORT_ERROR

    def __init__(self, vllm_config: object, device: object) -> None:
        if _LoRAModelRunnerMixin_IMPORT_ERROR is not None:
            raise RuntimeError(
                "GPUFFNModelRunner requires an importable vLLM runtime",
            ) from _LoRAModelRunnerMixin_IMPORT_ERROR

        self.vllm_config = vllm_config
        self.model_config = getattr(vllm_config, "model_config", None)
        self.load_config = getattr(vllm_config, "load_config", None)
        self.device = device
        self.dtype = getattr(self.model_config, "dtype", None)
        self.afd_config = self.parse_config(vllm_config)
        if not self.afd_config.enabled:
            raise ValueError("AFD FFN runtime requires enabled=true")
        fail_if_ubatching_enabled(vllm_config)
        fail_if_cuda_graph_enabled(vllm_config)

        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.model: Any | None = None
        self.model_memory_usage = 0
        self.num_layers = _resolve_num_hidden_layers(self.model_config)

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def get_model(self) -> Any:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load the vLLM model lazily, without importing vLLM at module import."""

        del load_dummy_weights
        del kwargs
        if self.model_config is None or self.load_config is None:
            raise RuntimeError("GPUFFNModelRunner requires vLLM model/load config")

        from vllm.model_executor.model_loader import get_model_loader
        from vllm.utils.mem_utils import DeviceMemoryProfiler

        model_loader = get_model_loader(self.load_config)
        with DeviceMemoryProfiler() as profiler:
            if self.model is None:
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.model_config,
                )
            else:
                model_loader.load_weights(
                    self.model,
                    model_config=self.model_config,
                )
        self.model_memory_usage = profiler.consumed_memory

    def profile_run(self) -> None:
        return None

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        del kv_cache_config
        return None

    def execute_model(
        self,
        scheduler_output: Any = None,
        intermediate_tensors: Any = None,
        *,
        dp_metadata_list: dict[int, Any] | None = None,
        is_graph_capturing: bool = False,
    ) -> None:
        del scheduler_output, intermediate_tensors
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner requires dp_metadata_list")
        self._ffn_forward(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
        )
        return None

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, Any],
        is_graph_capturing: bool = False,
    ) -> Any:
        self._update_connector_state(
            dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
        )

        rank_ffn_output = None
        num_layers = max(int(self.num_layers or 0), 1)
        with _ffn_forward_context(
            getattr(self, "vllm_config", None),
        ) as forward_context:
            for layer_idx in range(num_layers):
                hidden_states, metadata = self.connector.recv_attn_output()
                metadata.layer_idx = layer_idx
                if forward_context is not None:
                    forward_context.dp_metadata = dp_metadata_list.get(
                        getattr(metadata, "stage_idx", 0),
                    )
                recv_handle_list = getattr(metadata, "recv_handle_list", None)
                if recv_handle_list is not None:
                    for work in recv_handle_list:
                        work.wait()
                    metadata.recv_handle_list = None
                rank_ffn_output = self._execute_eager_mode(hidden_states, layer_idx)
                self.connector.send_ffn_output(rank_ffn_output, metadata)
        return rank_ffn_output

    def _execute_eager_mode(self, hidden_states: Any, layer_idx: int) -> Any:
        model = self.model
        compute = getattr(model, "compute_ffn_output", None)
        if callable(compute):
            return compute(hidden_states, layer_idx)
        return hidden_states

    def _update_connector_state(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool,
    ) -> None:
        update = getattr(self.connector, "update_state_from_dp_metadata", None)
        if not callable(update):
            return
        try:
            update(dp_metadata_list, is_graph_capturing=is_graph_capturing)
        except TypeError:
            update(dp_metadata_list, is_graph_capturing)

    def update_config(self, overrides: dict[str, Any]) -> None:
        for config_name, config_overrides in overrides.items():
            config = getattr(self, config_name)
            try:
                from vllm.config import update_config

                updated_config = update_config(config, config_overrides)
            except Exception:
                updated_config = config_overrides
            setattr(self, config_name, updated_config)

    def reload_weights(self) -> None:
        if self.model is None:
            raise RuntimeError("Cannot reload weights before model is loaded")
        self.load_model()

    def capture_model(self, *args: Any, **kwargs: Any) -> int:
        del args, kwargs
        return 0

    def _dummy_run(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    def _dummy_sampler_run(self, hidden_states: Any) -> None:
        del hidden_states
        return None

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        del grammar_output
        raise RuntimeError("FFN runners do not sample tokens")

    def add_lora(self, lora_request: Any) -> bool:
        del lora_request
        return False

    def remove_lora(self, lora_id: int) -> bool:
        del lora_id
        return False

    def pin_lora(self, lora_id: int) -> bool:
        del lora_id
        return False

    def list_loras(self) -> set[int]:
        return set()

    @property
    def lora_config(self) -> None:
        return None

    @property
    def is_pooling_model(self) -> bool:
        return False

    def get_supported_tasks(self) -> tuple[Any, ...]:
        return ()

    def shutdown(self) -> None:
        close = getattr(getattr(self, "connector", None), "close", None)
        if callable(close):
            close()


def _resolve_world_ranks() -> tuple[int, int]:
    try:
        from vllm.distributed.parallel_state import get_world_group

        group = get_world_group()
        return int(group.rank), int(group.local_rank)
    except Exception:
        return 0, 0


@contextmanager
def _ffn_forward_context(vllm_config: object):
    try:
        from vllm.forward_context import get_forward_context, set_forward_context
    except Exception:
        yield None
        return

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        yield get_forward_context()


def _resolve_num_hidden_layers(model_config: object | None) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", None)
    value = getattr(text_config, "num_hidden_layers", None)
    if value is None:
        value = getattr(hf_config, "num_hidden_layers", None)
    if value is None:
        return 1
    return int(value)


__all__ = ["GPUFFNModelRunner"]
