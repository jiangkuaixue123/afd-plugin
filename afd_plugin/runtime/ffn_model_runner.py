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
    _with_dp_derived_afd_rank,
    fail_if_unsupported_ubatching,
)
from afd_plugin.runtime.cuda_graph import (
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
    validate_cuda_graph_mode,
)
from afd_plugin.tracing import afd_trace, dp_metadata_summary, tensor_summary

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
        self.model_config = vllm_config.model_config
        self.load_config = vllm_config.load_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.afd_config = self.parse_config(vllm_config)
        if not self.afd_config.enabled:
            raise ValueError("AFD FFN runtime requires enabled=true")
        fail_if_unsupported_ubatching(vllm_config)
        self.afd_cudagraph_policy = validate_cuda_graph_mode(
            vllm_config,
            role="ffn",
        )
        self.afd_config = _with_dp_derived_afd_rank(vllm_config, self.afd_config)

        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.model: Any | None = None
        self.model_memory_usage = 0
        self.num_layers = int(self.model_config.hf_config.num_hidden_layers)
        self.use_cuda_graph = bool(
            self.afd_cudagraph_policy.enable_ffn_graph_cache,
        )
        self._cuda_graphs: dict[tuple, dict[str, Any]] = {}
        self._graph_memory_pool: Any | None = None
        afd_trace(
            "ffn_runner_init",
            role=self.afd_config.role,
            rank=rank,
            local_rank=local_rank,
            afd_server_rank=self.afd_config.afd_server_rank,
            num_layers=self.num_layers,
            use_cuda_graph=self.use_cuda_graph,
        )

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
        is_warmup: bool = False,
    ) -> None:
        del scheduler_output, intermediate_tensors
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner requires dp_metadata_list")
        graph_key = self._make_graph_key(dp_metadata_list)
        cuda_graph_info = self._cuda_graphs.get(graph_key)
        run_mode = graph_run_mode(
            is_warmup=is_warmup,
            is_graph_capturing=is_graph_capturing,
            graph_enabled=bool(self.use_cuda_graph),
            graph_exists=cuda_graph_info is not None,
        )
        afd_trace(
            "ffn_execute_model",
            graph_key=graph_key,
            run_mode=run_mode.value,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        if run_mode is AFDGraphRunMode.REPLAY:
            cuda_graph_info["graph"].replay()
            afd_trace("ffn_replay_cudagraph_done", graph_key=graph_key)
            return None

        if self.use_cuda_graph and run_mode is AFDGraphRunMode.EAGER:
            afd_trace("ffn_cudagraph_miss_fallback", graph_key=graph_key)

        self._ffn_forward(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
        )
        return None

    @staticmethod
    def _make_graph_key(dp_metadata_list: dict[int, Any]) -> tuple:
        return make_ffn_graph_key(dp_metadata_list)

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, Any],
        is_graph_capturing: bool = False,
        update_connector_state: bool = True,
    ) -> Any:
        if update_connector_state:
            self._update_connector_state(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
            )

        rank_ffn_output = None
        num_layers = max(int(self.num_layers or 0), 1)
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        afd_trace(
            "ffn_forward_begin",
            num_layers=num_layers,
            stages=stage_ids,
            dp_metadata=dp_metadata_summary(dp_metadata_list),
            is_graph_capturing=is_graph_capturing,
        )
        with _ffn_forward_context(self.vllm_config) as forward_context:
            for layer_idx in range(num_layers):
                for stage_idx in stage_ids:
                    afd_trace(
                        "ffn_layer_stage_recv_begin",
                        layer_idx=layer_idx,
                        stage_idx=stage_idx,
                    )
                    hidden_states, metadata = self._recv_attn_output(stage_idx)
                    afd_trace(
                        "ffn_layer_stage_recv_done",
                        layer_idx=layer_idx,
                        stage_idx=stage_idx,
                        metadata_layer_idx=metadata.layer_idx,
                        metadata_stage_idx=metadata.stage_idx,
                        metadata_ubatch_idx=metadata.ubatch_idx,
                        tensor=tensor_summary(hidden_states),
                    )
                    metadata.layer_idx = layer_idx
                    if forward_context is not None:
                        forward_context.dp_metadata = dp_metadata_list.get(
                            metadata.stage_idx,
                        )
                        forward_context.additional_kwargs["afd_metadata"] = metadata
                        _set_moe_layer_index(forward_context, layer_idx)
                    recv_handle_list = metadata.recv_handle_list
                    if recv_handle_list is not None:
                        for work in recv_handle_list:
                            work.wait()
                        metadata.recv_handle_list = None
                    afd_trace(
                        "ffn_layer_stage_compute_begin",
                        layer_idx=layer_idx,
                        stage_idx=stage_idx,
                        tensor=tensor_summary(hidden_states),
                    )
                    rank_ffn_output = self._execute_eager_mode(hidden_states, layer_idx)
                    afd_trace(
                        "ffn_layer_stage_compute_done",
                        layer_idx=layer_idx,
                        stage_idx=stage_idx,
                        tensor=tensor_summary(rank_ffn_output),
                    )
                    self.connector.send_ffn_output(rank_ffn_output, metadata)
                    afd_trace(
                        "ffn_layer_stage_send_done",
                        layer_idx=layer_idx,
                        stage_idx=stage_idx,
                        tensor=tensor_summary(rank_ffn_output),
                    )
        afd_trace("ffn_forward_done", stages=stage_ids)
        return rank_ffn_output

    def _execute_eager_mode(self, hidden_states: Any, layer_idx: int) -> Any:
        model = self.model
        compute = getattr(model, "compute_ffn_output", None)
        if callable(compute):
            return compute(hidden_states, layer_idx)
        return hidden_states

    def _recv_attn_output(self, stage_idx: int) -> tuple[Any, Any]:
        try:
            return self.connector.recv_attn_output(ubatch_idx=stage_idx)
        except TypeError:
            return self.connector.recv_attn_output()

    def _update_connector_state(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool,
    ) -> None:
        afd_trace(
            "ffn_update_connector_state_begin",
            dp_metadata=dp_metadata_summary(dp_metadata_list),
            is_graph_capturing=is_graph_capturing,
        )
        self.connector.update_state_from_dp_metadata(
            dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
        )
        afd_trace(
            "ffn_update_connector_state_done",
            dp_metadata=dp_metadata_summary(dp_metadata_list),
            is_graph_capturing=is_graph_capturing,
        )

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

    def _dummy_run(
        self,
        cudagraph_runtime_mode: Any,
        dp_metadata_list: dict[int, Any],
        is_attn_graph_capturing: bool,
    ) -> None:
        mode_name = getattr(cudagraph_runtime_mode, "name", str(cudagraph_runtime_mode))
        if mode_name.endswith(".FULL"):
            mode_name = "FULL"

        if mode_name == "FULL":
            import torch

            if self._graph_memory_pool is None:
                self._graph_memory_pool = torch.cuda.graph_pool_handle()
            graph_key = self._make_graph_key(dp_metadata_list)
            cudagraph = torch.cuda.CUDAGraph()
            # DP metadata receive/update is a control-plane side effect and must
            # complete before CUDA graph capture starts.
            self._update_connector_state(
                dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )
            with torch.cuda.graph(cudagraph, pool=self._graph_memory_pool):
                output = self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    update_connector_state=False,
                )
            self._cuda_graphs[graph_key] = {
                "graph": cudagraph,
                "input_hidden_states": output,
                "output": output,
            }
            afd_trace("ffn_capture_cudagraph_done", graph_key=graph_key)
        else:
            self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )

    def capture_model(
        self,
        dp_metadata_list: dict[int, Any] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if not self.use_cuda_graph:
            afd_trace("ffn_capture_cudagraph_skip_disabled")
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner.capture_model requires metadata")

        import time

        import torch
        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        from vllm.config import CUDAGraphMode
        from vllm.distributed.parallel_state import graph_capture

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()

        set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(device=self.device):
                if is_warmup:
                    afd_trace(
                        "ffn_capture_cudagraph_warmup",
                        graph_key=self._make_graph_key(dp_metadata_list),
                    )
                    self._update_connector_state(
                        dp_metadata_list,
                        is_graph_capturing=False,
                    )
                    self._ffn_forward(
                        dp_metadata_list=dp_metadata_list,
                        is_graph_capturing=False,
                        update_connector_state=False,
                    )
                else:
                    self._capture_graphs(
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        afd_trace(
            "ffn_capture_cudagraph_finished",
            graph_key=self._make_graph_key(dp_metadata_list),
            seconds=time.perf_counter() - start_time,
            bytes=cuda_graph_size,
        )
        return int(cuda_graph_size)

    def _capture_graphs(
        self,
        *,
        cudagraph_runtime_mode: Any,
        dp_metadata_list: dict[int, Any],
        is_attn_graph_capturing: bool = True,
    ) -> None:
        self._dummy_run(
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            dp_metadata_list=dp_metadata_list,
            is_attn_graph_capturing=is_attn_graph_capturing,
        )

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
        self.connector.close()


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


def _set_moe_layer_index(forward_context: object, layer_idx: int) -> None:
    all_moe_layers = forward_context.all_moe_layers
    if not all_moe_layers:
        return

    target = f".layers.{int(layer_idx)}."
    for idx, layer_name in enumerate(all_moe_layers):
        if target in f".{layer_name}.":
            forward_context.moe_layer_index = idx
            return


__all__ = ["GPUFFNModelRunner"]
