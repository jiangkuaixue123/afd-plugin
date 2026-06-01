# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned Ascend ubatch wrapper.

This is the plugin copy of the basic Ascend DBO wrapper from vLLM-Ascend
commit ``cdd212830271249a1cafcb850c210133f21771c5``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from afd_plugin.v1.worker._optional import optional_class

_UBatchWrapper, _UBatchWrapper_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.gpu_ubatch_wrapper",
    "UBatchWrapper",
)


@dataclass
class AscendUbatchMetadata:
    context: Any
    input_ids: Any | None
    positions: Any
    inputs_embeds: Any | None
    intermediate_tensors: Any | None
    num_tokens: int


@dataclass
class AscendNPUGraphMetaData:
    aclgraph: Any
    ubatch_metadata: list[AscendUbatchMetadata]
    outputs: Any | None = None


class AscendUBatchWrapper(_UBatchWrapper):  # type: ignore[misc, valid-type]
    """Ascend microbatch wrapper used only by AFD NPU runtimes."""

    vllm_base_import_error = _UBatchWrapper_IMPORT_ERROR

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: Any,
        runtime_mode: Any,
        device: Any,
    ) -> None:
        if _UBatchWrapper_IMPORT_ERROR is not None:
            raise RuntimeError(
                "AscendUBatchWrapper requires an importable vLLM runtime",
            ) from _UBatchWrapper_IMPORT_ERROR

        import torch
        from vllm.config import CUDAGraphMode
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(3)
        self.cudagraphs: dict[int, AscendNPUGraphMetaData] = {}
        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = ACLGraphWrapper(
                runnable,
                vllm_config,
                runtime_mode=runtime_mode,
            )
        self.device = device

    @property
    def graph_pool(self) -> Any:
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        return None

    def clear_graphs(self) -> None:
        self.cudagraphs.clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.concrete_aclgraph_entries.clear()

    def __getattr__(self, key: str) -> Any:
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not found in AscendUBatchWrapper runnable.",
        )

    def unwrap(self) -> Callable[..., Any]:
        return self.runnable

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import DPMetadata, get_forward_context

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if ubatch_slices is None:
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE
            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            assert self.cudagraph_wrapper is not None
            return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.npu.current_stream()

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            if dp_size > 1:
                ubatch_num_tokens_across_dp = torch.tensor(
                    [ubatch_slice.num_tokens] * dp_size,
                    device="cpu",
                    dtype=torch.int32,
                )
                ubatch_dp_metadata.append(
                    DPMetadata.make(
                        self.vllm_config.parallel_config,
                        ubatch_slice.num_tokens,
                        ubatch_num_tokens_across_dp,
                    ),
                )
            else:
                ubatch_dp_metadata.append(None)

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                torch.npu.Stream(device=torch.npu.current_device()),
                ubatch_dp_metadata,
                batch_descriptor,
                CUDAGraphMode.NONE,
            )
            return self._capture_ubatches(ubatch_metadata, self.runnable)
        if (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.aclgraph.replay()
            get_forward_context().dbo_enabled = True
            return cudagraph_metadata.outputs

        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            compute_stream,
            ubatch_dp_metadata,
            batch_descriptor,
            CUDAGraphMode.NONE,
        )
        return self._run_ubatches(ubatch_metadata, self.runnable)

    def _make_ubatch_metadata(
        self,
        ubatch_slices: Any,
        attn_metadata: Any,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
        compute_stream: Any,
        dp_metadata: list[Any],
        batch_descriptor: Any,
        cudagraph_runtime_mode: Any,
    ) -> list[AscendUbatchMetadata]:
        from vllm.forward_context import get_forward_context

        from afd_plugin.v1.worker.ascend.forward_context import (
            create_ascend_forward_context,
        )
        from afd_plugin.v1.worker.ascend.ubatching import make_ubatch_contexts

        cur_forward_context = get_forward_context()
        forward_contexts = []
        for i, _ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=attn_metadata[i]
                    if attn_metadata is not None
                    else None,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    ubatch_slices=ubatch_slices,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    ubatch_num=i,
                    skip_compiled=cur_forward_context.skip_compiled,
                ),
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        metadata_list: list[AscendUbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            metadata_list.append(
                AscendUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                ),
            )
        return metadata_list

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
    ) -> tuple[Any, Any, Any, Any]:
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        sliced_positions = (
            positions[:, tokens_slice]
            if positions.ndim == 2
            else positions[tokens_slice]
        )
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )

        if intermediate_tensors is not None and _enable_sp(self.vllm_config):
            from vllm.distributed import get_tensor_model_parallel_world_size

            tp_size = get_tensor_model_parallel_world_size()
            start = (tokens_slice.start + tp_size - 1) // tp_size
            if start != 0:
                stop = (
                    start
                    + (tokens_slice.stop - tokens_slice.start + tp_size - 1) // tp_size
                )
            else:
                stop = (tokens_slice.stop + tp_size - 1) // tp_size
            tokens_slice = slice(start, stop)
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )
        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def _merge_intermediate_tensors(self, intermediate_tensor_list: list[Any]) -> Any:
        from vllm.sequence import IntermediateTensors

        assert len(intermediate_tensor_list) == 2
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = _torch_cat(
                [
                    intermediate_tensor_list[0].tensors[key],
                    intermediate_tensor_list[1].tensors[key],
                ],
                dim=0,
            )
        return IntermediateTensors(result)

    def _merge_outputs(
        self,
        sorted_results: list[Any],
        ubatch_metadata: list[AscendUbatchMetadata],
    ) -> Any:
        from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather

        if not get_pp_group().is_last_rank:
            return self._merge_intermediate_tensors(sorted_results)

        ubatch_forward_context = ubatch_metadata[0].context.forward_context
        if ubatch_forward_context.flash_comm_v1_enabled:
            for i, result in enumerate(sorted_results):
                sorted_results[i] = tensor_model_parallel_all_gather(result, 0)
                pad_size = ubatch_metadata[i].context.forward_context.pad_size
                if pad_size > 0:
                    sorted_results[i] = sorted_results[i][:-pad_size, :]
        return _torch_cat(sorted_results, dim=0)

    def _run_ubatch_thread(
        self, results: list[Any], model: Any, ubatch_metadata: Any
    ) -> None:
        with ubatch_metadata.context:
            model_output = model(
                input_ids=ubatch_metadata.input_ids,
                positions=ubatch_metadata.positions,
                intermediate_tensors=ubatch_metadata.intermediate_tensors,
                inputs_embeds=ubatch_metadata.inputs_embeds,
            )
        results.append((ubatch_metadata.context.id, model_output))

    def _run_ubatches(
        self,
        ubatch_metadata: list[AscendUbatchMetadata],
        model: Any,
    ) -> Any:
        from vllm.forward_context import get_forward_context, override_forward_context

        results: list[tuple[int, Any]] = []
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=self._run_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        sorted_results = [value for _, value in sorted(results)]
        get_forward_context().dbo_enabled = True
        return self._merge_outputs(sorted_results, ubatch_metadata)

    def _capture_ubatches(
        self,
        ubatch_metadata: list[AscendUbatchMetadata],
        model: Any,
    ) -> Any:
        import torch
        from vllm.forward_context import get_forward_context, override_forward_context

        results: list[tuple[int, Any]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = sum(metadata.num_tokens for metadata in ubatch_metadata)

        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=self._run_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()

            cudagraph_metadata = AscendNPUGraphMetaData(
                aclgraph=torch.npu.NPUGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            with torch.npu.graph(
                cudagraph_metadata.aclgraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for _, value in sorted(results)]
                cudagraph_metadata.outputs = self._merge_outputs(
                    sorted_results,
                    ubatch_metadata,
                )
            self.cudagraphs[num_tokens] = cudagraph_metadata
        get_forward_context().dbo_enabled = True
        return cudagraph_metadata.outputs


def _torch_cat(values: list[Any], dim: int) -> Any:
    import torch

    return torch.cat(values, dim=dim)


def _enable_sp(vllm_config: Any) -> bool:
    try:
        from vllm_ascend.utils import enable_sp

        return bool(enable_sp(vllm_config))
    except Exception:
        return False


__all__ = [
    "AscendNPUGraphMetaData",
    "AscendUBatchWrapper",
    "AscendUbatchMetadata",
]
