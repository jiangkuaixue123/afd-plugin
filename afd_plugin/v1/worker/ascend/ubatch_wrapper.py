# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU-specific AFD ubatch wrapper.

This is adapted from the original in-tree NPU AFD
``vllm_ascend.worker.npu_ubatch_wrapper`` and kept under the plugin's Ascend
runtime so the GPU wrapper can remain CUDA-specific.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from afd_plugin.connectors import AFDMetadata
from afd_plugin.v1.worker._optional import optional_class
from afd_plugin.v1.worker.ubatch_wrapper import (
    build_ubatch_additional_kwargs,
    build_ubatch_afd_metadata,
    build_ubatch_dp_metadata_list,
)

_GPUUBatchWrapper, _GPUUBatchWrapper_IMPORT_ERROR = optional_class(
    "vllm.v1.worker.gpu_ubatch_wrapper",
    "UBatchWrapper",
)


@dataclass
class ACLGraphMetaData:
    aclgraph: Any
    ubatch_metadata: Any
    outputs: Any = None


class _EmptyContextManager:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False


class AFDNPUUBatchWrapper(_GPUUBatchWrapper):  # type: ignore[misc, valid-type]
    """AFD-aware ubatch wrapper for vLLM-Ascend model runner v1."""

    vllm_base_import_error = _GPUUBatchWrapper_IMPORT_ERROR

    def __init__(
        self,
        runnable: Any,
        vllm_config: object,
        runtime_mode: Any,
        device: object,
    ) -> None:
        if _GPUUBatchWrapper_IMPORT_ERROR is not None:
            raise RuntimeError(
                "AFDNPUUBatchWrapper requires an importable vLLM runtime",
            ) from _GPUUBatchWrapper_IMPORT_ERROR

        import torch

        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(
            int(self.vllm_config.parallel_config.num_ubatches) + 1,
        )
        self.aclgraphs: dict[int, ACLGraphMetaData] = {}
        self.cudagraphs = self.aclgraphs
        self.aclgraph_wrapper = None
        self.cudagraph_wrapper = None
        self._graph_pool = None
        self.sm_control = _EmptyContextManager()
        self.device = device
        self.is_debugging_mode = _vllm_logging_level() == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        if _runtime_mode_name(runtime_mode) != "NONE":
            from vllm.platforms import current_platform
            from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

            self.aclgraph_wrapper = ACLGraphWrapper(
                runnable,
                vllm_config,
                runtime_mode=runtime_mode,
            )
            self.cudagraph_wrapper = self.aclgraph_wrapper
            self._graph_pool = current_platform.get_global_graph_pool()
        _print_npu_ubatch(
            "initialized",
            runtime_mode=runtime_mode,
            num_ubatches=self.vllm_config.parallel_config.num_ubatches,
        )

    @staticmethod
    def _create_sm_control_context(vllm_config: object) -> object:
        del vllm_config
        return _EmptyContextManager()

    @property
    def graph_pool(self) -> Any:
        return self._graph_pool

    def clear_graphs(self) -> None:
        self.aclgraphs.clear()
        if self.aclgraph_wrapper is not None:
            self.aclgraph_wrapper.clear_graphs()

    def _capture_ubatches(self, ubatch_metadata: list[Any], model: Any) -> Any:
        import torch
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.forward_context import get_forward_context, override_forward_context
        from vllm.platforms import current_platform

        _print_npu_ubatch(
            "capture ubatches",
            token_counts=[metadata.num_tokens for metadata in ubatch_metadata],
        )

        @torch.inference_mode()
        def _capture_ubatch_thread(
            results: list[tuple[int, Any]],
            metadata: Any,
        ) -> None:
            torch.npu.set_device(self.device)
            ubatch_context = metadata.context
            ubatch_context.forward_context.capturing = True
            with ubatch_context:
                model_output = model(
                    input_ids=metadata.input_ids,
                    positions=metadata.positions,
                    intermediate_tensors=metadata.intermediate_tensors,
                    inputs_embeds=metadata.inputs_embeds,
                )
            results.append((metadata.context.id, model_output))

        with _torch_cuda_wrapper():
            results: list[tuple[int, Any]] = []
            compute_stream = ubatch_metadata[0].context.compute_stream
            num_tokens = sum(int(metadata.num_tokens) for metadata in ubatch_metadata)
            forward_context = get_forward_context()
            with override_forward_context(None):
                ubatch_threads = [
                    threading.Thread(
                        target=_capture_ubatch_thread,
                        args=(results, metadata),
                    )
                    for metadata in ubatch_metadata
                ]
                for thread in ubatch_threads:
                    thread.start()
                self.ready_barrier.wait()
                aclgraph_metadata = ACLGraphMetaData(
                    aclgraph=torch.npu.NPUGraph(),
                    ubatch_metadata=ubatch_metadata,
                )
                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())
                forward_context.capturing = True
                with torch.npu.graph(
                    aclgraph_metadata.aclgraph,
                    stream=compute_stream,
                    pool=self.graph_pool,
                ):
                    ubatch_metadata[0].context.cpu_wait_event.set()
                    for thread in ubatch_threads:
                        thread.join()
                    sorted_results = [value for position, value in sorted(results)]
                    aclgraph_metadata.outputs = torch.cat(sorted_results, dim=0)
                self.aclgraphs[num_tokens] = aclgraph_metadata
            return aclgraph_metadata.outputs

    def _run_ubatches(self, ubatch_metadata: list[Any], model: Any) -> Any:
        import torch
        from vllm.forward_context import override_forward_context

        _print_npu_ubatch(
            "run ubatches eagerly",
            token_counts=[metadata.num_tokens for metadata in ubatch_metadata],
        )

        @torch.inference_mode()
        def _ubatch_thread(results: list[tuple[int, Any]], metadata: Any) -> None:
            torch.npu.set_device(self.device)
            _print_npu_ubatch(
                "ubatch thread waiting",
                ubatch_idx=metadata.context.id,
                num_tokens=metadata.num_tokens,
            )
            with metadata.context:
                _print_npu_ubatch(
                    "ubatch thread running",
                    ubatch_idx=metadata.context.id,
                    num_tokens=metadata.num_tokens,
                )
                model_output = model(
                    input_ids=metadata.input_ids,
                    positions=metadata.positions,
                    intermediate_tensors=metadata.intermediate_tensors,
                    inputs_embeds=metadata.inputs_embeds,
                )
            results.append((metadata.context.id, model_output))
            _print_npu_ubatch(
                "ubatch thread finished",
                ubatch_idx=metadata.context.id,
            )

        with _torch_cuda_wrapper():
            results: list[tuple[int, Any]] = []
            with override_forward_context(None):
                ubatch_threads = [
                    threading.Thread(
                        target=_ubatch_thread,
                        args=(results, metadata),
                    )
                    for metadata in ubatch_metadata
                ]
                for thread in ubatch_threads:
                    thread.start()
                self.ready_barrier.wait()
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                return torch.cat(sorted_results, dim=0)

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
    ) -> tuple[Any, Any, Any, Any]:
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )
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

    def _make_afd_ubatch_metadata(
        self,
        ubatch_slices: Any,
        attn_metadata: Any,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
        dp_metadata: Any,
        afd_metadata: AFDMetadata,
    ) -> AFDMetadata:
        if ubatch_slices is None:
            afd_metadata.input_ids_list.append(input_ids)
            afd_metadata.positions_list.append(positions)
            afd_metadata.inputs_embeds_list.append(inputs_embeds)
            afd_metadata.intermediate_tensors_list.append(intermediate_tensors)
            afd_metadata.attn_metadata_list.append(attn_metadata)
            afd_metadata.dp_metadata_list.append(dp_metadata)
            return afd_metadata

        ubatch_dp_metadata = build_ubatch_dp_metadata_list(
            self.vllm_config,
            ubatch_slices,
        )
        for idx, ubatch_slice in enumerate(ubatch_slices):
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
            afd_metadata.input_ids_list.append(sliced_input_ids)
            afd_metadata.positions_list.append(sliced_positions)
            afd_metadata.inputs_embeds_list.append(sliced_inputs_embeds)
            afd_metadata.intermediate_tensors_list.append(sliced_intermediate_tensors)
            afd_metadata.attn_metadata_list.append(
                _select_ubatch_attn_metadata(attn_metadata, idx),
            )
            afd_metadata.dp_metadata_list.append(ubatch_dp_metadata[idx])
        return afd_metadata

    def _make_ubatch_metadata(
        self,
        ubatch_slices: Any,
        attn_metadata: Any,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
        compute_stream: Any,
        dp_metadata: Any,
        batch_descriptor: Any,
        aclgraph_runtime_mode: Any,
        afd_metadata: AFDMetadata | None,
    ) -> list[Any]:
        import torch
        from vllm.forward_context import get_forward_context
        from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata
        from vllm.v1.worker.ubatching import make_ubatch_contexts

        parent_forward_context = get_forward_context()
        parent_additional_kwargs = dict(parent_forward_context.additional_kwargs)
        ubatch_dp_metadata = (
            afd_metadata.dp_metadata_list
            if afd_metadata is not None
            and len(afd_metadata.dp_metadata_list) == len(ubatch_slices)
            else build_ubatch_dp_metadata_list(self.vllm_config, ubatch_slices)
        )

        forward_contexts = []
        for idx, ubatch_slice in enumerate(ubatch_slices):
            forward_context = copy.copy(parent_forward_context)
            child_afd_metadata = (
                build_ubatch_afd_metadata(afd_metadata, ubatch_slices, idx)
                if afd_metadata is not None
                else None
            )
            forward_context.dp_metadata = ubatch_dp_metadata[idx]
            forward_context.ubatch_idx = idx
            forward_context.attn_metadata = _select_ubatch_attn_metadata(
                attn_metadata,
                idx,
            )
            forward_context.no_compile_layers = (
                self.vllm_config.compilation_config.static_forward_context
            )
            forward_context.cudagraph_runtime_mode = aclgraph_runtime_mode
            forward_context.batch_descriptor = batch_descriptor
            forward_context.afd_metadata = child_afd_metadata
            forward_context.additional_kwargs = build_ubatch_additional_kwargs(
                parent_additional_kwargs,
                child_afd_metadata,
            )
            forward_context.num_ubatches = len(ubatch_slices)
            forward_context.num_tokens = int(ubatch_slice.num_tokens)
            forward_context.afd_comm_event = torch.npu.Event()
            forward_contexts.append(forward_context)
            _print_npu_ubatch(
                "build child forward context",
                ubatch_idx=idx,
                token_start=ubatch_slice.token_slice.start,
                num_tokens=ubatch_slice.num_tokens,
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        has_afd_sliced_inputs = (
            afd_metadata is not None
            and len(afd_metadata.input_ids_list) == len(ubatch_slices)
            and len(afd_metadata.positions_list) == len(ubatch_slices)
            and len(afd_metadata.inputs_embeds_list) == len(ubatch_slices)
            and len(afd_metadata.intermediate_tensors_list) == len(ubatch_slices)
        )
        ubatch_metadata = []
        for idx, ubatch_slice in enumerate(ubatch_slices):
            if has_afd_sliced_inputs:
                sliced_input_ids = afd_metadata.input_ids_list[idx]
                sliced_positions = afd_metadata.positions_list[idx]
                sliced_inputs_embeds = afd_metadata.inputs_embeds_list[idx]
                sliced_intermediate_tensors = afd_metadata.intermediate_tensors_list[
                    idx
                ]
            else:
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
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[idx],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=int(ubatch_slice.num_tokens),
                ),
            )
        return ubatch_metadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        with _torch_cuda_wrapper():
            forward_context = get_forward_context()
            batch_descriptor = forward_context.batch_descriptor
            ubatch_slices = forward_context.ubatch_slices
            cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
            afd_metadata = _get_forward_context_afd_metadata(forward_context)
            _print_npu_ubatch(
                "__call__",
                runtime_mode=cudagraph_runtime_mode,
                ubatches=None
                if ubatch_slices is None
                else [ubatch.num_tokens for ubatch in ubatch_slices],
                has_afd_metadata=afd_metadata is not None,
            )

            if ubatch_slices is None:
                if cudagraph_runtime_mode in (
                    CUDAGraphMode.NONE,
                    CUDAGraphMode.PIECEWISE,
                ):
                    return self.runnable(*args, **kwargs)
                if self.aclgraph_wrapper is not None:
                    return self.aclgraph_wrapper(*args, **kwargs)
                return self.runnable(*args, **kwargs)

            attn_metadata = forward_context.attn_metadata
            num_tokens = sum(int(ubatch.num_tokens) for ubatch in ubatch_slices)
            input_ids = kwargs["input_ids"]
            positions = kwargs["positions"]
            intermediate_tensors = kwargs["intermediate_tensors"]
            inputs_embeds = kwargs["inputs_embeds"]
            compute_stream = torch.npu.current_stream()
            dp_metadata = forward_context.dp_metadata

            if afd_metadata is not None:
                afd_metadata = self._make_afd_ubatch_metadata(
                    ubatch_slices=ubatch_slices,
                    attn_metadata=attn_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    intermediate_tensors=intermediate_tensors,
                    dp_metadata=dp_metadata,
                    afd_metadata=afd_metadata,
                )
                forward_context.afd_metadata = afd_metadata
                forward_context.additional_kwargs["afd_metadata"] = afd_metadata

            if (
                num_tokens not in self.aclgraphs
                and cudagraph_runtime_mode is CUDAGraphMode.FULL
            ):
                ubatch_metadata = self._make_ubatch_metadata(
                    ubatch_slices=ubatch_slices,
                    attn_metadata=attn_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    compute_stream=compute_stream,
                    dp_metadata=dp_metadata,
                    batch_descriptor=batch_descriptor,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    afd_metadata=afd_metadata,
                )
                return self._capture_ubatches(ubatch_metadata, self.runnable)

            if (
                num_tokens in self.aclgraphs
                and cudagraph_runtime_mode is CUDAGraphMode.FULL
            ):
                aclgraph_metadata = self.aclgraphs[num_tokens]
                _print_npu_ubatch("replay ubatch graph", num_tokens=num_tokens)
                aclgraph_metadata.aclgraph.replay()
                return aclgraph_metadata.outputs

            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=dp_metadata,
                batch_descriptor=batch_descriptor,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                afd_metadata=afd_metadata,
            )
            return self._run_ubatches(ubatch_metadata, self.runnable)


@contextmanager
def _torch_cuda_wrapper():
    import torch

    original_cuda = getattr(torch, "cuda", None)
    original_attrs = {}
    if original_cuda is not None:
        for name in (
            "Event",
            "Stream",
            "default_stream",
            "current_stream",
            "stream",
            "set_stream",
        ):
            original_attrs[name] = getattr(original_cuda, name, None)
    try:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.set_stream = torch.npu.set_stream
        yield
    finally:
        if original_cuda is not None:
            for name, value in original_attrs.items():
                if value is None:
                    delattr(torch.cuda, name)
                else:
                    setattr(torch.cuda, name, value)
        else:
            del torch.cuda


class _EventPlaceholder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def record(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def synchronize(self) -> None:
        return None


def _get_forward_context_afd_metadata(forward_context: object) -> Any:
    additional_kwargs = getattr(forward_context, "additional_kwargs", {}) or {}
    metadata = additional_kwargs.get("afd_metadata")
    if metadata is not None:
        return metadata
    return getattr(forward_context, "afd_metadata", None)


def _select_ubatch_attn_metadata(attn_metadata: Any, idx: int) -> Any:
    if isinstance(attn_metadata, (list, tuple)):
        return attn_metadata[idx]
    return attn_metadata


def _runtime_mode_name(runtime_mode: object) -> str:
    name = getattr(runtime_mode, "name", None)
    if isinstance(name, str):
        return name
    return str(runtime_mode).rsplit(".", 1)[-1]


def _vllm_logging_level() -> str:
    import vllm.envs as envs

    return envs.VLLM_LOGGING_LEVEL


def _print_npu_ubatch(message: str, **fields: Any) -> None:
    del message, fields


__all__ = ["AFDNPUUBatchWrapper"]
