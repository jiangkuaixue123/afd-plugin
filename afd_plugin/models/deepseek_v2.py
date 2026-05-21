# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 AFD E2E model wrapper.

This is a deliberately small Phase 4 wrapper for 1A1F end-to-end validation.
It does not prune weights by role.
Both Attention and FFN sides load the full model; only the forward path is
split so hidden states can travel through the AFD connector.
"""

from itertools import islice
from typing import Any

import torch
from vllm.model_executor.models import deepseek_v2 as native

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDConnectorMetadata
from afd_plugin.models import get_afd_metadata_from_forward_context
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


class AFDDeepseekV2DecoderLayer(native.DeepseekV2DecoderLayer):
    """DeepSeek decoder layer with separable Attention and FFN execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role if afd_config.enabled else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs: dict[str, Any] = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, native.DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        if self.afd_role == "attention":
            return hidden_states, residual

        hidden_states = self.mlp(hidden_states)
        if (
            isinstance(self.mlp, native.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
        return hidden_states, residual

    def compute_attn_output(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs: dict[str, Any] = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, native.DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        return hidden_states, residual

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.mlp(hidden_states)
        if (
            isinstance(self.mlp, native.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
        return hidden_states


@native.support_torch_compile
class AFDDeepseekV2Model(torch.nn.Module):
    """DeepSeek model wrapper that routes Attention outputs through AFD."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = native.current_platform.device_type

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV2DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                config.hidden_size,
            )
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: native.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | native.IntermediateTensors:
        if native.get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to AFDDeepseekV2Model.forward",
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling = self._get_llama_4_scaling(positions)
        afd_metadata = get_afd_metadata_from_forward_context()

        aux_hidden_states = []
        if afd_metadata is not None:
            if self.aux_hidden_state_layers:
                raise RuntimeError(
                    "AFD DeepSeekV2 E2E wrapper does not support aux hidden "
                    "state capture yet",
                )
            hidden_states, residual = self.forward_with_afd(
                hidden_states,
                residual,
                positions,
                afd_metadata,
                llama_4_scaling,
            )
        else:
            for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
            ):
                if idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    llama_4_scaling,
                )

        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual},
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def forward_with_afd(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        afd_metadata: object,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        afd_connector = afd_metadata.afd_connector
        stage_idx = int(getattr(afd_metadata, "afd_stage_idx", 0))

        for layer_offset, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
        ):
            if layer_offset > 0:
                hidden_states = afd_connector.recv_ffn_output(
                    ref_tensor=hidden_states,
                    ubatch_idx=stage_idx,
                )

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            metadata = AFDConnectorMetadata.create_attention_metadata(
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
            afd_connector.send_attn_output(hidden_states, metadata)
            hidden_states = maybe_apply_dbo_yield(
                hidden_states,
                role="attention",
            )

        hidden_states = afd_connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )
        return hidden_states, residual

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        output = self.layers[layer_idx].compute_ffn_output(hidden_states)
        return output

    def _get_llama_4_scaling(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor | None:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is None:
            return None
        return native._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )


class AFDDeepseekV2ForCausalLM(native.DeepseekV2ForCausalLM):
    """DeepSeekV2 causal LM wrapper for AFD Phase 4 E2E smoke tests."""

    model_cls = AFDDeepseekV2Model

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)


class AFDDeepseekForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


class AFDDeepseekV3ForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


class AFDGlmMoeDsaForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


__all__ = [
    "AFDDeepseekForCausalLM",
    "AFDDeepseekV2ForCausalLM",
    "AFDDeepseekV3ForCausalLM",
    "AFDGlmMoeDsaForCausalLM",
]
