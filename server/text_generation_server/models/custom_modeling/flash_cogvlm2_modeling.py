# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, List, Tuple

from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)
from loguru import logger
if SYSTEM == "rocm":
    try:
        from vllm import _custom_C
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_C`. Full error: {e}")



class FlashCogvlm2Attention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=500000,
            device=weights.device,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )
        
        
        self.query_key_value = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.language_expert_query", f"{prefix}.language_expert_key", f"{prefix}.language_expert_value"],
            dim=0,
            weights=weights,
            bias=False,
        )

        self.vision_query_key_value = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.vision_expert_query", f"{prefix}.vision_expert_key", f"{prefix}.vision_expert_value"],
            dim=0,
            weights=weights,
            bias=True,
        )
        
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.language_expert_dense",
            weights=weights,
            bias=False,
        )
        
        self.vision_o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.vision_expert_dense",
            weights=weights,
            bias=False,
        )
        
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        vision_mask,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        # logger.info(f"vision_mask {vision_mask}")
    
        if cu_seqlen_prefill is not None:
            total_len = hidden_states.shape[0]
            qkv_hidden_size = self.head_size * (self.num_heads + 2 * self.num_key_value_heads)
            qkv = torch.empty(total_len, qkv_hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
            
            text_hidden_states = hidden_states[~vision_mask]
            vision_hidden_states = hidden_states[vision_mask]
            vision_qkv = self.vision_query_key_value(vision_hidden_states)
            
            text_qkv = self.query_key_value(text_hidden_states)
            
            qkv[vision_mask] = vision_qkv
            qkv[~vision_mask] = text_qkv
        else:
            qkv = self.query_key_value(hidden_states)
            
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        paged_attention.reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attention.attention(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )
        
        if cu_seqlen_prefill is not None:
            o = torch.empty_like(hidden_states)
            text_o = self.o_proj(
                attn_output[~vision_mask].view(-1, self.num_heads * self.head_size))
            vision_o = self.vision_o_proj(
                attn_output[vision_mask].view(-1, self.num_heads * self.head_size))
            o[~vision_mask] = text_o
            o[vision_mask] = vision_o
        else:
            o = self.o_proj(
                attn_output.view(-1, self.num_heads * self.head_size))

        return o


class Cogvlm2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.act = (
            ACT2FN[self.hidden_act]
            if "gelu" not in self.hidden_act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh"
                    if self.hidden_act in ["gelu_fast", "gelu_pytorch_tanh"]
                    else "none"
                ),
            )
        )
        
        # Fuse gate and up proj
        bias = getattr(config, "mlp_bias", False)
        
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.language_mlp.gate_proj", f"{prefix}.language_mlp.up_proj"],
                weights=weights,
                dim=0,
                bias=bias,
            )
        self.vision_gate_up_proj = TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.vision_mlp.gate_proj", f"{prefix}.vision_mlp.up_proj"],
                weights=weights,
                dim=0,
                bias=bias,
            )
        
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.language_mlp.down_proj",
            weights=weights,
            bias=bias,
        )
        self.vision_down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.vision_mlp.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

        # TODO: This is a hotfix to be removed & properly refactored.
        self.quantize = config.quantize

    def forward(self, hidden_states, vision_mask):
        
        if vision_mask is not None:
            o = torch.empty_like(hidden_states)
            text_gate_up_states = self.gate_up_proj(hidden_states[~vision_mask])
            vision_gate_up_states = self.vision_gate_up_proj(hidden_states[vision_mask])
            if text_gate_up_states.shape[0] > 0:
                text_gate_up_states = text_gate_up_states.view(-1, 2, self.intermediate_size)
                text_o = self.down_proj(self.act(text_gate_up_states[:, 0]) * text_gate_up_states[:, 1])
                o[~vision_mask] = text_o
            if vision_gate_up_states.shape[0] > 0:
                vision_gate_up_states = vision_gate_up_states.view(-1, 2, self.intermediate_size)
                vision_o = self.vision_down_proj(self.act(vision_gate_up_states[:, 0]) * vision_gate_up_states[:, 1])
                o[vision_mask] = vision_o
        else:   
            gate_up_states = self.gate_up_proj(hidden_states)
            gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
            o = self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])
        return o


class FlashCogvlm2Layer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.self_attn = FlashCogvlm2Attention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights
        )
        self.mlp = Cogvlm2MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        vision_mask,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            vision_mask,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        
        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        if cu_seqlen_prefill is not None:
            mlp_output = self.mlp(normed_attn_res_output, vision_mask)
        else:
            mlp_output = self.mlp(normed_attn_res_output, None)
        
        return mlp_output, attn_res


class FlashCogvlm2Model(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.layers = nn.ModuleList(
            [
                FlashCogvlm2Layer(
                    prefix=(
                        f"{prefix}.model.layers.{layer_id}"
                    ),
                    config=config,
                    weights=weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.model.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        vision_mask,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        true_max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = inputs_embeds

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                vision_mask,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )
            
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class FlashCogvlm2ForCausalLM(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        
        self.embed_tokens = TensorParallelEmbedding(
            prefix=(
                f"{prefix}.model.embed_tokens"
            ),
            weights=weights,
        )
        self.model = FlashCogvlm2Model(prefix, config, weights)
        
        self.lm_head = SpeculativeHead.load(
            config,
            prefix=f"{prefix}.lm_head",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            true_max_s=max_s,
            prefill_cache_indices=prefill_cache_indices,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
