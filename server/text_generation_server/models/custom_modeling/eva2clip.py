from typing import Optional, Tuple, Union

import math
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

from text_generation_server.layers.tensor_parallel import (
    TensorParallelEmbedding,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)
from text_generation_server.layers.linear import get_linear, FastLinear

from argparse import Namespace
import xformers.ops as xops
from apex.normalization.fused_layer_norm import FusedLayerNorm

from loguru import logger

class PatchEmbedding(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size)
        self.proj.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.proj.weight"), requires_grad=False
        )
        self.proj.bias = nn.Parameter(
            weights.get_tensor(f"{prefix}.proj.bias"), requires_grad=False
        )
        
        position_embedding = weights.get_tensor(f"{prefix}.position_embedding.weight")
        
        self.position_embedding = nn.Parameter(position_embedding, requires_grad=False)
        
        cls_embedding = weights.get_tensor(f"{prefix}.cls_embedding.weight")
        self.cls_embedding = nn.Parameter(cls_embedding, requires_grad=False)
        

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        x = self.proj(
            pixel_values
        )  # shape = [*, width, grid, grid]
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.unsqueeze(0)
        
        return x

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.head_size = self.head_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.embed_dim = self.embed_dim // weights.process_group.size()
        self.scale = self.head_dim**-0.5
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)
        
        self.query_key_value = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.query", f"{prefix}.key", f"{prefix}.value"],
            dim=0,
            weights=weights,
            bias=True,
        )
        
        self.out_proj = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.dense", weights=weights, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        B, L, _ = hidden_states.size()
        # logger.info(f"hidden_states is {hidden_states}")
        qkv = self.query_key_value(hidden_states)
        query, key, value = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
            ],
            dim=-1,
        )

        query = query.view(B, L, self.num_heads, self.head_size)
        key = key.view(B, L, self.num_heads, self.head_size)
        value = value.view(B, L, self.num_heads, self.head_size)
        
        out = xops.memory_efficient_attention(
            query, key, value, scale=self.scale,
        )
        output = self.out_proj(out.view(B, L, -1))
        output = self.output_dropout(output)
        return output
    

class MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(  # config.hidden_size, config.intermediate_size
            prefix=f"{prefix}.fc1", config=config, weights=weights, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(  # config.intermediate_size, config.hidden_size
            prefix=f"{prefix}.fc2", config=config, weights=weights, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class LayerNorm(FusedLayerNorm):
        def __init__(self, prefix, config, weights):
            hidden_size = config.hidden_size
            eps = config.layer_norm_eps
            super().__init__(hidden_size, eps=eps)
            weight = weights.get_tensor(f"{prefix}.weight")
            bias = weights.get_tensor(f"{prefix}.bias")
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)
        def forward(self, x):
            return super().forward(x)


class TransformerLayer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            prefix=f"{prefix}.attention", config=config, weights=weights
        )
        
        self.input_layernorm = LayerNorm(
            prefix=f"{prefix}.input_layernorm", config=config, weights=weights)
        
        self.mlp = MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        
        self.post_attention_layernorm = LayerNorm(
            prefix=f"{prefix}.post_attention_layernorm", config=config, weights=weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor]:
        
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.self_attn(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output
        


class Transformer(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    prefix=f"{prefix}.layers.{i}", config=config, weights=weights
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
    ):

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states

class GLU(nn.Module):
    def __init__(self, prefix, config, weights, in_features):
        super().__init__()
        
        self.linear_proj = nn.Linear(in_features, 4096, bias=False)
        self.linear_proj.weight = nn.Parameter(weights.get_tensor(f"{prefix}.linear_proj.weight"), requires_grad=False)
        self.linear_proj.bias = None
        
        self.norm1 = nn.LayerNorm.load(
            prefix=f"{prefix}.norm1", weights=weights, eps=config.layer_norm_eps
        )
        
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
    
        self.intermediate_size = 14336 // weights.process_group.size()
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.dense_h_to_4h"],
                weights=weights,
                dim=0,
                bias=False,
            )

        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense_4h_to_h",
            weights=weights,
            bias=False,
        )
        
    def forward(self, x):
        B, L, _ = x.shape
        
        x = self.linear_proj(x)
        
        x = self.act1(self.norm1(x))
        
        gate_up_states = self.gate_up_proj(x)

        gate_up_states = gate_up_states.view(B, L, 2, self.intermediate_size)
        x = self.act2(gate_up_states[:,:,0,:]) * gate_up_states[:,:,1,:]
        x = self.down_proj(x)
        return x

class EVA2CLIPModel(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(
            prefix=f"{prefix}.patch_embedding", config=config, weights=weights
        )
        self.transformer = Transformer(
            prefix=f"{prefix}.transformer", config=config, weights=weights
        )
        self.linear_proj = GLU(prefix=f"{prefix}.linear_proj", config=config, weights=weights, in_features=config.hidden_size)
        
        self.conv = nn.Conv2d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=2, stride=2)
        self.conv.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.conv.weight"), requires_grad=False
        )
        self.conv.bias = nn.Parameter(
            weights.get_tensor(f"{prefix}.conv.bias"), requires_grad=False
        )
        boi = weights.get_tensor(f"{prefix}.boi")
        eoi = weights.get_tensor(f"{prefix}.eoi")
        self.boi = nn.Parameter(boi)
        self.eoi = nn.Parameter(eoi)

        self.dtype = weights.dtype
        self.device = weights.device
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Returns:

        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        pixel_values = pixel_values.to(self.dtype).to(self.device)
        
        x = self.patch_embedding(pixel_values)
        
        x = self.transformer(x)
        x = x[:, 1:]
        
        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        
        return x
