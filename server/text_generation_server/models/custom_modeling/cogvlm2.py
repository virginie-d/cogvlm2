# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Llava-NeXT model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from argparse import Namespace
from transformers.activations import ACT2FN
from transformers.image_processing_utils import select_best_resolution

from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
    load_vision_model,
)
from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)

from loguru import logger



class Cogvlm2ForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        vision_config = Namespace(**config.vision_config)
        vision_config.quantize = None
        # Instead of selecting in hidden_states[-2].
        # Instead compute only the n -2 + 1 layers and don't pool
        logger.info(f"vision_config is {vision_config}, model type is {vision_config.model_type}")
        self.vision_tower = load_vision_model(
            prefix="vision_tower",
            config=vision_config,
            weights=weights,
        )

        self.vocab_size = config.vocab_size
        
        self.language_model = load_text_model(
            prefix="language_model",
            config=config,
            weights=weights,
        )
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        mask = input_ids == self.config.pad_token_id
        # Let's pray we have enabled enough slots !
        try:
            inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
        except Exception as e:
            raise RuntimeError(
                f"Cannot fill images right now. If error happens at warmup, make sure you have enough `--max-input-tokens`  to handle images. If error happens at regular runtime, please fill in an issue: {e}"
            )
        return inputs_embeds, mask

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
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
        pixel_values: torch.FloatTensor = None,
        # Unused for this model
        pixel_attention_mask=None,
        image_sizes: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None and len(pixel_values) > 0:

            num_images, _, _, _ = pixel_values.shape
            
            image_features = self.vision_tower(pixel_values)
            
            inputs_embeds, vision_mask = self._merge_input_ids_with_image_features(
                input_ids, inputs_embeds, image_features
            )
            for i in range(num_images):
                cu_seqlen_start = cu_seqlen_prefill[i]
                vision_mask[cu_seqlen_start + 1 + 1 + 2304] = False
        else:
            vision_mask = None
        
        hidden_states = self.language_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            vision_mask=vision_mask,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            true_max_s=max_s,
            prefill_cache_indices=None,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.language_model.lm_head(hidden_states)
        return logits, speculative_logits
