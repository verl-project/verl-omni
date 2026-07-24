# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""BAGEL model wrapper for Uni-COT supervised fine-tuning.

This module deliberately lives outside ``bagel_flow_grpo`` so the existing
FlowGRPO velocity-replay contract remains unchanged.  ``BagelForSFT`` reuses
the BAGEL MoT backbone and adds the text head and SFT output contract required
for interleaved Uni-COT supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import (
    BagelForTraining,
    BagelTrainingConfig,
    _map_checkpoint_to_training,
)

__all__ = ["BagelSFTOutput", "BagelForSFT"]


@dataclass
class BagelSFTOutput:
    """Outputs consumed by the BAGEL SFT loss."""

    logits: Tensor
    image_velocity: Optional[Tensor] = None


class BagelForSFT(BagelForTraining):
    """FSDP-compatible BAGEL module for supervised text + image training.

    The image branch delegates to the existing BAGEL denoising forward and
    accepts multiple target image spans as ``(B, N, L, D)``.  Text CE is
    produced by a lightweight LM head over the MoT text pathway.
    """

    _no_split_modules = ["BagelMoTLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: BagelTrainingConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_lm_head()

    def tie_lm_head(self) -> None:
        """Tie token embeddings and output projection when shapes match."""

        if self.lm_head.weight.shape == self.embed_tokens.weight.shape:
            self.lm_head.weight = self.embed_tokens.weight

    def _forward_text_logits(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, L = input_ids.shape
        device = input_ids.device
        sequence = self.embed_tokens(input_ids)
        text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        latent_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

        key_padding_mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            key_padding_mask = attention_mask.to(device=device, dtype=torch.bool)

        for layer in self.layers:

            def _layer_fn(seq, pos_ids, text_mask_, latent_mask_, kpm, *, _layer=layer):
                return _layer(seq, pos_ids, text_mask_, latent_mask_, L, key_padding_mask=kpm)

            sequence = self._checkpointed_call(
                _layer_fn, sequence, position_ids, text_mask, latent_mask, key_padding_mask
            )

        hidden = self.norm(sequence)
        return self.lm_head(hidden)

    def _forward_image_velocity(
        self,
        image_hidden_states: Optional[Tensor],
        timesteps: Optional[Tensor],
        text_token_ids: Optional[Tensor],
        latent_pos_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        if image_hidden_states is None:
            return None
        if timesteps is None or latent_pos_ids is None:
            raise ValueError("BAGEL SFT image supervision requires timesteps and latent_pos_ids.")

        original_shape = image_hidden_states.shape
        if image_hidden_states.ndim == 4:
            batch_size, num_images, latent_len, latent_dim = original_shape
            flat_hidden = image_hidden_states.reshape(batch_size * num_images, latent_len, latent_dim)
            flat_timesteps = timesteps.reshape(batch_size * num_images)
            if latent_pos_ids.ndim == 2:
                flat_pos = latent_pos_ids.unsqueeze(1).expand(batch_size, num_images, -1)
            else:
                flat_pos = latent_pos_ids
            flat_pos = flat_pos.reshape(batch_size * num_images, latent_len)
            if text_token_ids is not None:
                text_token_ids = (
                    text_token_ids.unsqueeze(1).expand(batch_size, num_images, -1).reshape(batch_size * num_images, -1)
                )
            if attention_mask is not None:
                attention_mask = (
                    attention_mask.unsqueeze(1).expand(batch_size, num_images, -1).reshape(batch_size * num_images, -1)
                )
            velocity = super().forward(
                hidden_states=flat_hidden,
                timestep=flat_timesteps,
                text_token_ids=text_token_ids,
                latent_pos_ids=flat_pos,
                text_attention_mask=attention_mask,
            )[0]
            return velocity.reshape(batch_size, num_images, latent_len, latent_dim)

        return super().forward(
            hidden_states=image_hidden_states,
            timestep=timesteps,
            text_token_ids=text_token_ids,
            latent_pos_ids=latent_pos_ids,
            text_attention_mask=attention_mask,
        )[0]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        image_hidden_states: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        latent_pos_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> BagelSFTOutput:
        """Run BAGEL SFT forward.

        Args:
            input_ids: Text tokens for CE and image conditioning.
            attention_mask: Text padding mask.
            image_hidden_states: Optional noisy target latents, either
                ``(B, L, D)`` or ``(B, N, L, D)`` for multiple Uni-COT diagrams.
            timesteps: Flow timesteps for image spans.
            latent_pos_ids: Latent patch position IDs.
        """

        del kwargs
        logits = self._forward_text_logits(input_ids=input_ids, attention_mask=attention_mask)
        image_velocity = self._forward_image_velocity(
            image_hidden_states=image_hidden_states,
            timesteps=timesteps,
            text_token_ids=input_ids,
            latent_pos_ids=latent_pos_ids,
            attention_mask=attention_mask,
        )
        return BagelSFTOutput(logits=logits, image_velocity=image_velocity)

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16) -> BagelForSFT:
        """Load BAGEL SFT weights from the released ``ema.safetensors`` file."""

        import os

        from safetensors.torch import load_file

        config = BagelTrainingConfig.from_model_path(model_path)
        ckpt_path = os.path.join(model_path, "ema.safetensors")
        state_dict = load_file(ckpt_path)

        model = cls(config)
        mapped = _map_checkpoint_to_training(state_dict, config)
        missing, _unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            import logging

            logging.getLogger(__name__).warning("Missing keys when loading BagelForSFT: %d keys", len(missing))
        model.tie_lm_head()
        return model.to(torch_dtype)
