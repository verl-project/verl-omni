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

"""BAGEL text likelihood replay alongside the existing diffusion forward."""

import os

import torch
from safetensors import safe_open
from torch import nn

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelForTraining

from .reflection_context import context_layout


class BagelForAlphaGRPO(BagelForTraining):
    """Joint understanding and generation policy; see the BAGEL README Gotchas."""

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=torch.bfloat16):
        """Load both MoT pathways and reject checkpoints without the language head."""
        with safe_open(os.path.join(model_path, "ema.safetensors"), framework="pt") as checkpoint:
            if "language_model.lm_head.weight" not in checkpoint.keys():
                raise ValueError("AlphaGRPO requires language_model.lm_head.weight in the BAGEL checkpoint.")
        return super().from_pretrained(model_path, torch_dtype=torch_dtype)

    def forward(
        self,
        *,
        thinking_input_ids=None,
        thinking_attention_mask=None,
        thinking_labels=None,
        thinking_mask=None,
        thinking_temperature=1.0,
        condition_image_embeds=None,
        condition_image_layout=None,
        **kwargs,
    ):
        """Replay masked text log-probs [B, L], or delegate the image velocity forward."""
        if condition_image_embeds is not None:
            return self._forward_reflection(
                condition_image_embeds,
                condition_image_layout,
                thinking_input_ids,
                thinking_attention_mask,
                thinking_labels,
                thinking_mask,
                thinking_temperature,
                **kwargs,
            )
        if thinking_input_ids is None:
            return super().forward(**kwargs)
        sequence = self.embed_tokens(thinking_input_ids)
        batch_size, length = thinking_input_ids.shape
        positions = torch.arange(length, device=sequence.device).expand(batch_size, -1)
        text_mask = torch.ones_like(thinking_input_ids, dtype=torch.bool)
        latent_mask = ~text_mask
        for layer in self.layers:
            sequence = self._checkpointed_call(
                layer, sequence, positions, text_mask, latent_mask, length, thinking_attention_mask
            )
        selected = self.norm(sequence[thinking_mask])
        logits = self.lm_head(selected).float() / thinking_temperature  # [N_response, V]
        labels = thinking_labels[thinking_mask]
        log_probs = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
        return log_probs.new_zeros(thinking_input_ids.shape).masked_scatter(thinking_mask, log_probs)

    def _forward_reflection(
        self, image_embeds, image_layout, thinking_ids, thinking_valid, labels, action_mask, temperature, **kwargs
    ):
        """Replay frozen visual inputs through trainable MoT layers, including both image blocks."""
        if any(
            p.requires_grad
            for module in (self.embed_tokens, self.vae2llm, self.time_embedder, self.latent_pos_embed)
            for p in module.parameters()
        ):
            raise ValueError(
                "Image replay requires frozen token embeddings, VAE projection, time and position embeddings."
            )
        token_ids = thinking_ids if thinking_ids is not None else kwargs["text_token_ids"]
        valid = thinking_valid if thinking_ids is not None else kwargs["text_attention_mask"]
        outputs = []
        for row in range(token_ids.shape[0]):
            length = int(valid[row].sum())
            start, vae_length, vit_length = image_layout[row].tolist()
            sequence = self.embed_tokens(token_ids[row : row + 1, :length])
            image_length = vae_length + vit_length
            if image_length:
                sequence = sequence.clone()
                sequence[:, start : start + image_length] = image_embeds[row, :image_length].to(sequence.dtype)
            target_length = 0
            if thinking_ids is None:
                latents = kwargs["hidden_states"][row : row + 1]
                target = self.vae2llm(latents) + self.time_embedder(kwargs["timestep"][row : row + 1])[:, None]
                target = target + self.latent_pos_embed(kwargs["latent_pos_ids"][row : row + 1])
                boundaries = self.embed_tokens(
                    token_ids.new_tensor([[self.config.start_of_image_id, self.config.end_of_image_id]])
                )
                sequence = torch.cat((sequence, boundaries[:, :1], target.to(sequence.dtype), boundaries[:, 1:]), dim=1)
                target_length = latents.shape[1] + 2
            positions, generation, visible = context_layout(
                length, start, vae_length, vit_length, sequence.device, target_length
            )
            for layer in self.layers:
                sequence = self._checkpointed_call(
                    layer, sequence, positions[None], ~generation[None], generation[None], 0, None, visible[None, None]
                )
            if thinking_ids is not None:
                mask = action_mask[row, :length]
                logits = self.lm_head(self.norm(sequence[0, mask])).float() / temperature
                log_probs = logits.gather(-1, labels[row, :length][mask, None]).squeeze(-1) - logits.logsumexp(-1)
                output = log_probs.new_zeros(token_ids.shape[1])
                output[:length] = output[:length].masked_scatter(mask, log_probs)
                outputs.append(output[None])
            else:
                outputs.append(self.llm2vae(self.norm_moe_gen(sequence[:, length + 1 : -1])))
        result = torch.cat(outputs)
        return result if thinking_ids is not None else (result,)
