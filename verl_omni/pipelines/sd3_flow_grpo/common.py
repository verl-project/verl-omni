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

"""Shared SD3.5 helpers: token-id-native prompt encoding for rollout adapters."""

import logging

import torch

logger = logging.getLogger(__name__)

# ``extra_prompt_ids`` entries produced by the agent loop for SD3.5, configured
# via ``actor_rollout_ref.model.extra_tokenizers``. CLIP-L and CLIP-G share one
# vocabulary, so a single CLIP tokenization feeds both CLIP encoders; T5 has
# its own tokenizer.
SD3_CLIP_TOKENS_KEY = "clip"
SD3_T5_TOKENS_KEY = "t5"
SD3_ENCODER_TOKEN_KEYS = (SD3_CLIP_TOKENS_KEY, SD3_T5_TOKENS_KEY)


def pad_token_id_batch(
    ids_list: list[list[int]],
    max_length: int,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Right-pad token id lists to a fixed length (tail-truncating as a safety net)."""
    batch = []
    for ids in ids_list:
        ids = [int(token_id) for token_id in ids]
        if len(ids) > max_length:
            logger.warning("Prompt of %d tokens exceeds the encoder max length %d; truncating.", len(ids), max_length)
            ids = ids[:max_length]
        batch.append(ids + [pad_token_id] * (max_length - len(ids)))
    return torch.tensor(batch, dtype=torch.long, device=device)


class SD3TokenIdPromptMixin:
    """Encode pre-tokenized SD3.5 prompts (CLIP + T5 token ids) for rollout adapters.

    Mirrors the text-based ``encode_prompt`` of the vLLM-Omni SD3 pipeline while
    consuming token ids produced once in the agent loop, so the pipeline never
    decodes ids back to text. Sequences arrive unpadded (with special tokens,
    truncated at tokenization time); each encoder pads with its own tokenizer's
    pad token, matching the ``padding="max_length"`` behaviour of the text path.
    """

    def _get_clip_prompt_embeds_from_ids(
        self,
        prompt_ids: list[list[int]],
        num_images_per_prompt: int = 1,
        clip_model_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        batch_size = len(prompt_ids)
        text_input_ids = pad_token_id_batch(prompt_ids, self.tokenizer_max_length, tokenizer.pad_token_id, self.device)

        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0].to(dtype=self.od_config.dtype, device=self.device)
        prompt_embeds = prompt_embeds.hidden_states[-2].to(dtype=self.od_config.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def _get_t5_prompt_embeds_from_ids(
        self,
        prompt_ids: list[list[int]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
    ) -> torch.Tensor:
        batch_size = len(prompt_ids)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (batch_size * num_images_per_prompt, max_sequence_length, self.transformer.joint_attention_dim),
                device=self.device,
                dtype=self.od_config.dtype,
            )

        text_input_ids = pad_token_id_batch(prompt_ids, max_sequence_length, self.tokenizer_3.pad_token_id, self.device)
        prompt_embeds = self.text_encoder_3(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.od_config.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt_from_token_ids(
        self,
        clip_prompt_ids: list[list[int]],
        t5_prompt_ids: list[list[int]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token-id counterpart of ``encode_prompt``: CLIP-L/G + T5 -> SD3 embeddings."""
        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds_from_ids(
            clip_prompt_ids, num_images_per_prompt=num_images_per_prompt, clip_model_index=0
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds_from_ids(
            clip_prompt_ids, num_images_per_prompt=num_images_per_prompt, clip_model_index=1
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = self._get_t5_prompt_embeds_from_ids(
            t5_prompt_ids, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        return prompt_embeds, pooled_prompt_embeds
