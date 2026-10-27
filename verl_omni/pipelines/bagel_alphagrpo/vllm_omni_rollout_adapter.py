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

"""Sample thinking actions and inject exact native KV contexts for image rollout."""

from copy import deepcopy

import numpy as np
import torch
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb, _to_token_list
from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_request import condition_images_from_payload, prompt_ids_from_payload

from .reflection import REFLECTION_CFG

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, "
    "i.e. <think> planning process here </think> image here"
)


@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="alphagrpo")
class BagelAlphaGRPOPipeline(BagelPipelineWithLogProb):
    """Reasoning-to-image and image-conditioned refinement rollout."""

    supports_step_execution = False

    def __init__(self, *, od_config, prefix=""):
        parallel = od_config.parallel_config
        if any(
            size != 1
            for size in (parallel.tensor_parallel_size, parallel.cfg_parallel_size, parallel.sequence_parallel_size)
        ):
            raise ValueError("AlphaGRPO rollout currently requires TP=CFG=SP=1.")
        super().__init__(od_config=od_config, prefix=prefix)

    def load_weights(self, weights):
        """Route the actor's language head to the rollout head rather than the transformer body."""
        loaded = set()
        for name, tensor in weights:
            if name.startswith("transformer.lm_head."):
                loaded |= self.language_model.load_weights([(name.removeprefix("transformer."), tensor)])
            else:
                loaded |= super().load_weights([(name, tensor)])
        return loaded

    def _prefill(self, cache, token_ids, offset=0):
        """Append exact native token IDs [L] to a BAGEL cache."""
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        return self.bagel.forward_cache_update_text(
            past_key_values=cache,
            packed_text_ids=ids,
            packed_text_position_ids=torch.arange(offset, offset + len(token_ids), device=self.device),
            text_token_lens=torch.tensor([len(token_ids)], dtype=torch.int, device=self.device),
        )

    def _sample_thinking(self, cache, prefix_length, max_tokens, temperature, generator):
        """Return sampled labels and matching temperature-scaled old log-probs [T]."""
        current = torch.tensor([self.new_token_ids["bos_token_id"]], device=self.device)
        tokens, log_probs = [], []
        for step in range(max_tokens):
            output = self.bagel.language_model(
                packed_text_ids=current,
                query_lens=torch.ones_like(current),
                packed_query_position_ids=torch.tensor([prefix_length + step], device=self.device),
                past_key_values=cache,
                update_past_key_values=True,
                is_causal=True,
                mode="und",
            )
            cache = output.past_key_values
            logits = self.bagel.language_model.lm_head(output.packed_query_sequence).float() / temperature
            current = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
            tokens.append(int(current.item()))
            log_probs.append(logits.log_softmax(-1).gather(-1, current.unsqueeze(-1)).item())
            if tokens[-1] == self.new_token_ids["eos_token_id"]:
                break
        return tokens, log_probs

    def _prefill_image(self, cache, image, offset):
        """Encode one image into frozen VAE/ViT inputs [N, H] and append native cache blocks."""
        image = image.convert("RGB")
        stride = self.bagel.latent_downsample
        if any(size % stride or size > self.bagel.max_latent_size * stride for size in image.size):
            raise ValueError("Reflection input must use the generated image's stride-aligned BAGEL canvas.")
        sequence_parts, lengths = [], []
        kv_length = offset
        for kind in ("vae", "vit"):
            if kind == "vae":
                transforms = lambda img: torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1
                prepare = self.bagel.prepare_vae_images
            else:
                transforms = lambda img: self.image_processor(images=img, return_tensors="pt").pixel_values[0]
                prepare = self.bagel.prepare_vit_images
            inputs, new_lengths, new_rope = prepare(
                curr_kvlens=[kv_length],
                curr_rope=[offset],
                images=[image],
                transforms=transforms,
                new_token_ids=self.new_token_ids,
            )
            inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            boundary = self.bagel.language_model.forward(
                packed_text_ids=inputs["packed_text_ids"], return_embeddings_only=True
            ).packed_query_sequence
            sequence = boundary.new_zeros((int(inputs["packed_seqlens"].sum()), boundary.shape[-1]))
            sequence[inputs["packed_text_indexes"]] = boundary
            if kind == "vae":
                latent = self.vae.encode(inputs["padded_images"])[0]
                h, w = inputs["patchified_vae_latent_shapes"][0]
                patch = self.bagel.latent_patch_size
                latent = latent[:, : h * patch, : w * patch].reshape(self.bagel.latent_channel, h, patch, w, patch)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, patch * patch * self.bagel.latent_channel)
                embeds = self.bagel.vae2llm(latent) + self.bagel.time_embedder(inputs["packed_timesteps"])
                embeds = embeds + self.bagel.latent_pos_embed(inputs["packed_vae_position_ids"])
                sequence[inputs["packed_vae_token_indexes"]] = embeds.to(sequence.dtype)
                routing = {
                    "mode": "gen",
                    "packed_vae_token_indexes": inputs["packed_vae_token_indexes"],
                    "packed_text_indexes": inputs["packed_text_indexes"],
                }
            else:
                seqlens = inputs["vit_token_seqlens"]
                embeds = self.bagel.vit_model(
                    packed_pixel_values=inputs["packed_vit_tokens"],
                    packed_flattened_position_ids=inputs["packed_vit_position_ids"],
                    cu_seqlens=torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).to(torch.int32),
                    max_seqlen=int(seqlens.max()),
                )
                embeds = self.bagel.connector(embeds) + self.bagel.vit_pos_embed(inputs["packed_vit_position_ids"])
                sequence[inputs["packed_vit_token_indexes"]] = embeds.to(sequence.dtype)
                routing = {"mode": "und"}
            cache = self.bagel.language_model.forward(
                packed_query_sequence=sequence,
                query_lens=inputs["packed_seqlens"],
                packed_query_position_ids=inputs["packed_position_ids"],
                past_key_values=cache,
                update_past_key_values=True,
                is_causal=False,
                **routing,
            ).past_key_values
            sequence_parts.append(sequence)
            lengths.append(sequence.shape[0])
            kv_length, offset = new_lengths[0], new_rope[0]
        return cache, torch.cat(sequence_parts), lengths

    @torch.no_grad()
    def forward(self, req):
        """Record fixed-width text replay tensors and reuse the existing image SDE rollout."""
        if req.is_dummy_run():
            return super().forward(req)
        if len(req.prompts) != 1 or not isinstance(req.prompts[0], dict):
            raise ValueError("AlphaGRPO requires one native token prompt per request.")
        if req.sampling_params.extra_args.get("alphagrpo_stage1", False):
            return super().forward(req)
        prompt = req.prompts[0]
        images = condition_images_from_payload(prompt)
        if images and len(images) != 1:
            raise ValueError("AlphaGRPO refinement requires exactly one initial image.")
        sampling = req.sampling_params
        extra = sampling.extra_args
        if images:
            for key, value in REFLECTION_CFG.items():
                if key in extra and extra[key] != value:
                    raise ValueError(f"Reflection requires the reference {key}={value!r} on both rollout and replay.")
                extra[key] = value
        negative = self._decode_token_prompt(prompt.get("negative_prompt_ids")) or extra.get("negative_prompt", "")
        if negative.strip():
            raise ValueError(
                "AlphaGRPO uses the planning system prompt as its text CFG context, without a negative prompt."
            )
        max_tokens = int(extra.get("max_think_tokens", 128))
        temperature = float(extra.get("think_temperature", 1.0))
        top_p = float(extra.get("think_top_p", 1.0))
        width = int(sampling.max_sequence_length or 2048)
        if max_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("Thinking requires max_think_tokens >= 1, temperature > 0 and 0 < top_p <= 1.")
        if top_p != 1.0:
            raise ValueError("AlphaGRPO requires think_top_p=1 so sampling and replay likelihoods match.")
        bos, eos = self.new_token_ids["bos_token_id"], self.new_token_ids["eos_token_id"]
        prompt_ids = _to_token_list(prompt_ids_from_payload(prompt))
        if not prompt_ids or prompt_ids[0] != bos or prompt_ids[-1] != eos:
            raise ValueError("AlphaGRPO requires BAGEL-framed prompt_token_ids; run bagel_dvreward.py first.")
        system_ids = [bos, *self.tokenizer.encode(GEN_THINK_SYSTEM_PROMPT, add_special_tokens=False), eos]
        prefix_ids = system_ids + prompt_ids
        if len(prefix_ids) + max_tokens + 2 > width:
            raise ValueError("System, prompt and thinking budget exceed pipeline.max_sequence_length.")
        if sampling.past_key_values is not None:
            raise ValueError("AlphaGRPO owns its thinking KV context and cannot accept an injected cache.")

        layers = self.bagel.config.llm_config.num_hidden_layers
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(sampling.seed)) if sampling.seed is not None else generator.seed()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type != "cpu"):
            system_cache = self._prefill(NaiveCache(layers), system_ids)
            cfg_image_cache = self._prefill(deepcopy(system_cache), prompt_ids, len(system_ids))
            image_embeds, image_lengths = None, [0, 0]
            if images:
                system_cache, image_embeds, image_lengths = self._prefill_image(
                    system_cache, images[0], len(system_ids)
                )
                prefix_ids = system_ids + [0] * sum(image_lengths) + prompt_ids
                if len(prefix_ids) + max_tokens + 2 > width:
                    raise ValueError(
                        f"Image, prompt and thinking budget require {len(prefix_ids) + max_tokens + 2} tokens; "
                        f"pipeline.max_sequence_length={width}."
                    )
            prefix_rope = len(system_ids) + (2 if images else 0) + len(prompt_ids)
            prefix_cache = self._prefill(deepcopy(system_cache), prompt_ids, prefix_rope - len(prompt_ids))
            tokens, old_log_probs = self._sample_thinking(
                deepcopy(prefix_cache), prefix_rope, max_tokens, temperature, generator
            )
            thinking_ids = [bos, *tokens]
            if thinking_ids[-1] != eos:
                thinking_ids.append(eos)
            image_cache = self._prefill(deepcopy(prefix_cache), thinking_ids, prefix_rope)

        condition_ids = prefix_ids + thinking_ids
        replay = {}
        for name, ids in (
            ("image_condition", condition_ids),
            ("cfg_text", system_ids + [0] * sum(image_lengths)),
            ("cfg_image", system_ids + prompt_ids),
        ):
            replay[f"{name}_ids"] = torch.zeros(1, width, dtype=torch.long)
            replay[f"{name}_ids"][0, : len(ids)] = torch.tensor(ids)
            replay[f"{name}_mask"] = torch.arange(width).unsqueeze(0) < len(ids)
        replay["thinking_input_ids"] = replay["image_condition_ids"].clone()
        replay["thinking_attention_mask"] = replay["image_condition_mask"].clone()
        replay["thinking_labels"] = torch.zeros(1, width, dtype=torch.long)
        replay["thinking_mask"] = torch.zeros(1, width, dtype=torch.bool)
        replay["old_text_log_probs"] = torch.zeros(1, width)
        start, end = len(prefix_ids), len(prefix_ids) + len(tokens)
        replay["thinking_labels"][0, start:end] = torch.tensor(tokens)
        replay["thinking_mask"][0, start:end] = True
        replay["old_text_log_probs"][0, start:end] = torch.tensor(old_log_probs)
        replay["thinking_text"] = self.tokenizer.decode(
            tokens[:-1] if tokens[-1] == eos else tokens, skip_special_tokens=False
        )

        if image_embeds is not None:
            replay["condition_image_embeds"] = image_embeds.detach().cpu().unsqueeze(0)
            replay["condition_image_layout"] = torch.tensor([[len(system_ids), *image_lengths]])
        overrides = {
            "past_key_values": image_cache,
            "kv_metadata": {"ropes": [prefix_rope + len(thinking_ids)]},
            "cfg_text_past_key_values": system_cache,
            "cfg_text_kv_metadata": {"ropes": [len(system_ids) + (2 if images else 0)]},
            "cfg_img_past_key_values": cfg_image_cache,
            "cfg_img_kv_metadata": {"ropes": [len(system_ids) + len(prompt_ids)]},
        }
        original = {key: getattr(sampling, key) for key in overrides}
        try:
            for key, value in overrides.items():
                setattr(sampling, key, value)
            output = super().forward(req)
        finally:
            for key, value in original.items():
                setattr(sampling, key, value)
        return with_rollout_data(output, rl=replay)
