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

import torch
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb, _to_token_list
from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_request import condition_images_from_payload, prompt_ids_from_payload

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, "
    "i.e. <think> planning process here </think> image here"
)


@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="alphagrpo")
class BagelAlphaGRPOPipeline(BagelPipelineWithLogProb):
    """Single-turn reasoning-to-image rollout; see the BAGEL README Gotchas."""

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

    def _sample_thinking(self, cache, prefix_length, max_tokens, temperature, top_p, generator):
        """Return sampled labels and unfiltered temperature-scaled old log-probs [T]."""
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
            sorted_logits, indices = logits.sort(descending=True)
            probabilities = sorted_logits.softmax(-1)
            remove = probabilities.cumsum(-1) - probabilities >= top_p
            filtered = sorted_logits.masked_fill(remove, -torch.inf).softmax(-1)
            selected = torch.multinomial(filtered, 1, generator=generator)
            current = indices.gather(-1, selected).squeeze(-1)
            tokens.append(int(current.item()))
            log_probs.append(logits.log_softmax(-1).gather(-1, current.unsqueeze(-1)).item())
            if tokens[-1] == self.new_token_ids["eos_token_id"]:
                break
        return tokens, log_probs

    @torch.no_grad()
    def forward(self, req):
        """Record fixed-width text replay tensors and reuse the existing image SDE rollout."""
        if req.is_dummy_run():
            return super().forward(req)
        if len(req.prompts) != 1 or not isinstance(req.prompts[0], dict):
            raise ValueError("AlphaGRPO requires one native token prompt per request.")
        prompt = req.prompts[0]
        if condition_images_from_payload(prompt) or prompt.get("multi_modal_data"):
            raise ValueError("This AlphaGRPO adapter supports text-to-image, not image editing.")
        sampling = req.sampling_params
        extra = sampling.extra_args
        negative = self._decode_token_prompt(prompt.get("negative_prompt_ids")) or extra.get("negative_prompt", "")
        if negative.strip():
            raise ValueError(
                "AlphaGRPO uses the planning system prompt as its text CFG context, without a negative prompt."
            )
        max_tokens = int(extra.get("max_think_tokens", 512))
        temperature = float(extra.get("think_temperature", 1.0))
        top_p = float(extra.get("think_top_p", 0.8))
        width = int(sampling.max_sequence_length or 2048)
        if max_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("Thinking requires max_think_tokens >= 1, temperature > 0 and 0 < top_p <= 1.")
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
            prefix_cache = self._prefill(deepcopy(system_cache), prompt_ids, len(system_ids))
            tokens, old_log_probs = self._sample_thinking(
                deepcopy(prefix_cache), len(prefix_ids), max_tokens, temperature, top_p, generator
            )
            thinking_ids = [bos, *tokens]
            if thinking_ids[-1] != eos:
                thinking_ids.append(eos)
            image_cache = self._prefill(deepcopy(prefix_cache), thinking_ids, len(prefix_ids))

        condition_ids = prefix_ids + thinking_ids
        replay = {}
        for name, ids in (("image_condition", condition_ids), ("cfg_text", system_ids), ("cfg_image", prefix_ids)):
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

        overrides = {
            "past_key_values": image_cache,
            "kv_metadata": {"ropes": [len(condition_ids)]},
            "cfg_text_past_key_values": system_cache,
            "cfg_text_kv_metadata": {"ropes": [len(system_ids)]},
            "cfg_img_past_key_values": prefix_cache,
            "cfg_img_kv_metadata": {"ropes": [len(prefix_ids)]},
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
