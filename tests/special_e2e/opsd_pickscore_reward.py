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
"""PickScore endpoint reward for the DiffusionOPSD e2e test."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from verl_omni.utils.reward_score import pickscore_reward as _pickscore
from verl_omni.utils.reward_score.image_preference_utils import clip_feature_tensor

_PICKSCORE_PATH = os.environ["PICKSCORE_PATH"]
_FALLBACK_PROMPT = "a red circle on a white background"


class _LocalPickScoreInferencer:
    """Tiny-checkpoint loader used only by the DiffusionOPSD e2e smoke test."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(_PICKSCORE_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(_PICKSCORE_PATH)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = CLIPModel.from_pretrained(_PICKSCORE_PATH).eval().to(device=device, dtype=dtype)

    @torch.no_grad()
    def score(self, prompts, images):
        prompts = [p if isinstance(p, str) and p.strip() else _FALLBACK_PROMPT for p in prompts]
        unique_prompts = list(dict.fromkeys(prompts))
        prompt_to_index = {prompt: index for index, prompt in enumerate(unique_prompts)}
        prompt_indices = [prompt_to_index[prompt] for prompt in prompts]

        image_inputs = self.image_processor(images=images, return_tensors="pt")
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
        text_inputs = self.tokenizer(unique_prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_inputs = {
            key: value.to(self.device) for key, value in text_inputs.items() if key in ("input_ids", "attention_mask")
        }

        image_embeds = clip_feature_tensor(self.model.get_image_features(**image_inputs))
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = clip_feature_tensor(self.model.get_text_features(**text_inputs))
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds[prompt_indices]

        scores = self.model.logit_scale.exp() * (text_embeds @ image_embeds.T)
        return scores.diag() / 26


assert hasattr(_pickscore, "_PickScoreInferencer"), "stale patch point"
_pickscore._PickScoreInferencer = _LocalPickScoreInferencer
compute_score_pickscore = _pickscore.compute_score_pickscore
