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
        **kwargs,
    ):
        """Replay masked text log-probs [B, L], or delegate the image velocity forward."""
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
