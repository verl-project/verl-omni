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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import (
    QwenImageConditionProvider,
    QwenImageDMD2,
)


class ToyPromptTokenizer:
    def __call__(self, texts, **kwargs):
        width = max(len(text) for text in texts)
        ids = torch.tensor([[len(text)] * len(text) + [0] * (width - len(text)) for text in texts])
        return SimpleNamespace(input_ids=ids, attention_mask=ids.ne(0).long())


class ToyTextEncoder(torch.nn.Module):
    dtype = torch.float32

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=(input_ids.float().unsqueeze(-1),))


class ToyConditionPipeline:
    prompt_template_encode = "x" * 34 + "{}"
    prompt_template_encode_start_idx = 34
    device = torch.device("cpu")

    def __init__(self):
        from diffusers import QwenImagePipeline

        self.tokenizer = ToyPromptTokenizer()
        self.text_encoder = ToyTextEncoder()
        self._extract_masked_hidden = QwenImagePipeline._extract_masked_hidden.__get__(self)


class TestQwenDMD2:
    def test_registry_does_not_claim_original_dmd_or_edit_support(self):
        assert DiffusionModelBase.get_class_by_name("QwenImagePipeline", "dmd2") is QwenImageDMD2
        with pytest.raises(NotImplementedError):
            DiffusionModelBase.get_class_by_name("QwenImagePipeline", "dmd")
        with pytest.raises(NotImplementedError):
            DiffusionModelBase.get_class_by_name("QwenImageEditPlusPipeline", "dmd2")

    def test_short_prompts_keep_tokens_after_real_prefix_removal(self):
        provider = QwenImageConditionProvider("unused", 64, " ")
        provider.pipeline = ToyConditionPipeline()
        for size in (1, 3, 2):
            batch = TensorDict({"dummy_tensor": torch.zeros(size, 1)}, batch_size=[size])
            tu.assign_non_tensor_stack(batch, "raw_prompt", [[{"role": "user", "content": "cat"}]] * size)
            positive, negative = provider.encode(
                batch, device=torch.device("cpu"), dtype=torch.float32, require_negative=True
            )
            assert positive["prompt_embeds"].shape == (size, 3, 1)
            assert negative["prompt_embeds"].shape == (size, 1, 1)
            assert positive["prompt_embeds"][0, 0, 0] == 37
            assert not positive["prompt_embeds"].requires_grad

    @pytest.mark.parametrize(
        "row", [[], [{"role": "system", "content": "custom"}], [{"role": "assistant", "content": "cat"}]]
    )
    def test_invalid_chat_is_not_generic_chat_formatted(self, row):
        provider = QwenImageConditionProvider("unused", 64, " ")
        with pytest.raises(ValueError, match="single user"):
            provider.tokenize_rows(Mock(), [row], torch.device("cpu"))

    def test_precomputed_inputs_do_not_load_encoder_for_fake_stage(self):
        provider = QwenImageConditionProvider("unused", 2, " ")
        batch = TensorDict({"prompt_embeds": torch.ones(2, 3, 4, requires_grad=True)}, batch_size=[2])
        positive, negative = provider.encode(
            batch, device=torch.device("cpu"), dtype=torch.float32, require_negative=False
        )
        assert provider.pipeline is None and negative is None
        assert positive["prompt_embeds"].shape == (2, 2, 4)
        assert positive["prompt_embeds_mask"].shape == (2, 2)
        assert not positive["prompt_embeds"].requires_grad
        with pytest.raises(ValueError, match="negative_prompt_embeds"):
            provider.encode(batch, device=torch.device("cpu"), dtype=torch.float32, require_negative=True)

    def test_geometry_uses_vae_config_and_packs_consistently(self, tmp_path):
        (tmp_path / "vae").mkdir()
        (tmp_path / "vae" / "config.json").write_text(json.dumps({"z_dim": 4, "temperal_downsample": [False, True]}))
        model = SimpleNamespace(config=SimpleNamespace(in_channels=16))
        config = SimpleNamespace(local_path=str(tmp_path), pipeline=SimpleNamespace(height=32, width=48))
        shape, geometry = QwenImageDMD2.latent_geometry(model, config, TensorDict({}, batch_size=[2]))
        assert shape == (2, 4, 1, 8, 12)
        assert geometry["vae_scale_factor"] == 4
        latents = QwenImageDMD2.pack_latents(torch.ones(shape))
        assert latents.shape == (2, 24, 16)
        batch = TensorDict({"height": torch.tensor([32, 64])}, batch_size=[2])
        with pytest.raises(ValueError, match="homogeneous"):
            QwenImageDMD2.latent_geometry(model, config, batch)
