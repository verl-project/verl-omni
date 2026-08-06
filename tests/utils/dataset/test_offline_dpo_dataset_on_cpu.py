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
"""CPU tests for offline diffusion DPO dataset utilities."""

import io
import json

import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.utils.dataset.offline_dpo_dataset import (
    OFFLINE_DPO_PAIR_MARKER,
    OfflineDPODataset,
    _resolve_raw_prompts,
    _tensor_from_column,
    expand_offline_dpo_features,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, prompt, **kwargs):
        del kwargs
        return " ".join(message["content"] for message in prompt if message.get("role") == "user")

    def __call__(self, text, **kwargs):
        del text, kwargs
        return {"input_ids": torch.tensor([[11, 12]], dtype=torch.long)}


def _config():
    return OmegaConf.create(
        {
            "max_prompt_length": 4,
            "prompt_key": "prompt",
            "negative_prompt_key": "negative_prompt",
            "img_win_key": "img_win",
            "img_lose_key": "img_lose",
        }
    )


def _row(**overrides):
    row = {
        "uid": "pair-0",
        "prompt": [{"role": "user", "content": "a cat"}],
        "negative_prompt": [{"role": "user", "content": "blurry"}],
        "img_win": "win.png",
        "img_lose": "lose.png",
        "img_win_latents": [[1.0, 2.0]],
        "img_lose_latents": [[0.5, 0.25]],
        "prompt_embeds": [[0.1, 0.2]],
        "prompt_embeds_mask": [1, 1],
        "pooled_prompt_embeds": [0.3, 0.4],
        "win_score": 1.0,
        "lose_score": 0.0,
        "extra_info": {"raw_prompt": "raw cat", "raw_negative_prompt": "raw blurry"},
    }
    row.update(overrides)
    return row


def test_resolve_raw_prompts_prefers_extra_info():
    prompt = [{"role": "user", "content": "fallback prompt"}]
    negative_prompt = [{"role": "user", "content": "fallback negative"}]

    raw_prompt, raw_negative = _resolve_raw_prompts(
        prompt,
        negative_prompt,
        {"raw_prompt": " raw prompt ", "raw_negative_prompt": " raw negative "},
    )

    assert raw_prompt == "raw prompt"
    assert raw_negative == "raw negative"


def test_tensor_from_column_supports_serialized_tensor_bytes():
    buffer = io.BytesIO()
    torch.save(torch.tensor([1, 2], dtype=torch.int64), buffer)

    tensor = _tensor_from_column(buffer.getvalue(), dtype=torch.float32)

    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))


def test_dataset_item_decodes_pair_row(tmp_path):
    data_file = tmp_path / "pairs.json"
    data_file.write_text(json.dumps([_row()]))

    dataset = OfflineDPODataset(str(data_file), tokenizer=DummyTokenizer(), config=_config())
    item = dataset[0]

    assert item[OFFLINE_DPO_PAIR_MARKER] is True
    assert item["uid"] == "pair-0"
    assert item["prompts"].tolist() == [0, 0, 11, 12]
    assert item["raw_prompt"] == "raw cat"
    assert item["raw_negative_prompt"] == "raw blurry"
    assert item["img_win"] == str(tmp_path / "win.png")
    assert item["img_lose"] == str(tmp_path / "lose.png")
    torch.testing.assert_close(item["img_win_latents"], torch.tensor([[1.0, 2.0]]))
    assert item["prompt_embeds_mask"].dtype == torch.int32


def test_dataset_rejects_inverted_pair_scores(tmp_path):
    data_file = tmp_path / "pairs.json"
    data_file.write_text(json.dumps([_row(win_score=0.0, lose_score=1.0)]))
    dataset = OfflineDPODataset(str(data_file), tokenizer=DummyTokenizer(), config=_config())

    with pytest.raises(ValueError, match="win_score < lose_score"):
        dataset[0]


def test_expand_offline_dpo_features_preserves_pair_order():
    feature = {
        OFFLINE_DPO_PAIR_MARKER: True,
        "prompts": torch.tensor([1, 2]),
        "uid": "pair-0",
        "raw_prompt": "prompt",
        "raw_negative_prompt": "",
        "data_source": "offline_dpo",
        "reward_model": {"ground_truth": "prompt"},
        "extra_info": {"index": 0},
        "prompt_embeds": torch.ones(1, 2),
        "prompt_embeds_mask": torch.ones(2, dtype=torch.int32),
        "pooled_prompt_embeds": torch.ones(2),
        "img_win": "win.png",
        "img_lose": "lose.png",
        "img_win_latents": torch.ones(1, 2),
        "img_lose_latents": torch.zeros(1, 2),
        "win_score": 1.0,
        "lose_score": 0.25,
    }

    expanded = expand_offline_dpo_features([feature])

    assert len(expanded) == 2
    assert expanded[0]["is_chosen"] is True
    assert expanded[0]["image_path"] == "win.png"
    torch.testing.assert_close(expanded[0]["sample_level_scores"], torch.tensor([1.0]))
    assert expanded[1]["is_chosen"] is False
    assert expanded[1]["image_path"] == "lose.png"
    torch.testing.assert_close(expanded[1]["sample_level_scores"], torch.tensor([0.25]))
