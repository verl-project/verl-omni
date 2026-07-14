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
"""CPU tests for offline MLLM DPO dataset helpers."""

import numpy as np
import pytest
import torch

from verl_omni.utils.dataset.offline_mllm_dpo_dataset import (
    _as_python,
    _build_preference_branch,
    _merge_chosen_rejected,
    _normalise_source_name,
    _prepare_qwen3_omni_processor,
)


def test_as_python_decodes_json_strings_and_numpy_values():
    assert _as_python('[{"role": "user"}]') == [{"role": "user"}]
    assert _as_python(b'{"chosen": "yes"}') == {"chosen": "yes"}
    assert _as_python(np.array([1, 2])) == [1, 2]


def test_build_preference_branch_collects_media_and_answer():
    sample = {
        "prompt": [
            {"role": "system", "content": "ignored"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "image": "image-0.png"},
                    {"type": "audio", "audio": "audio-0.wav"},
                ],
            },
        ],
        "source_name": "Omni-Preference-Image",
    }

    branch = _build_preference_branch(sample, "a small cat")

    assert branch["source_name"] == "Omni-Preference-Image"
    assert branch["images"] == ["image-0.png"]
    assert branch["audios"] == ["audio-0.wav"]
    assert branch["conversations"] == [
        ["user", ("text", "describe this"), ("image", None), ("audio", None)],
        ["assistant", ("text", "a small cat")],
    ]


def test_merge_chosen_rejected_concatenates_sequence_tensors():
    chosen = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "pixel_values": torch.ones(1, 3, 2, 2),
        "metadata": "chosen",
    }
    rejected = {
        "input_ids": torch.tensor([[3, 4]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "metadata": "rejected",
    }

    merged = _merge_chosen_rejected(chosen, rejected)

    torch.testing.assert_close(merged["input_ids"], torch.tensor([[1, 2, 3, 4]]))
    torch.testing.assert_close(merged["attention_mask"], torch.tensor([[1, 1, 1, 1]]))
    assert merged["pixel_values"].shape == (2, 3, 2, 2)
    assert merged["metadata"] == "chosen"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("image-preference", "Omni-Preference-Image"),
        ("video sample", "Omni-Preference-Video"),
        ("audio sample", "Omni-Preference-Audio"),
        ("custom-source", "custom-source"),
        (None, "fallback"),
    ],
)
def test_normalise_source_name(value, expected):
    assert _normalise_source_name(value, "fallback") == expected


def test_prepare_qwen3_omni_processor_normalises_audio_and_rope_result():
    class Processor:
        def __init__(self):
            self.last_kwargs = None

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return {"ok": True}

        def get_rope_index(self):
            return torch.tensor([1]), torch.tensor([2])

    processor = Processor()
    proxy = _prepare_qwen3_omni_processor(processor)

    assert proxy(audios=["audio"], images=[]) == {"ok": True}
    assert processor.last_kwargs == {"audio": ["audio"]}
    rope = proxy.get_rope_index()
    assert set(rope) == {"position_ids", "mrope_position_deltas"}
