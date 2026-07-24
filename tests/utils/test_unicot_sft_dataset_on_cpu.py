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

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("VERL_OMNI_SKIP_AUTO_IMPORTS", "1")

import torch

from verl_omni.utils.dataset.unicot_sft_dataset import (
    IGNORE_INDEX,
    UniCOTSFTDataset,
    build_unicot_events,
    unicot_sft_collate_fn,
)


def test_build_unicot_events_maps_generated_images():
    events = build_unicot_events(
        ["problem.png", "step1.png", "step2.png"],
        ["Solve step by step."],
        [
            "think_start draw first think_end <image_start>",
            "<image_end> think_start draw second think_end <image_start>",
            "<image_end> final answer",
        ],
    )

    assert [event.type for event in events] == [
        "context_image",
        "text",
        "text",
        "generated_image",
        "text",
        "generated_image",
        "text",
    ]
    assert [event.image_path for event in events if event.type == "generated_image"] == ["step1.png", "step2.png"]
    assert events[-1].text == "final answer"


def test_build_unicot_events_supports_t2i_without_context_image():
    events = build_unicot_events(
        ["target.png"],
        ["Draw a triangle diagram."],
        ["think_start prepare image think_end <image_start>"],
        num_context_images=0,
    )

    assert [event.type for event in events] == ["text", "text", "generated_image"]
    assert events[-1].image_path == "target.png"


def test_unicot_collate_pads_text_and_keeps_metadata():
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([IGNORE_INDEX, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "unicot_sft_events": [{"type": "text"}],
            "context_image_paths": ["problem.png"],
            "generated_image_paths": ["step1.png"],
            "task_type": "editing",
            "data_source": "unicot",
            "extra_info": {"index": 0},
            "image_hidden_states": torch.zeros(1, 2, 4),
            "image_velocity_target": torch.ones(1, 2, 4),
            "image_loss_mask": torch.ones(1, 2),
            "timesteps": torch.zeros(1),
            "latent_pos_ids": torch.zeros(1, 2, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([4]),
            "labels": torch.tensor([4]),
            "attention_mask": torch.tensor([1]),
            "unicot_sft_events": [{"type": "text"}],
            "context_image_paths": ["problem2.png"],
            "generated_image_paths": [],
            "task_type": "vlm_sft",
            "data_source": "unicot",
            "extra_info": {"index": 1},
            "image_hidden_states": torch.zeros(1, 2, 4),
            "image_velocity_target": torch.ones(1, 2, 4),
            "image_loss_mask": torch.ones(1, 2),
            "timesteps": torch.zeros(1),
            "latent_pos_ids": torch.zeros(1, 2, dtype=torch.long),
        },
    ]

    batch = unicot_sft_collate_fn(features)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"][1, 1:].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][1].tolist() == [1, 0, 0]
    assert batch["generated_image_paths"][0] == ["step1.png"]
    assert batch["task_type"] == ["editing", "vlm_sft"]
    assert batch["image_hidden_states"].shape == (2, 1, 2, 4)
    assert batch["image_velocity_target"].shape == (2, 1, 2, 4)


def test_unicot_dataset_uses_hf_load_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name, split):
        calls.append((name, split))
        return [
            {
                "image_list": ["problem.png", "step1.png"],
                "instruction_list": ["Solve."],
                "output_text_list": ["reason <image_start>"],
            }
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    dataset = UniCOTSFTDataset(
        "Fr0zencr4nE/UniCoT-Self-Reflection-6K",
        tokenizer=None,
        config={"custom_cls": {"train_split": "train"}},
        is_train=True,
    )

    assert calls == [("Fr0zencr4nE/UniCoT-Self-Reflection-6K", "train")]
    assert len(dataset) == 1
