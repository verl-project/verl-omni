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

"""Locks the Pick-a-Pic prompt split and Boogu prompt-template contracts."""

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[2] / "examples/flowgrpo_trainer/data_process/boogu_image_pickapic.py"
    spec = importlib.util.spec_from_file_location("boogu_image_pickapic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load_module()


def test_split_is_deterministic_deduplicated_and_disjoint() -> None:
    captions = ["  red   cube ", "red cube", "blue sphere", "green tree", "", None, "x"]

    train, validation = converter.split_captions(captions, seed=7, validation_size=1, train_size=2)
    train_again, validation_again = converter.split_captions(captions, seed=7, validation_size=1, train_size=2)

    assert (train, validation) == (train_again, validation_again)
    assert len(train) == 2
    assert len(validation) == 1
    assert set(train).isdisjoint(validation)
    assert set(train + validation) == {"red cube", "blue sphere", "green tree"}


def test_build_split_uses_boogu_t2i_and_empty_negative_templates() -> None:
    caption = "a red cube on a table"
    row = converter.build_split([caption], "train")[0]

    assert row["prompt"] == [
        {"role": "system", "content": converter.BOOGU_SYSTEM_PROMPT_T2I},
        {"role": "user", "content": caption},
    ]
    assert row["negative_prompt"] == [
        {"role": "system", "content": converter.BOOGU_SYSTEM_PROMPT_DROP},
        {"role": "user", "content": ""},
    ]
    assert row["reward_model"]["ground_truth"] == caption
    assert row["data_source"] == "pickapic_v1"
    assert row["extra_info"]["source"] == converter.DEFAULT_DATASET


def test_build_split_records_the_selected_source_dataset() -> None:
    row = converter.build_split(["a red cube"], "train", source="/data/my_pickapic_export")[0]

    assert row["extra_info"]["source"] == "/data/my_pickapic_export"


def test_split_requires_enough_unique_prompts() -> None:
    with pytest.raises(ValueError, match="unique prompts"):
        converter.split_captions(["one prompt"], seed=0, validation_size=1, train_size=1)
