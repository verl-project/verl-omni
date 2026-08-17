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

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/geo3k.py"
    spec = importlib.util.spec_from_file_location("geo3k_data_process", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


geo3k = _load_module()


def test_build_rl_row_preserves_image_and_geo3k_reward_contract():
    image = {"bytes": b"png", "path": None}
    row = geo3k.build_rl_row(
        {"problem": "<image>Find x.", "answer": "3", "images": [image]},
        split="train",
        index=7,
    )

    assert row["data_source"] == "hiyouga/geometry3k"
    assert row["ability"] == "math_vl"
    assert row["images"] == [image]
    assert row["prompt"][0] == {"role": "system", "content": geo3k.SYSTEM_PROMPT}
    assert row["prompt"][1]["content"].startswith("<image>Find x.")
    assert "<think>" in row["prompt"][1]["content"]
    assert "\\boxed{}" in row["prompt"][1]["content"]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "3"}
    assert row["extra_info"] == {
        "split": "train",
        "index": 7,
        "answer": "3",
        "question": "<image>Find x.",
    }
