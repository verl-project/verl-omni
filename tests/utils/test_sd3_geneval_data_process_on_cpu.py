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
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[2] / "examples/flowgrpo_trainer/data_process/sd3_geneval.py"
    spec = importlib.util.spec_from_file_location("sd3_geneval_data_process", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sd3_geneval = _load_module()


def test_build_row_carries_full_metadata_as_ground_truth():
    metadata = {
        "tag": "color_attr",
        "include": [{"class": "bus", "count": 1, "color": "yellow"}],
        "exclude": [{"class": "car", "count": 1}],
        "prompt": "a photo of a yellow bus",
    }
    row = sd3_geneval.build_row(metadata, split="train", index=3)

    assert row["data_source"] == "flow_grpo/geneval"
    assert row["prompt"] == [{"role": "user", "content": "a photo of a yellow bus"}]
    assert row["ability"] == "geneval"
    assert row["reward_model"]["style"] == "model"
    assert json.loads(row["reward_model"]["ground_truth"]) == metadata
    assert row["extra_info"] == {
        "split": "train",
        "index": 3,
        "raw_prompt": "a photo of a yellow bus",
        "tag": "color_attr",
    }
