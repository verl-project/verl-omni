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

"""CPU tests for the GSM8K converter's pure helpers (ground-truth extraction, row shape)."""

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "examples/gspo_trainer/data_process/gsm8k.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("gsm8k_data_process", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the gsm8k data_process script.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extracts_numeric_ground_truth_after_the_marker():
    converter = _load_converter()
    raw = "Natalia sold 48/2 = 24 clips in May.\n#### 72"
    assert converter.extract_ground_truth(raw) == "72"


def test_strips_thousands_commas_and_currency_marks():
    converter = _load_converter()
    assert converter.extract_ground_truth("...\n#### 1,000") == "1000"
    assert converter.extract_ground_truth("...\n#### $12.50") == "12.50"


def test_rejects_answers_without_the_marker():
    converter = _load_converter()
    assert converter.extract_ground_truth("no marker here") is None
    assert converter.extract_ground_truth("") is None


def test_row_shape_follows_the_verl_rl_schema():
    converter = _load_converter()
    row = converter.make_map_fn("train")({"question": "What is 2 + 3?", "answer": "2 + 3 = 5\n#### 5"}, 7)

    assert row["data_source"] == "openai/gsm8k"
    assert row["ability"] == "math"
    assert len(row["prompt"]) == 1 and row["prompt"][0]["role"] == "user"
    assert "What is 2 + 3?" in row["prompt"][0]["content"]
    assert "\\boxed{}" in row["prompt"][0]["content"]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "5"}
    assert row["extra_info"]["split"] == "train" and row["extra_info"]["index"] == 7
    assert "images" not in row and "audios" not in row
