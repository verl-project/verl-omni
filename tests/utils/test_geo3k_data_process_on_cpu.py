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
from io import BytesIO
from pathlib import Path

import datasets
import pytest
from PIL import Image


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
    assert row["ability"] == "math"
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


def test_preserve_image_bytes_disables_decode_without_reencoding():
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    source = datasets.Dataset.from_dict(
        {
            "images": [[{"bytes": image_bytes, "path": None}]],
            "problem": ["<image>Find x."],
            "answer": ["3"],
        },
        features=datasets.Features(
            {
                "images": datasets.Sequence(datasets.Image()),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        ),
    )

    raw_source = geo3k.preserve_image_bytes(source)

    assert raw_source.features["images"].feature.decode is False
    assert raw_source[0]["images"][0]["bytes"] == image_bytes


@pytest.mark.parametrize(
    ("problem", "images"),
    [
        ("Find x.", [{"bytes": b"png", "path": None}]),
        ("<image><image>Find x.", [{"bytes": b"png", "path": None}]),
        ("<image>Find x.", []),
    ],
)
def test_build_rl_row_rejects_image_placeholder_mismatch(problem, images):
    with pytest.raises(ValueError, match="image"):
        geo3k.build_rl_row(
            {"problem": problem, "answer": "3", "images": images},
            split="train",
            index=0,
        )


def test_converted_row_routes_to_real_geo3k_reward():
    """Validate actual reward dispatch and its accuracy/format components."""
    from verl.utils.reward_score import default_compute_score

    row = geo3k.build_rl_row(
        {"problem": "<image>Find x.", "answer": "4", "images": [{"bytes": b"png", "path": None}]},
        split="train",
        index=0,
    )

    def score(response):
        return default_compute_score(row["data_source"], response, row["reward_model"]["ground_truth"])

    assert score(r"<think>Compute the answer.</think>\boxed{4}") == pytest.approx(1.0)
    assert score(r"<think>Wrong answer.</think>\boxed{5}") == pytest.approx(0.1)
    assert score("4") == pytest.approx(0.0)
