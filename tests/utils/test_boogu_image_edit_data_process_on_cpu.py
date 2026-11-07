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

"""Locks the Boogu-Image edit converter's negative-prompt and reward contracts.

``boogu_image_edit_ocr.py`` is a script, not an importable module, so it is loaded by path the
same way ``test_ltx2_ti2va_data_process_on_cpu.py`` loads its converter.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


def _load_module():
    path = Path(__file__).parents[2] / "examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py"
    spec = importlib.util.spec_from_file_location("boogu_image_edit_ocr", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


edit_converter = _load_module()

INSTRUCTION = 'Change the text to "HELLO"'


def _write_split(input_dir: Path, image_name: str = "0001.png") -> None:
    (input_dir / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color="white").save(input_dir / "images" / image_name)
    record = {"image": image_name, "instruction": INSTRUCTION}
    (input_dir / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def _convert(tmp_path: Path, reward: str):
    _write_split(tmp_path)
    return edit_converter.convert_split(tmp_path, "train", max_samples=-1, image_size=64, reward=reward)


@pytest.mark.parametrize("reward", ["pickscore", "ocr"])
def test_negative_prompt_is_text_only(tmp_path: Path, reward: str) -> None:
    """The negative branch never receives the reference image, so it must carry no placeholder.

    Re-adding ``Picture 1: <image>`` would be tokenized but never expanded into image features
    (:meth:`BooguImageDiffusionNFTPipeline.encode_prompt` is called without ``condition_images``
    for the negative branch), which passes the media-count check while quietly shifting the
    guidance. A merge once restored the placeholder after it had been fixed; this pins it.
    """
    row = _convert(tmp_path, reward).iloc[0]

    negative = row["negative_prompt"]
    assert negative[0]["content"] == edit_converter.BOOGU_SYSTEM_PROMPT_TI2I
    assert negative[1]["content"] == ""
    assert "<image>" not in negative[1]["content"]

    # The positive prompt still carries exactly one placeholder for the row's one image.
    assert row["prompt"][1]["content"] == f"Picture 1: <image>{INSTRUCTION}"
    assert len(row["images"]) == 1


def test_reward_selects_ground_truth_and_data_source(tmp_path: Path) -> None:
    """``--reward`` sets both fields together so the val-core key and the scored text agree."""
    pickscore = _convert(tmp_path / "pickscore", "pickscore").iloc[0]
    assert pickscore["data_source"] == "pickscore"
    assert pickscore["reward_model"]["ground_truth"] == INSTRUCTION

    ocr = _convert(tmp_path / "ocr", "ocr").iloc[0]
    assert ocr["data_source"] == "ocr"
    assert ocr["reward_model"]["ground_truth"] == "HELLO"


def test_unknown_reward_raises() -> None:
    with pytest.raises(ValueError, match="unknown reward"):
        edit_converter.ground_truth_for("hpsv3", INSTRUCTION, "HELLO")


def test_extract_target_text_requires_a_quoted_span() -> None:
    assert edit_converter.extract_target_text(INSTRUCTION) == "HELLO"
    with pytest.raises(ValueError, match="double-quoted span"):
        edit_converter.extract_target_text("make it brighter")
