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
"""MiniCPM simplex converter and the real RL dataset media contract."""

import importlib.util
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset


@pytest.fixture
def converter():
    path = Path(__file__).resolve().parents[2] / "examples/opd_trainer/minicpm_o/prepare_data.py"
    spec = importlib.util.spec_from_file_location("minicpm_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_file


def test_converter_and_rl_dataset_preserve_image_audio_slots(tmp_path, converter):
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(320))
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"prompt": "What?", "images": ["image.png"], "audios": ["audio.wav"]}))
    destination = tmp_path / "train.parquet"
    converter(source, destination)
    frame = pd.read_parquet(destination)
    assert frame.iloc[0].images[0]["bytes"] == (tmp_path / "image.png").read_bytes()
    assert frame.iloc[0].audios[0] == str(tmp_path / "audio.wav")
    dataset = RLHFDataset(
        data_files=[str(destination)],
        tokenizer=MagicMock(),
        processor=MagicMock(),
        config=OmegaConf.create({"filter_overlong_prompts": False}),
    )
    content = dataset[0]["raw_prompt"][0]["content"]
    assert [item["type"] for item in content if item["type"] != "text"] == ["image", "audio"]
    assert content[-1]["text"].endswith("What?")


@pytest.mark.parametrize(
    "row,match",
    [
        ({"prompt": "text", "videos": ["video.mp4"]}, "Video"),
        ({"prompt": "text", "audios": ["a.wav", "b.wav"]}, "one audio"),
        ({"prompt": [{"role": "user", "content": "<image>"}]}, "exactly once"),
    ],
)
def test_converter_rejects_unsupported_or_unconsumed_media(tmp_path, converter, row, match):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=match):
        converter(source, tmp_path / "train.parquet")


def test_converter_accepts_text_only_and_rejects_empty_input(tmp_path, converter):
    source = tmp_path / "train.jsonl"
    source.write_text('{"prompt":"Hello"}\n')
    destination = tmp_path / "train.parquet"
    converter(source, destination)
    assert pd.read_parquet(destination).iloc[0].prompt[0]["content"] == "Hello"
    source.write_text("")
    with pytest.raises(ValueError, match="No prompts"):
        converter(source, destination)
