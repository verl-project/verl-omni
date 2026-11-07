#!/usr/bin/env python
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
"""Build deterministic dummy AVQA media and parquet for the MiniCPM-o GSPO smoke.

Rows reuse the real AVQA converter (``examples/gspo_trainer/data_process/avqa.py``
``build_rl_row``) so the parquet schema matches production training data; only the
media is synthetic. Images are 448x448 (one full-resolution slice, no tiling) and
audio clips are 16 kHz mono tones with deliberate hop-tail lengths so the dataset's
hop-multiple padding is exercised.
"""

import argparse
import math
import runpy
import struct
import wave
from pathlib import Path

import pandas as pd

# Megatron-LM also ships an `examples` package; load the converter by path.
build_rl_row = runpy.run_path(str(Path(__file__).parents[2] / "examples/gspo_trainer/data_process/avqa.py"))[
    "build_rl_row"
]

_IMAGE_SIZE = 448
_SAMPLE_RATE = 16000


def _write_image(path: Path, index: int) -> None:
    from PIL import Image

    image = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE))
    pixels = image.load()
    base = (10 * index + 20) % 255
    for y in range(_IMAGE_SIZE):
        for x in range(_IMAGE_SIZE):
            pixels[x, y] = ((x + base) % 256, (y + base) % 256, (x * y // 256 + base) % 256)
    image.save(path)


def _write_audio(path: Path, index: int) -> None:
    # A tail that is not a hop multiple: the RL dataset must zero-pad it.
    sample_count = 8000 + index * 161
    frequency = 220 + 40 * index
    pcm = struct.pack(
        f"<{sample_count}h",
        *(int(8000 * math.sin(2 * math.pi * frequency * i / _SAMPLE_RATE)) for i in range(sample_count)),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(pcm)


def build(output_dir: Path) -> None:
    # Deterministic output: re-running into a partially written directory must
    # self-heal, not wedge the smoke behind a FileExistsError.
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", 8), ("validation", 2)):
        rows = []
        for index in range(count):
            stem = f"{split}_{index}"
            image_name = f"{stem}.png"
            audio_name = f"{stem}.wav"
            _write_image(output_dir / image_name, index)
            _write_audio(output_dir / audio_name, index)
            row, drop_reason = build_rl_row(
                {
                    "problem": "Which sound is present in the clip?",
                    "data_type": "image_audio",
                    "options": ["A. A tone", "B. Silence"],
                    "solution": "<answer>A</answer>",
                    "path": {"image": image_name, "audio": audio_name},
                    "problem_id": stem,
                    "problem_type": "smoke",
                },
                output_dir,
                split,
                index,
            )
            if row is None:
                raise ValueError(f"Dummy AVQA row {stem} was rejected by the converter: {drop_reason}")
            rows.append(row)
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet", index=False, use_dictionary=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args().output_dir)
