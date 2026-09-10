#!/usr/bin/env python3
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
"""Create deterministic Qwen3-TTS GRPO smoke data and speaker conditioning."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _speaker_dimension(model_config_path: Path) -> int:
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    talker_dimension = model_config["talker_config"]["hidden_size"]
    speaker_dimension = model_config["speaker_encoder_config"]["enc_dim"]
    if talker_dimension != speaker_dimension:
        raise ValueError(
            f"speaker encoder output must match the talker hidden size: {speaker_dimension} != {talker_dimension}"
        )
    if not isinstance(speaker_dimension, int) or speaker_dimension <= 0:
        raise ValueError(f"speaker dimension must be a positive integer, got {speaker_dimension!r}")
    return speaker_dimension


def _row(text: str, sample_id: str, split: str) -> dict:
    extra_info = {"id": sample_id, "split": split}
    return {
        "data_source": "tts",
        "prompt": [{"role": "user", "content": text}],
        "reward_model": {"style": "model", "ground_truth": text},
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_texts = (
        "Please read this sentence at a calm and steady pace.",
        "A short speech sample checks the complete training path.",
        "Clear pronunciation makes this audio easy to inspect.",
        "The weather is pleasant and the morning train is on time.",
        "Four simple prompts are enough for one smoke-test batch.",
        "This second batch verifies another optimizer update.",
        "Generated speech is decoded before the reward is computed.",
        "The final checkpoint confirms that training completed.",
    )
    validation_texts = (
        "This is fixed validation sample one.",
        "This is fixed validation sample two.",
        "This is fixed validation sample three.",
        "This is fixed validation sample four.",
    )
    train_rows = [_row(text, f"train-{index}", "train") for index, text in enumerate(train_texts)]
    validation_rows = [_row(text, f"validation-{index}", "validation") for index, text in enumerate(validation_texts)]
    pd.DataFrame(train_rows).to_parquet(args.output_dir / "train.parquet", index=False)
    pd.DataFrame(validation_rows).to_parquet(args.output_dir / "validation.parquet", index=False)

    speaker_dimension = _speaker_dimension(args.model_config)
    speaker = [1.0 / math.sqrt(speaker_dimension)] * speaker_dimension
    (args.output_dir / "speaker.json").write_text(json.dumps(speaker), encoding="utf-8")


if __name__ == "__main__":
    main()
