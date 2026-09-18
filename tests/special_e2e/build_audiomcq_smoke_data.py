# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Build short PCM WAVs and AudioMCQ parquet offline for a structural smoke."""

import argparse
import math
import runpy
import struct
import wave
from pathlib import Path

import pandas as pd

# Megatron-LM also ships an `examples` package; load this script by path.
build_rl_row = runpy.run_path(str(Path(__file__).parents[2] / "examples/gspo_trainer/data_process/audiomcq.py"))[
    "build_rl_row"
]


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, count in (("train", 16), ("validation", 2)):
        rows = []
        for index in range(count):
            name = f"{split}_{index}.wav"
            # Deliberately include a tail that exercises audio hop-size padding.
            sample_count = 8000 + index * 161
            frequency = 220 + 40 * index
            pcm = struct.pack(
                f"<{sample_count}h",
                *(int(8000 * math.sin(2 * math.pi * frequency * i / 16000)) for i in range(sample_count)),
            )
            with wave.open(str(output_dir / name), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(pcm)
            row = build_rl_row(
                {
                    "audio_path": name,
                    "question": "Which sound is present?",
                    "choices": ["A tone", "Silence"],
                    "answer": "A tone",
                    "source_dataset": "synthetic_audio_smoke",
                    "id": name,
                },
                output_dir,
                index,
            )
            row["extra_info"]["split"] = split
            rows.append(row)
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet", index=False, use_dictionary=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args().output_dir)
