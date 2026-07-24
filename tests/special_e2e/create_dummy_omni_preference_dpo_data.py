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
"""Create a tiny Omni-Preference-style multimodal parquet dataset.

The generated data mirrors the offline MLLM DPO schema produced by
``omni_preference_dpo_multisource.py``:

    prompt, chosen, rejected, win_score, lose_score, data_source, ability,
    reward_model, extra_info

It writes three modality folders (image/video/audio) so e2e smoke tests exercise
mixed multimodal batches without downloading the real Omni-Preference dataset.
"""

from __future__ import annotations

import argparse
import os
import struct
import wave

import av
import pandas as pd
from PIL import Image

MODALITY_SPECS = {
    "image": {
        "data_source": "omni_preference/image",
        "ability": "image_qa",
        "source_media": "rlaif-v-dataset/dummy_images/dummy_image.png",
    },
    "video": {
        "data_source": "omni_preference/video",
        "ability": "video_qa",
        "source_media": "academic_source/dummy_clip.mp4",
    },
    "audio": {
        "data_source": "omni_preference/audio",
        "ability": "audio_qa",
        "source_media": "audio/dummy_audio.wav",
    },
}


def _write_png(path: str, color: tuple[int, int, int]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)
    return os.path.abspath(path)


def _write_video(
    path: str,
    colors: list[tuple[int, int, int]],
    *,
    fps: int = 2,
    size: tuple[int, int] = (64, 64),
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = size
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for color in colors:
        frame = av.VideoFrame.from_image(Image.new("RGB", size, color=color))
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return os.path.abspath(path)


def _write_wav(path: str, *, sample_rate: int = 16_000, duration_s: float = 1.0) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    num_frames = int(sample_rate * duration_s)
    frames = bytearray()
    for idx in range(num_frames):
        sample = int(12_000 * (1 if idx % 80 < 40 else -1))
        frames.extend(struct.pack("<h", sample))

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return os.path.abspath(path)


def _make_assets(root: str) -> dict[str, str]:
    media_dir = os.path.join(root, "media")
    image_path = _write_png(os.path.join(media_dir, "images", "dummy_image.png"), (240, 64, 64))
    audio_path = _write_wav(os.path.join(media_dir, "audio", "dummy_audio.wav"))

    video_path = _write_video(
        os.path.join(media_dir, "videos", "dummy_clip.mp4"),
        [(64, 128, 240), (64, 200, 120), (220, 180, 64)],
    )
    return {"image": image_path, "video": video_path, "audio": audio_path}


def _base_row(split: str, modality: str, index: int, question: str) -> dict:
    spec = MODALITY_SPECS[modality]
    source_media = spec["source_media"]
    return {
        "data_source": spec["data_source"],
        "chosen": {
            "role": "assistant",
            "content": "The preferred answer correctly describes the dummy content.",
        },
        "rejected": {
            "role": "assistant",
            "content": "The rejected answer gives an unrelated description.",
        },
        "win_score": 8.0,
        "lose_score": 4.0,
        "ability": spec["ability"],
        "reward_model": {"style": "preference"},
        "extra_info": {
            "split": split,
            "index": index,
            "sample_id": f"omni_pref_{modality}_{split}_{index}",
            "question": question,
            "modality": modality,
            "source_media": source_media,
            "source_media_name": os.path.basename(source_media),
            "better": "B",
        },
    }


def _image_row(split: str, index: int, image_path: str) -> dict:
    question = "What is shown in this dummy image?"
    row = _base_row(split, "image", index, question)
    row["prompt"] = [
        {
            "role": "user",
            "content": f"<image>{question}",
        },
    ]
    row["images"] = [image_path]
    row["extra_info"]["image_path"] = image_path
    return row


def _video_row(split: str, index: int, video_path: str) -> dict:
    question = "What changes across the dummy video frames?"
    row = _base_row(split, "video", index, question)
    row["prompt"] = [
        {
            "role": "user",
            "content": f"<video>{question}",
        },
    ]
    row["videos"] = [video_path]
    row["extra_info"]["video_path"] = video_path
    return row


def _audio_row(split: str, index: int, audio_path: str) -> dict:
    question = "What sound is described in this dummy audio clip?"
    row = _base_row(split, "audio", index, question)
    row["prompt"] = [
        {
            "role": "user",
            "content": f"<audio>{question}",
        },
    ]
    row["audios"] = [audio_path]
    row["extra_info"]["audio_path"] = audio_path
    return row


def _build_rows(split: str, size: int, assets: dict[str, str]) -> dict[str, list[dict]]:
    rows = {"image": [], "video": [], "audio": []}
    image_path = str(assets["image"])
    video_path = str(assets["video"])
    audio_path = str(assets["audio"])
    for idx in range(size):
        rows["image"].append(_image_row(split, idx, image_path))
        rows["video"].append(_video_row(split, idx, video_path))
        rows["audio"].append(_audio_row(split, idx, audio_path))
    return rows


def _write_modality_parquet(root: str, modality: str, train_rows: list[dict], val_rows: list[dict]) -> None:
    modality_dir = os.path.join(root, modality)
    os.makedirs(modality_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(modality_dir, "train.parquet"), index=False)
    pd.DataFrame(val_rows).to_parquet(os.path.join(modality_dir, "test.parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy Omni-Preference parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_omni_preference_dpo"),
        help="Directory to write modality parquet folders and media files",
    )
    parser.add_argument("--train_size", type=int, default=4, help="Rows per modality for train split")
    parser.add_argument("--val_size", type=int, default=2, help="Rows per modality for val split")
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.local_save_dir))
    assets = _make_assets(root)
    train_rows = _build_rows("train", args.train_size, assets)
    val_rows = _build_rows("test", args.val_size, assets)

    for modality in ("image", "video", "audio"):
        _write_modality_parquet(root, modality, train_rows[modality], val_rows[modality])
        print(
            f"Wrote {len(train_rows[modality])} train and {len(val_rows[modality])} val rows for {modality}",
            flush=True,
        )

    print(f"Dummy Omni-Preference data root: {root}", flush=True)


if __name__ == "__main__":
    main()
