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

"""Uni-COT dataset helpers for supervised fine-tuning.

The Uni-COT record is an interleaved visual reasoning trajectory:

For visual reasoning / editing / VLM SFT rows, ``image_list[0]`` is treated as
the context image by default. For T2I rows, all images are generation targets.
Generated images are opened by ``<image_start>`` markers in ``output_text_list``.
Text spans are supervised with causal CE; generated image spans can be
supervised by a model-specific image objective after preprocessing.

This module keeps heavy image encoding out of the dataset. It emits a
deterministic event stream, padded text tensors, image paths, and optional
precomputed image-training tensors when the source dataset provides them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGE_START = "<image_start>"
IMAGE_END = "<image_end>"
IGNORE_INDEX = -100

__all__ = [
    "IMAGE_START",
    "IMAGE_END",
    "IGNORE_INDEX",
    "UniCOTEvent",
    "build_unicot_events",
    "UniCOTSFTDataset",
    "unicot_sft_collate_fn",
]


@dataclass(frozen=True)
class UniCOTEvent:
    """One teacher-forced event in a Uni-COT SFT sequence."""

    type: str
    text: str | None = None
    image_path: str | None = None
    supervise: bool = False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
    return [value]


def _strip_structural_markers(text: str) -> str:
    return text.replace(IMAGE_END, "").strip()


def _append_text_event(events: list[UniCOTEvent], text: str, *, supervise: bool) -> None:
    text = _strip_structural_markers(text)
    if text:
        events.append(UniCOTEvent(type="text", text=text, supervise=supervise))


def build_unicot_events(
    image_list: Iterable[str],
    instruction_list: Iterable[str],
    output_text_list: Iterable[str],
    *,
    num_context_images: int = 1,
) -> list[UniCOTEvent]:
    """Convert a Uni-COT row into an explicit SFT event stream."""

    images = [str(path) for path in _as_list(image_list)]
    instructions = [str(text) for text in _as_list(instruction_list)]
    outputs = [str(text) for text in _as_list(output_text_list)]

    if not images:
        raise ValueError("Uni-COT sample must contain at least one image in image_list.")

    if num_context_images < 0:
        raise ValueError(f"num_context_images must be non-negative, got {num_context_images}.")
    if num_context_images > len(images):
        raise ValueError("num_context_images cannot exceed the number of images in image_list.")

    events: list[UniCOTEvent] = [
        UniCOTEvent(type="context_image", image_path=image_path, supervise=False)
        for image_path in images[:num_context_images]
    ]
    if instructions:
        events.append(UniCOTEvent(type="text", text="\n".join(instructions).strip(), supervise=False))

    next_generated_image = num_context_images
    for output in outputs:
        remainder = output
        while IMAGE_START in remainder:
            before, remainder = remainder.split(IMAGE_START, 1)
            _append_text_event(events, before, supervise=True)
            if next_generated_image >= len(images):
                raise ValueError(
                    "Uni-COT output contains more <image_start> markers than generated images in image_list."
                )
            events.append(UniCOTEvent(type="generated_image", image_path=images[next_generated_image], supervise=True))
            next_generated_image += 1
        _append_text_event(events, remainder, supervise=True)

    return events


def _read_local_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else payload.get("data", [])
    raise ValueError(f"Unsupported Uni-COT data file type: {path}")


def _load_rows(source: str | os.PathLike[str], split: str) -> list[dict[str, Any]]:
    source = str(source)
    if os.path.exists(source):
        return _read_local_rows(source)

    from datasets import load_dataset

    return list(load_dataset(source, split=split))


def _config_get(config: dict[str, Any], key: str, default=None):
    if key in config:
        return config.get(key, default)
    custom_cls = config.get("custom_cls", None)
    if custom_cls is not None and key in custom_cls:
        return custom_cls.get(key, default)
    return default


def _read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload or {}


def _normalise_task_type(value: Any) -> str:
    task_type = str(value or "unicot").lower()
    if task_type in {"text_editing", "image_editing", "edit"}:
        return "editing"
    if task_type in {"vlm", "vlm_sft", "understanding"}:
        return "vlm_sft"
    if task_type in {"t2i", "text_to_image"}:
        return "t2i"
    return task_type


def _num_context_images_for_task(task_type: str) -> int:
    # TorchUMM/BAGEL-style task groups:
    # - T2I: text prompt -> target image, no context image.
    # - Editing / VLM SFT / Uni-COT reasoning: first image is input context.
    return 0 if _normalise_task_type(task_type) == "t2i" else 1


class UniCOTSFTDataset(Dataset):
    """Dataset for Uni-COT supervised visual reasoning trajectories."""

    def __init__(
        self,
        data_files,
        tokenizer=None,
        config: dict[str, Any] | None = None,
        is_train: bool = True,
        **kwargs,
    ):
        del kwargs
        self.config = config or {}
        self.tokenizer = tokenizer
        if isinstance(data_files, (str | os.PathLike)):
            files = [data_files]
        else:
            files = list(data_files)

        dataset_config_file = _config_get(self.config, "dataset_config_file", None)
        self.dataset_config = _read_yaml(dataset_config_file) if dataset_config_file else {}
        self.task_type_key = _config_get(self.config, "task_type_key", "task_type")
        self.default_task_type = _normalise_task_type(_config_get(self.config, "task_type", "unicot"))
        split = _config_get(self.config, "split", None)
        if split is None:
            split = _config_get(self.config, "train_split" if is_train else "val_split", "train")
        self.rows = [row for data_file in files for row in _load_rows(data_file, split=split)]
        self.image_key = _config_get(self.config, "image_key", "image_list")
        self.instruction_key = _config_get(self.config, "instruction_key", "instruction_list")
        self.output_key = _config_get(self.config, "output_key", "output_text_list")
        self.max_text_length = int(_config_get(self.config, "max_text_length", 8192))

    def __len__(self) -> int:
        return len(self.rows)

    def _tokenize_events(self, events: list[UniCOTEvent]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids: list[int] = []
        labels: list[int] = []
        if self.tokenizer is None:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        for event in events:
            if event.type != "text" or not event.text:
                continue
            token_ids = self.tokenizer.encode(event.text, add_special_tokens=False)
            token_ids = [int(token_id) for token_id in token_ids]
            input_ids.extend(token_ids)
            labels.extend(token_ids if event.supervise else [IGNORE_INDEX] * len(token_ids))

        input_ids = input_ids[: self.max_text_length]
        labels = labels[: self.max_text_length]
        attention_mask = [1] * len(input_ids)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        task_type = _normalise_task_type(row.get(self.task_type_key, self.default_task_type))
        num_context_images = int(
            row.get(
                "num_context_images",
                _config_get(self.config, "num_context_images", _num_context_images_for_task(task_type)),
            )
        )
        events = build_unicot_events(
            row[self.image_key],
            row.get(self.instruction_key, []),
            row.get(self.output_key, []),
            num_context_images=num_context_images,
        )
        input_ids, labels, attention_mask = self._tokenize_events(events)
        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "unicot_sft_events": [event.__dict__ for event in events],
            "context_image_paths": [event.image_path for event in events if event.type == "context_image"],
            "generated_image_paths": [event.image_path for event in events if event.type == "generated_image"],
            "task_type": task_type,
            "data_source": row.get("data_source", f"unicot_sft/{task_type}"),
            "extra_info": row.get("extra_info", {"index": int(index)}),
        }
        for key in ("image_hidden_states", "image_velocity_target", "image_loss_mask", "timesteps", "latent_pos_ids"):
            if key in row and row[key] is not None:
                sample[key] = torch.as_tensor(row[key])
        return sample


def _pad_1d(tensor: torch.Tensor, length: int, value: int) -> torch.Tensor:
    if tensor.numel() >= length:
        return tensor
    return torch.nn.functional.pad(tensor, (0, length - tensor.numel()), value=value)


def unicot_sft_collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad Uni-COT SFT samples and keep event metadata as Python lists."""

    max_len = max((feature["input_ids"].numel() for feature in features), default=0)
    batch = {
        "input_ids": torch.stack([_pad_1d(feature["input_ids"], max_len, 0) for feature in features]),
        "labels": torch.stack([_pad_1d(feature["labels"], max_len, IGNORE_INDEX) for feature in features]),
        "attention_mask": torch.stack([_pad_1d(feature["attention_mask"], max_len, 0) for feature in features]),
        "unicot_sft_events": [feature["unicot_sft_events"] for feature in features],
        "context_image_paths": [feature["context_image_paths"] for feature in features],
        "generated_image_paths": [feature["generated_image_paths"] for feature in features],
        "task_type": [feature["task_type"] for feature in features],
        "data_source": [feature["data_source"] for feature in features],
        "extra_info": [feature["extra_info"] for feature in features],
    }
    for key in ("image_hidden_states", "image_velocity_target", "image_loss_mask", "timesteps", "latent_pos_ids"):
        if all(key in feature for feature in features):
            batch[key] = torch.stack([feature[key] for feature in features])
    return batch
