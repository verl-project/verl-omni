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

"""Offline diffusion DPO dataset utilities.

The on-policy DPO path forms pairs after rollout and reward scoring. Offline DPO
receives those pairs directly, so each parquet row is a logical pair and the
collate step expands it to adjacent ``chosen, rejected`` samples.
"""

import io
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

OFFLINE_DPO_PAIR_MARKER = "__offline_dpo_pair__"


def _as_list(data_files: str | Sequence[str]) -> list[str]:
    if isinstance(data_files, str):
        return [data_files]
    return list(data_files)


def _read_dataframe(data_files: str | Sequence[str]) -> pd.DataFrame:
    frames = []
    for data_file in _as_list(data_files):
        path = Path(os.path.expanduser(data_file))
        if path.suffix == ".jsonl":
            frames.append(pd.read_json(path, lines=True))
        elif path.suffix == ".json":
            frames.append(pd.read_json(path))
        else:
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("Offline DPO dataset requires at least one data file.")
    return pd.concat(frames, ignore_index=True)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _coerce_extra_info(extra_info: Any) -> dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if extra_info is None:
        return {}
    return {"raw_extra_info": extra_info}


def _plain_text_from_extra_info(extra_info: dict[str, Any], key: str) -> str | None:
    """Return non-empty plain text from ``extra_info[key]`` when present."""
    if key not in extra_info:
        return None
    value = extra_info[key]
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    text = str(value).strip()
    return text if text else None


def resolve_materialize_prompts(
    prompt: Any,
    negative_prompt: Any,
    extra_info: Any,
) -> tuple[str, str]:
    """Resolve plain prompts for SD3 materialization.

    Prefer ``extra_info["raw_prompt"]`` and ``extra_info["raw_negative_prompt"]``
    (written by ``prepare_offline_dpo``). Fall back to parsing chat-style
    ``prompt`` / ``negative_prompt`` columns when extra_info text is missing.
    """
    info = _coerce_extra_info(extra_info)
    raw_prompt = _plain_text_from_extra_info(info, "raw_prompt")
    if raw_prompt is None:
        raw_prompt = prompt_to_text(prompt)

    raw_negative_prompt = _plain_text_from_extra_info(info, "raw_negative_prompt")
    if raw_negative_prompt is None:
        raw_negative_prompt = prompt_to_text(negative_prompt)
    return raw_prompt, raw_negative_prompt


def prompt_to_text(prompt: Any) -> str:
    """Extract plain caption text for diffusion text encoders and reward metadata."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        user_parts = []
        all_parts = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            text = _message_content_to_text(message.get("content"))
            if text:
                all_parts.append(text)
            if message.get("role") == "user" and text:
                user_parts.append(text)
        return "\n".join(user_parts or all_parts)
    return "" if prompt is None else str(prompt)


def _tokenize_prompt(prompt: Any, tokenizer, config: DictConfig) -> torch.Tensor:
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
            **config.get("apply_chat_template_kwargs", {}),
        )
    else:
        text = prompt_to_text(prompt)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length,
    )["input_ids"][0]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if encoded.shape[0] < config.max_prompt_length:
        pad = torch.full((config.max_prompt_length - encoded.shape[0],), pad_token_id, dtype=encoded.dtype)
        encoded = torch.cat((pad, encoded), dim=0)
    return encoded[-config.max_prompt_length :]


def _resolve_path(path: Any, data_file: str | None = None) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or data_file is None:
        return path
    return os.path.normpath(os.path.join(os.path.dirname(os.path.expanduser(data_file)), path))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_numeric_nested(value: Any) -> Any:
    """Convert parquet object arrays back to plain nested numeric lists."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return _to_numeric_nested(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_to_numeric_nested(item) for item in value]
    return value


def _tensor_from_bytes(value: bytes | bytearray | memoryview, *, dtype: torch.dtype) -> torch.Tensor:
    buffer = io.BytesIO(bytes(value))
    try:
        tensor = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        tensor = torch.load(buffer, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected serialized tensor bytes, got {type(tensor)} after torch.load.")
    return tensor.to(dtype=dtype)


def _tensor_from_column(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if _is_missing(value):
        raise ValueError("Offline DPO parquet contains a missing tensor column value.")
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _tensor_from_bytes(value, dtype=dtype)
    value = _to_numeric_nested(value)
    return torch.tensor(value, dtype=dtype)


def _optional_tensor_from_row(row: dict[str, Any], key: str, *, dtype: torch.dtype) -> torch.Tensor | None:
    if key not in row or _is_missing(row[key]):
        return None
    return _tensor_from_column(row[key], dtype=dtype)


class OfflineDPODataset(Dataset):
    """Dataset for rows containing offline DPO pairs plus precomputed SD3 tensors."""

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        del processor
        if config is None:
            raise ValueError("OfflineDPODataset requires a data config.")
        self.data_files = _as_list(data_files)
        self.dataframe = _read_dataframe(self.data_files)
        if max_samples is not None and max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_key = config.get("prompt_key", "prompt")
        self.negative_prompt_key = config.get("negative_prompt_key", "negative_prompt")
        self.win_key = config.get("img_win_key", "img_win")
        self.lose_key = config.get("img_lose_key", "img_lose")
        self.win_score_key = config.get("win_score_key", "win_score")
        self.lose_score_key = config.get("lose_score_key", "lose_score")
        self.default_negative_prompt = config.get("default_negative_prompt", " ")
        self.data_source = config.get("data_source", "offline_dpo")

        required = {
            self.prompt_key,
            self.win_key,
            self.lose_key,
            "img_win_latents",
            "img_lose_latents",
            "prompt_embeds",
            "prompt_embeds_mask",
            "pooled_prompt_embeds",
        }
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Offline DPO data is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe.iloc[item].to_dict()
        prompt = row[self.prompt_key]
        negative_prompt = row.get(self.negative_prompt_key, self.default_negative_prompt)
        data_file = self.data_files[0] if len(self.data_files) == 1 else None
        pair_uid = str(row.get("uid") or uuid.uuid4())

        win_score = float(row.get(self.win_score_key, 1.0))
        lose_score = float(row.get(self.lose_score_key, 0.0))
        if win_score < lose_score:
            raise ValueError(f"Offline DPO row {item} has win_score < lose_score: {win_score} < {lose_score}")

        extra_info = _coerce_extra_info(row.get("extra_info"))
        raw_prompt, raw_negative_prompt = resolve_materialize_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            extra_info=extra_info,
        )
        extra_info = {
            **extra_info,
            "index": int(item),
            "raw_prompt": raw_prompt,
            "raw_negative_prompt": raw_negative_prompt,
        }

        return {
            OFFLINE_DPO_PAIR_MARKER: True,
            "prompts": _tokenize_prompt(prompt, self.tokenizer, self.config),
            "uid": pair_uid,
            "prompt_text": raw_prompt,
            "negative_prompt_text": raw_negative_prompt,
            "img_win": _resolve_path(row[self.win_key], data_file),
            "img_lose": _resolve_path(row[self.lose_key], data_file),
            "img_win_latents": _tensor_from_column(row["img_win_latents"], dtype=torch.float32),
            "img_lose_latents": _tensor_from_column(row["img_lose_latents"], dtype=torch.float32),
            "prompt_embeds": _tensor_from_column(row["prompt_embeds"], dtype=torch.float32),
            "prompt_embeds_mask": _tensor_from_column(row["prompt_embeds_mask"], dtype=torch.int32),
            "pooled_prompt_embeds": _tensor_from_column(row["pooled_prompt_embeds"], dtype=torch.float32),
            "negative_prompt_embeds": _optional_tensor_from_row(row, "negative_prompt_embeds", dtype=torch.float32),
            "negative_prompt_embeds_mask": _optional_tensor_from_row(
                row, "negative_prompt_embeds_mask", dtype=torch.int32
            ),
            "negative_pooled_prompt_embeds": _optional_tensor_from_row(
                row, "negative_pooled_prompt_embeds", dtype=torch.float32
            ),
            "win_score": win_score,
            "lose_score": lose_score,
            "data_source": row.get("data_source", self.data_source),
            "reward_model": row.get("reward_model", {"style": "model", "ground_truth": raw_prompt}),
            "extra_info": extra_info,
        }


def expand_offline_dpo_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand logical DPO pairs into adjacent chosen/rejected samples."""
    expanded = []
    for feature in features:
        if not feature.get(OFFLINE_DPO_PAIR_MARKER):
            expanded.append(feature)
            continue

        base = {
            "prompts": feature["prompts"],
            "uid": feature["uid"],
            "raw_prompt": feature["prompt_text"],
            "raw_negative_prompt": feature["negative_prompt_text"],
            "data_source": feature["data_source"],
            "reward_model": feature["reward_model"],
            "extra_info": feature["extra_info"],
            "prompt_embeds": feature["prompt_embeds"],
            "prompt_embeds_mask": feature["prompt_embeds_mask"],
            "pooled_prompt_embeds": feature["pooled_prompt_embeds"],
        }
        for key in ("negative_prompt_embeds", "negative_prompt_embeds_mask", "negative_pooled_prompt_embeds"):
            if feature.get(key) is not None:
                base[key] = feature[key]
        expanded.append(
            {
                **base,
                "image_path": feature["img_win"],
                "image_latents": feature["img_win_latents"],
                "sample_level_scores": torch.tensor([feature["win_score"]], dtype=torch.float32),
                "is_chosen": True,
            }
        )
        expanded.append(
            {
                **base,
                "image_path": feature["img_lose"],
                "image_latents": feature["img_lose_latents"],
                "sample_level_scores": torch.tensor([feature["lose_score"]], dtype=torch.float32),
                "is_chosen": False,
            }
        )
    return expanded
