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

Parquet rows are expected to be produced by
``examples/dpo_trainer/data_process/prepare_offline_dpo.py``, which stores plain
captions in ``extra_info["raw_prompt"]`` and ``extra_info["raw_negative_prompt"]``.
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
from verl.utils.dataset.rl_dataset import collate_fn as _upstream_collate_fn

OFFLINE_DPO_PAIR_MARKER = "__offline_dpo_pair__"


def _read_dataframe(data_files: str | Sequence[str]) -> pd.DataFrame:
    paths = [data_files] if isinstance(data_files, str) else list(data_files)
    frames = []
    for data_file in paths:
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


def _extra_info_dict(extra_info: Any) -> dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if extra_info is None:
        return {}
    return {"raw_extra_info": extra_info}


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _user_text_from_chat(prompt: Any) -> str:
    """Fallback when ``extra_info`` does not contain raw caption text."""
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return "" if prompt is None else str(prompt)
    user_parts = []
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str) and content:
                user_parts.append(content)
    return "\n".join(user_parts)


def _resolve_raw_prompts(
    prompt: Any,
    negative_prompt: Any,
    extra_info: Any,
) -> tuple[str, str]:
    info = _extra_info_dict(extra_info)
    raw_prompt = _plain_text(info.get("raw_prompt")) or _user_text_from_chat(prompt)
    raw_negative_prompt = _plain_text(info.get("raw_negative_prompt")) or _user_text_from_chat(negative_prompt)
    return raw_prompt, raw_negative_prompt


def _tokenize_prompt(prompt: Any, tokenizer, config: DictConfig) -> torch.Tensor:
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
            **config.get("apply_chat_template_kwargs", {}),
        )
    else:
        text = _user_text_from_chat(prompt)

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


def _tensor_from_column(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        raise ValueError("Offline DPO parquet contains a missing tensor column value.")
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        raise ValueError("Offline DPO parquet contains a missing tensor column value.")

    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    if isinstance(value, bytes | bytearray | memoryview):
        buffer = io.BytesIO(bytes(value))
        try:
            tensor = torch.load(buffer, map_location="cpu", weights_only=True)
        except TypeError:
            buffer.seek(0)
            tensor = torch.load(buffer, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected serialized tensor bytes, got {type(tensor)} after torch.load.")
        return tensor.to(dtype=dtype)

    def _to_nested(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item
        if isinstance(item, np.ndarray):
            return [_to_nested(x) for x in item.tolist()]
        if isinstance(item, list | tuple):
            return [_to_nested(x) for x in item]
        return item

    return torch.tensor(_to_nested(value), dtype=dtype)


class OfflineDPODataset(Dataset):
    """Dataset for rows containing offline DPO pairs plus precomputed SD3 tensors."""

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        del processor
        if config is None:
            raise ValueError("OfflineDPODataset requires a data config.")
        self.data_files = [data_files] if isinstance(data_files, str) else list(data_files)
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
        self.default_negative_prompt = config.get("default_negative_prompt", "")
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

        extra_info = _extra_info_dict(row.get("extra_info"))
        raw_prompt, raw_negative_prompt = _resolve_raw_prompts(prompt, negative_prompt, extra_info)
        extra_info = {
            **extra_info,
            "index": int(item),
            "raw_prompt": raw_prompt,
            "raw_negative_prompt": raw_negative_prompt,
        }

        def _optional_tensor(key: str, dtype: torch.dtype) -> torch.Tensor | None:
            if key not in row:
                return None
            value = row[key]
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            return _tensor_from_column(value, dtype=dtype)

        return {
            OFFLINE_DPO_PAIR_MARKER: True,
            "prompts": _tokenize_prompt(prompt, self.tokenizer, self.config),
            "uid": pair_uid,
            "raw_prompt": raw_prompt,
            "raw_negative_prompt": raw_negative_prompt,
            "img_win": _resolve_path(row[self.win_key], data_file),
            "img_lose": _resolve_path(row[self.lose_key], data_file),
            "img_win_latents": _tensor_from_column(row["img_win_latents"], dtype=torch.float32),
            "img_lose_latents": _tensor_from_column(row["img_lose_latents"], dtype=torch.float32),
            "prompt_embeds": _tensor_from_column(row["prompt_embeds"], dtype=torch.float32),
            "prompt_embeds_mask": _tensor_from_column(row["prompt_embeds_mask"], dtype=torch.int32),
            "pooled_prompt_embeds": _tensor_from_column(row["pooled_prompt_embeds"], dtype=torch.float32),
            "win_score": win_score,
            "lose_score": lose_score,
            "data_source": row.get("data_source", self.data_source),
            "reward_model": row.get("reward_model", {"style": "model", "ground_truth": raw_prompt}),
            "extra_info": extra_info,
            "negative_prompt_embeds": _optional_tensor("negative_prompt_embeds", torch.float32),
            "negative_prompt_embeds_mask": _optional_tensor("negative_prompt_embeds_mask", torch.int32),
            "negative_pooled_prompt_embeds": _optional_tensor("negative_pooled_prompt_embeds", torch.float32),
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
            "raw_prompt": feature["raw_prompt"],
            "raw_negative_prompt": feature["raw_negative_prompt"],
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


def offline_dpo_collate_fn(features):
    """Collate offline DPO pairs after expanding each row to win/lose samples."""
    if features and isinstance(features[0], dict) and features[0].get(OFFLINE_DPO_PAIR_MARKER):
        features = expand_offline_dpo_features(features)
    return _upstream_collate_fn(features)
