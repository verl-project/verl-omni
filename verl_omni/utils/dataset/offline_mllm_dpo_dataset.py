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

"""Offline MLLM DPO dataset for Omni-Preference style parquet rows."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from verl.utils.dataset.rl_dataset import collate_fn as _upstream_collate_fn

_SOURCE_NAMES = (
    "Omni-Preference-Image",
    "Omni-Preference-Video",
    "Omni-Preference-Audio",
)


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
        raise ValueError("Offline MLLM DPO dataset requires at least one data file.")
    return pd.concat(frames, ignore_index=True)


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _append_content(conversation: list[Any], content: Any, media: dict[str, list[Any]]) -> None:
    content = _as_python(content)
    if isinstance(content, str):
        conversation.append(("text", content))
        return

    for item in content or []:
        item = _as_python(item)
        if not isinstance(item, dict):
            conversation.append(("text", str(item)))
            continue

        item_type = item.get("type")
        if item_type == "text":
            conversation.append(("text", item.get("text", "")))
        elif item_type == "image":
            media["images"].append(item.get("image"))
            conversation.append(("image", None))
        elif item_type == "video":
            media["videos"].append(item.get("video"))
            conversation.append(("video", None))
        elif item_type == "audio":
            media["audios"].append(item.get("audio"))
            conversation.append(("audio", None))
        else:
            conversation.append(("text", str(item)))


def _build_preference_branch(sample: dict[str, Any], answer: str) -> dict[str, Any]:
    prompt = _as_python(sample.get("prompt", []))
    media: dict[str, list[Any]] = {"images": [], "videos": [], "audios": []}
    conversations: list[list[Any]] = []

    for message in prompt:
        message = _as_python(message)
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        conversation = [role or "user"]
        _append_content(conversation, message.get("content", ""), media)
        if len(conversation) > 1:
            conversations.append(conversation)

    conversations.append(["assistant", ("text", str(_as_python(answer)))])
    branch = {
        "conversations": conversations,
        "source_name": sample.get("source_name") or sample.get("data_source"),
    }
    for key, values in media.items():
        if values:
            branch[key] = values
    return branch


def _cat_sequence_tensors(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    dim = -1 if chosen.ndim > 1 else 0
    return torch.cat([chosen, rejected], dim=dim)


def _merge_chosen_rejected(chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in chosen.keys() | rejected.keys():
        chosen_value = chosen.get(key)
        rejected_value = rejected.get(key)
        if chosen_value is None:
            merged[key] = rejected_value
            continue
        if rejected_value is None:
            merged[key] = chosen_value
            continue
        if not isinstance(chosen_value, torch.Tensor) or not isinstance(rejected_value, torch.Tensor):
            merged[key] = chosen_value
            continue
        if key in {"input_ids", "attention_mask", "labels", "position_ids", "image_mask", "video_mask", "audio_mask"}:
            merged[key] = _cat_sequence_tensors(chosen_value, rejected_value)
        else:
            merged[key] = torch.cat([chosen_value, rejected_value], dim=0)
    return merged


def _register_pass_through_preprocessors(source_names: Sequence[str]) -> None:
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY

    def _pass_through(conversations, **kwargs):
        return conversations

    for source_name in source_names:
        try:
            PREPROCESSOR_REGISTRY[source_name]
        except (KeyError, ValueError):
            PREPROCESSOR_REGISTRY.register(source_name)(_pass_through)


def _prepare_qwen3_omni_processor(processor):
    class ProcessorProxy:
        def __getattr__(self, name):
            return getattr(processor, name)

        def __call__(self, *args, **kwargs):
            if kwargs.get("audios"):
                kwargs["audio"] = kwargs.pop("audios")
            else:
                kwargs.pop("audios", None)
            kwargs = {key: value for key, value in kwargs.items() if value != []}
            return processor(*args, **kwargs)

    def get_rope_index(*args, **kwargs):
        result = processor.get_rope_index(*args, **kwargs)
        if isinstance(result, dict):
            return result
        position_ids, mrope_position_deltas = result
        return {"position_ids": position_ids, "mrope_position_deltas": mrope_position_deltas}

    proxy = ProcessorProxy()
    if hasattr(processor, "get_rope_index"):
        proxy.get_rope_index = get_rope_index
    return proxy


def _transform_sample(sample: dict[str, Any], base_transform, transform_kwargs: dict[str, Any]) -> dict[str, Any]:
    chosen_sample = _build_preference_branch(sample, sample["chosen"])
    rejected_sample = _build_preference_branch(sample, sample["rejected"])
    chosen = base_transform(chosen_sample, **transform_kwargs)[0]
    rejected = base_transform(rejected_sample, **transform_kwargs)[0]
    return _merge_chosen_rejected(chosen, rejected)


def _normalise_source_name(value: Any, default: str) -> str:
    text = str(value or default)
    lowered = text.lower()
    if "image" in lowered:
        return "Omni-Preference-Image"
    if "video" in lowered:
        return "Omni-Preference-Video"
    if "audio" in lowered:
        return "Omni-Preference-Audio"
    return text


class OfflineMLLMDPODataset(Dataset):
    """Dataset for Omni-Preference rows consumed by VeOmni Qwen3-Omni DPO."""

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        del tokenizer
        if config is None:
            raise ValueError("OfflineMLLMDPODataset requires a data config.")
        if processor is None:
            raise ValueError("OfflineMLLMDPODataset requires a multimodal processor.")

        from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY

        self.dataframe = _read_dataframe(data_files)
        if max_samples is not None and max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        self.config = config
        self.processor = _prepare_qwen3_omni_processor(processor)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")
        self.win_score_key = config.get("win_score_key", "win_score")
        self.lose_score_key = config.get("lose_score_key", "lose_score")
        self.source_name_key = config.get("source_name_key", "data_source")
        self.data_source = config.get("data_source", "offline_mllm_dpo")

        source_names = tuple(config.get("source_names", _SOURCE_NAMES))
        _register_pass_through_preprocessors(source_names)

        mm_configs = config.get("mm_configs", {})
        if isinstance(mm_configs, DictConfig):
            mm_configs = OmegaConf.to_container(mm_configs, resolve=True)
        self.transform_kwargs = dict(mm_configs or {})
        if "position_id_func" not in self.transform_kwargs and hasattr(self.processor, "get_rope_index"):
            self.transform_kwargs["position_id_func"] = self.processor.get_rope_index
        self.base_transform = DATA_TRANSFORM_REGISTRY[config.get("base_transform", "qwen3_omni_moe")]

        required = {self.prompt_key, self.chosen_key, self.rejected_key}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Offline MLLM DPO data is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe.iloc[item].to_dict()
        source_name = _normalise_source_name(
            row.get(self.source_name_key) or row.get("source_name"),
            self.data_source,
        )
        sample = {
            "prompt": row[self.prompt_key],
            "chosen": row[self.chosen_key],
            "rejected": row[self.rejected_key],
            "source_name": source_name,
        }
        transformed = _transform_sample(
            sample, self.base_transform, {"processor": self.processor, **self.transform_kwargs}
        )
        transformed["uid"] = str(row.get("uid") or uuid.uuid4())
        transformed["sample_level_scores"] = torch.tensor(
            [float(row.get(self.win_score_key, 1.0)), float(row.get(self.lose_score_key, 0.0))],
            dtype=torch.float32,
        )
        transformed["data_source"] = row.get(self.source_name_key) or self.data_source
        transformed["reward_model"] = row.get("reward_model", {"style": "model", "ground_truth": row[self.chosen_key]})
        transformed["extra_info"] = row.get("extra_info", {"index": int(item)})
        return transformed


def offline_mllm_dpo_collate_fn(features):
    return _upstream_collate_fn(features)
