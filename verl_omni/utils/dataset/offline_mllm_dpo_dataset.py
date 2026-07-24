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
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Sampler

from verl_omni.utils.dataset.qwen3_omni_transform import process_qwen3_omni_sample

_MEDIA_TOKEN_PATTERN = re.compile(r"<(image|video|audio)>")


def _read_dataframe(data_files: str | Sequence[str]) -> pd.DataFrame:
    paths = [data_files] if isinstance(data_files, (str | Path)) else list(data_files)
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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return value is pd.NA


def _append_media_path(media: dict[str, list[Any]], key: str, value: Any) -> None:
    if _is_missing(value):
        return
    if value not in media[key]:
        media[key].append(value)


def _normalise_media_list(value: Any) -> list[Any]:
    value = _as_python(value)
    if _is_missing(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [item for item in value if not _is_missing(item)]
    return [value]


def _initial_media(sample: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "images": _normalise_media_list(sample.get("images")),
        "videos": _normalise_media_list(sample.get("videos")),
        "audios": _normalise_media_list(sample.get("audios")),
    }


def _answer_text(answer: Any) -> str:
    answer = _as_python(answer)
    if isinstance(answer, dict):
        if "content" in answer:
            return _content_to_text(answer["content"])
        if "text" in answer:
            return str(answer["text"])
    return _content_to_text(answer)


def _content_to_text(content: Any) -> str:
    content = _as_python(content)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        if "content" in content:
            return _content_to_text(content["content"])
        if "text" in content:
            return str(content["text"])
        return str(content)
    if isinstance(content, Sequence):
        parts = [_content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if _is_missing(content):
        return ""
    return str(content)


def _append_string_content(conversation: list[Any], content: str) -> None:
    cursor = 0
    for match in _MEDIA_TOKEN_PATTERN.finditer(content):
        text = content[cursor : match.start()]
        if text:
            conversation.append(("text", text))
        conversation.append((match.group(1), None))
        cursor = match.end()
    remaining = content[cursor:]
    if remaining:
        conversation.append(("text", remaining))


def _append_content(conversation: list[Any], content: Any, media: dict[str, list[Any]]) -> None:
    content = _as_python(content)
    if isinstance(content, str):
        _append_string_content(conversation, content)
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
            _append_media_path(media, "images", item.get("image"))
            conversation.append(("image", None))
        elif item_type == "video":
            _append_media_path(media, "videos", item.get("video"))
            conversation.append(("video", None))
        elif item_type == "audio":
            _append_media_path(media, "audios", item.get("audio"))
            conversation.append(("audio", None))
        else:
            conversation.append(("text", str(item)))


def _count_media_tokens(conversations: Sequence[Sequence[Any]], modality: str) -> int:
    count = 0
    for conversation in conversations:
        for item in conversation[1:]:
            if isinstance(item, (list | tuple)) and item and item[0] == modality:
                count += 1
    return count


def _validate_media_alignment(conversations: Sequence[Sequence[Any]], media: dict[str, list[Any]]) -> None:
    for modality, media_key in (("image", "images"), ("video", "videos"), ("audio", "audios")):
        token_count = _count_media_tokens(conversations, modality)
        media_count = len(media[media_key])
        if token_count != media_count:
            raise ValueError(
                f"Prompt contains {token_count} <{modality}> token(s) but {media_key} has {media_count} item(s). "
                "Ensure compact multimodal rows include matching top-level media paths."
            )


def _build_preference_branch(sample: dict[str, Any], answer: Any) -> dict[str, Any]:
    prompt = _as_python(sample.get("prompt", []))
    media = _initial_media(sample)
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

    _validate_media_alignment(conversations, media)
    conversations.append(["assistant", ("text", _answer_text(answer))])
    branch = {
        "conversations": conversations,
        "source_name": sample.get("source_name") or sample.get("data_source"),
    }
    for key, values in media.items():
        if values:
            branch[key] = values
    return branch


def _pad_tensor_to_shape(tensor: torch.Tensor, shape: Sequence[int], pad_value: float | int | bool = 0) -> torch.Tensor:
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    output = torch.full(tuple(shape), pad_value, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, size) for size in tensor.shape)
    output[slices] = tensor
    return output


def _pad_value_for_key(key: str) -> float | int | bool:
    if key == "labels":
        return -100
    if key == "attention_mask":
        return 0
    return 0


def _stack_branch_tensors(key: str, chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    max_shape = tuple(max(chosen.shape[dim], rejected.shape[dim]) for dim in range(chosen.ndim))
    pad_value = _pad_value_for_key(key)
    return torch.stack(
        [
            _pad_tensor_to_shape(chosen, max_shape, pad_value),
            _pad_tensor_to_shape(rejected, max_shape, pad_value),
        ],
        dim=0,
    )


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
        merged[key] = _stack_branch_tensors(key, chosen_value, rejected_value)
    return merged


def _collate_tensor_values(key: str, values: Sequence[torch.Tensor | None]) -> torch.Tensor:
    present = [value for value in values if value is not None]
    if not present:
        raise ValueError(f"Cannot collate tensor key {key!r} without any tensor values.")

    max_shape = tuple(max(value.shape[dim] for value in present) for dim in range(present[0].ndim))
    pad_value = _pad_value_for_key(key)

    padded = []
    for value in values:
        if value is None:
            value = torch.zeros(max_shape, dtype=present[0].dtype, device=present[0].device)
        padded.append(_pad_tensor_to_shape(value, max_shape, pad_value))
    return torch.stack(padded, dim=0)


def _prepare_qwen3_omni_processor(processor):
    class ProcessorProxy:
        def __getattr__(self, name):
            return getattr(processor, name)

        def __call__(self, *args, **kwargs):
            audios = kwargs.pop("audios", None)
            if audios:
                audios = [audio for audio in audios if audio is not None]
                if audios:
                    kwargs["audio"] = audios
            else:
                kwargs.pop("audio", None)
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


def _normalise_modality(value: Any, default: str = "unknown") -> str:
    text = str(value or default).lower()
    if "image" in text:
        return "image"
    if "video" in text:
        return "video"
    if "audio" in text:
        return "audio"
    return default


def _row_modality(row: dict[str, Any], source_name_key: str, default: str = "unknown") -> str:
    extra_info = _as_python(row.get("extra_info", {}))
    if isinstance(extra_info, dict) and extra_info.get("modality"):
        return _normalise_modality(extra_info["modality"], default)
    for key, modality in (("images", "image"), ("videos", "video"), ("audios", "audio")):
        if _normalise_media_list(row.get(key)):
            return modality
    return _normalise_modality(row.get(source_name_key) or row.get("source_name"), default)


class OfflineMLLMDPODataset(Dataset):
    """Dataset for Omni-Preference rows consumed by Qwen3-Omni offline DPO."""

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        del tokenizer
        if config is None:
            raise ValueError("OfflineMLLMDPODataset requires a data config.")
        if processor is None:
            raise ValueError("OfflineMLLMDPODataset requires a multimodal processor.")

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

        mm_configs = config.get("mm_configs", {})
        if isinstance(mm_configs, DictConfig):
            mm_configs = OmegaConf.to_container(mm_configs, resolve=True)
        self.transform_kwargs = dict(mm_configs or {})
        if "position_id_func" not in self.transform_kwargs and hasattr(self.processor, "get_rope_index"):
            self.transform_kwargs["position_id_func"] = self.processor.get_rope_index
        if "position_id_func" not in self.transform_kwargs:
            raise ValueError(
                "OfflineMLLMDPODataset requires `mm_configs.position_id_func` or a processor with "
                "`get_rope_index`. For Qwen3-Omni, bind "
                "`Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index` to the processor before "
                "constructing the dataset."
            )
        base_transform = config.get("base_transform", "qwen3_omni_moe")
        if base_transform not in {"qwen3_omni_moe", "qwen2_5_omni"}:
            raise ValueError(
                f"Unsupported base_transform {base_transform!r}. Expected one of: 'qwen3_omni_moe', 'qwen2_5_omni'."
            )
        self.base_transform = process_qwen3_omni_sample

        required = {self.prompt_key, self.chosen_key, self.rejected_key}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Offline MLLM DPO data is missing required columns: {sorted(missing)}")
        self.modalities = [
            _row_modality(row, self.source_name_key, self.data_source)
            for row in self.dataframe.to_dict(orient="records")
        ]

    def get_modality(self, item: int) -> str:
        return self.modalities[item]

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
            "images": row.get("images"),
            "videos": row.get("videos"),
            "audios": row.get("audios"),
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
        modality = self.get_modality(item)
        transformed["modality"] = modality
        extra_info = _as_python(row.get("extra_info", {"index": int(item)}))
        if isinstance(extra_info, dict):
            extra_info = {**extra_info, "modality": modality}
        transformed["extra_info"] = extra_info
        return transformed


class ModalityGroupedBatchSampler(Sampler[int]):
    """Yield indices in same-modality chunks for regular DataLoader batching.

    ``StatefulDataLoader`` is configured with ``sampler=`` and ``batch_size=``,
    not ``batch_sampler=``. This sampler therefore yields individual indices,
    but orders them as contiguous same-modality chunks of ``batch_size``. When
    Each chunk first samples a modality uniformly by default, or by
    ``modality_sample_weights`` when provided, then samples rows from that
    modality with replacement.
    """

    def __init__(
        self,
        data_source: Dataset | None = None,
        *,
        dataset: Dataset | None = None,
        data_config: DictConfig | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        modality_sample_weights: dict[str, float] | None = None,
        num_batches: int | None = None,
    ):
        del shuffle
        self.data_source = data_source if data_source is not None else dataset
        if self.data_source is None:
            raise ValueError("ModalityGroupedBatchSampler requires a dataset.")
        if not hasattr(self.data_source, "get_modality"):
            raise TypeError("ModalityGroupedBatchSampler requires a dataset with get_modality(index).")

        if batch_size is None and data_config is not None:
            batch_size = data_config.get("gen_batch_size", data_config.get("train_batch_size", None))
        if batch_size is None or batch_size <= 0:
            raise ValueError("ModalityGroupedBatchSampler requires a positive batch_size.")

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.modality_sample_weights = modality_sample_weights
        self.num_batches = num_batches
        self.epoch = 0
        self._batches = self._build_batches()
        self._length = sum(len(batch) for batch in self._batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _indices_by_modality(self) -> dict[str, list[int]]:
        indices_by_modality: dict[str, list[int]] = defaultdict(list)
        for index in range(len(self.data_source)):
            indices_by_modality[self.data_source.get_modality(index)].append(index)
        return dict(indices_by_modality)

    def _build_weighted_batches(
        self,
        indices_by_modality: dict[str, list[int]],
        generator: torch.Generator,
    ) -> list[list[int]]:
        weights_by_modality = self.modality_sample_weights or {}
        modalities = sorted(indices_by_modality)
        weights = []
        for modality in modalities:
            weight = float(weights_by_modality.get(modality, 1.0))
            if weight < 0:
                raise ValueError(f"modality_sample_weights[{modality!r}] must be non-negative, got {weight}.")
            weights.append(weight)
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        if weights_tensor.sum().item() <= 0:
            raise ValueError("modality_sample_weights must contain at least one positive weight.")

        if self.num_batches is not None:
            num_batches = int(self.num_batches)
        elif self.drop_last:
            num_batches = len(self.data_source) // self.batch_size
        else:
            num_batches = (len(self.data_source) + self.batch_size - 1) // self.batch_size
        if num_batches <= 0:
            return []

        batches: list[list[int]] = []
        for _ in range(num_batches):
            modality_index = torch.multinomial(
                weights_tensor, num_samples=1, replacement=True, generator=generator
            ).item()
            indices = indices_by_modality[modalities[modality_index]]
            sampled = torch.randint(len(indices), (self.batch_size,), generator=generator).tolist()
            batches.append([indices[index] for index in sampled])
        return batches

    def _build_batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices_by_modality = self._indices_by_modality()
        return self._build_weighted_batches(indices_by_modality, generator)

    def __iter__(self):
        for batch in self._build_batches():
            yield from batch

    def __len__(self) -> int:
        return self._length


def offline_mllm_dpo_collate_fn(features):
    modalities = {feature.get("modality") for feature in features}
    if len(modalities) != 1:
        raise ValueError(f"Offline MLLM DPO batches must contain a single modality, got {sorted(modalities)}")

    tensor_keys = {key for feature in features for key, value in feature.items() if isinstance(value, torch.Tensor)}
    non_tensor_keys = {
        key for feature in features for key, value in feature.items() if not isinstance(value, torch.Tensor)
    }

    batch: dict[str, Any] = {}
    for key in sorted(tensor_keys):
        batch[key] = _collate_tensor_values(key, [feature.get(key) for feature in features])
        if key == "position_ids" and batch[key].ndim == 4 and batch[key].shape[2] == 1:
            batch[key] = batch[key].squeeze(2).contiguous()
        if key == "position_ids" and batch[key].ndim == 5 and batch[key].shape[3] == 1:
            batch[key] = batch[key].squeeze(3).contiguous()
    for key in sorted(non_tensor_keys):
        batch[key] = np.fromiter((feature.get(key) for feature in features), dtype=object, count=len(features))
    return batch
