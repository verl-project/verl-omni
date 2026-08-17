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
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torch.utils.data import Dataset, Sampler
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

from verl_omni.utils.dataset.qwen3_omni_transform import IGNORE_INDEX, process_qwen3_omni_sample

__all__ = [
    "ModalityGroupedBatchSampler",
    "OfflineMLLMDPODataset",
    "get_batch_modality",
    "offline_mllm_dpo_collate_fn",
]

_MEDIA_TOKEN_PATTERN = re.compile(r"<(image|video|audio)>")
_SUPPORTED_MODALITIES = {"audio", "image", "video"}
_TEXT_MODEL_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "labels",
    "loss_mask",
    "image_mask",
    "video_mask",
    "audio_mask",
}


def _normalise_data_files(data_files: str | os.PathLike | Sequence[str] | ListConfig) -> list[str | os.PathLike]:
    if isinstance(data_files, str | os.PathLike):
        return [data_files]
    if isinstance(data_files, ListConfig):
        data_files = OmegaConf.to_container(data_files, resolve=True)
    return list(data_files)


def _read_dataframe(data_files: str | Sequence[str] | ListConfig) -> pd.DataFrame:
    paths = _normalise_data_files(data_files)
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


def _first_batch_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, NonTensorStack):
        return _first_batch_value(value[0]) if len(value) else None
    if isinstance(value, NonTensorData):
        return _first_batch_value(value.data)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return _first_batch_value(value.item() if value.ndim == 0 else value[0])
    if isinstance(value, list | tuple):
        return _first_batch_value(value[0]) if value else None
    return value


def get_batch_modality(batch_dict: dict[str, Any]) -> str:
    """Return the first sample's ``audio`` / ``image`` / ``video`` modality."""

    extra_info = _first_batch_value(batch_dict.get("extra_info"))
    if not isinstance(extra_info, dict):
        raise ValueError(f"Offline MLLM DPO batch extra_info must contain modality, got {type(extra_info).__name__}.")
    modality = extra_info.get("modality")
    if modality not in _SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported offline MLLM DPO batch modality: {modality!r}.")
    return str(modality)


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
    branch = {"conversations": conversations}
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


def _branch_value(value: Any, branch_index: int) -> Any:
    if isinstance(value, dict):
        return {key: _branch_value(item, branch_index) for key, item in value.items()}
    if isinstance(value, list | tuple) and len(value) == 2:
        return value[branch_index]
    return value


def _is_pair_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_is_pair_value(item) for item in value.values())
    return isinstance(value, list | tuple) and len(value) == 2


def _is_paired_feature(feature: dict[str, Any]) -> bool:
    return any(_is_pair_value(value) for value in feature.values())


def _expand_paired_features(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for feature in features:
        if not _is_paired_feature(feature):
            expanded.append(dict(feature))
            continue
        for branch_index in range(2):
            expanded.append({key: _branch_value(value, branch_index) for key, value in feature.items()})
    return expanded


def _first_present(values: Sequence[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _collate_non_tensor_values(values: Sequence[Any]):
    return np.fromiter((value for value in values), dtype=object, count=len(values))


def _collate_grid_thw_values(key: str, values: Sequence[torch.Tensor | None]) -> torch.Tensor:
    present = [value for value in values if isinstance(value, torch.Tensor)]
    if not present:
        raise ValueError(f"Cannot collate grid key {key!r} without any tensor values.")
    if any(value is None for value in values):
        raise ValueError(f"Cannot collate missing grid values for key {key!r}.")

    grids = []
    for value in present:
        if value.numel() % 3 != 0:
            raise ValueError(f"{key} must contain triplets of (t, h, w), got shape {tuple(value.shape)}.")
        grids.append(value.reshape(-1, 3))
    return torch.cat(grids, dim=0).contiguous()


def _collate_no_padding_tensor_values(key: str, values: Sequence[torch.Tensor | None]) -> torch.Tensor:
    present = [value for value in values if isinstance(value, torch.Tensor)]
    if not present:
        raise ValueError(f"Cannot collate tensor key {key!r} without any tensor values.")
    if any(value is None for value in values):
        raise ValueError(f"Cannot collate missing tensor values for no-padding key {key!r}.")

    tensors = present
    if key in {"image_grid_thw", "video_grid_thw"}:
        return _collate_grid_thw_values(key, tensors)
    if tensors[0].dim() >= 2 and key == "position_ids":
        values_tensor = torch.cat(tensors, dim=-1)
        lengths = torch.tensor([tensor.shape[-1] for tensor in tensors], dtype=torch.long)
        offsets = torch.zeros(len(tensors) + 1, dtype=torch.long)
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        nested = torch.nested.nested_tensor_from_jagged(values_tensor, offsets=offsets)
        nested._ragged_idx = tensors[0].dim()
        return nested
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def _collate_no_padding_values(key: str, values: Sequence[Any]) -> Any:
    first = _first_present(values)
    if isinstance(first, torch.Tensor):
        return _collate_no_padding_tensor_values(key, values)
    return torch.stack([NonTensorData(value) for value in values], dim=0)


def _pad_mode_value(pad_mode: DatasetPadMode | str) -> str:
    return str(getattr(pad_mode, "value", pad_mode))


def _infer_collate_pad_mode(
    features: Sequence[dict[str, Any]], pad_mode: DatasetPadMode | str | None
) -> DatasetPadMode:
    if pad_mode is not None:
        return DatasetPadMode(pad_mode)

    feature_pad_modes = {
        _pad_mode_value(feature["pad_mode"]) for feature in features if feature.get("pad_mode") is not None
    }
    if len(feature_pad_modes) > 1:
        raise ValueError(f"Offline MLLM DPO batch contains mixed pad modes: {sorted(feature_pad_modes)}")
    if feature_pad_modes:
        return DatasetPadMode(feature_pad_modes.pop())
    return DatasetPadMode.NO_PADDING


def _normalise_multi_modal_sample(multi_modal_inputs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not multi_modal_inputs:
        return None
    sample_inputs: dict[str, Any] = {}
    for key, value in multi_modal_inputs.items():
        if value is None:
            continue
        if key in {"image_grid_thw", "video_grid_thw"} and isinstance(value, torch.Tensor):
            if value.numel() % 3 != 0:
                raise ValueError(f"{key} must contain triplets of (t, h, w), got shape {tuple(value.shape)}.")
            value = value.reshape(-1, 3).contiguous()
        sample_inputs[key] = value
    return sample_inputs or None


def _collate_multi_modal_inputs(features: Sequence[dict[str, Any]], *, no_padding: bool) -> list[dict[str, Any] | None]:
    del no_padding
    return [_normalise_multi_modal_sample(feature.get("multi_modal_inputs")) for feature in features]


def _collate_no_padding_batch(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch = SFTTensorCollator(pad_mode=DatasetPadMode.NO_PADDING)(features)
    if "labels" in batch and "loss_mask" not in batch:
        batch["loss_mask"] = batch["labels"].ne(IGNORE_INDEX)

    multi_modal_inputs = _collate_multi_modal_inputs(features, no_padding=True)
    if any(value is not None for value in multi_modal_inputs):
        batch["multi_modal_inputs"] = multi_modal_inputs
    return batch


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
            kwargs = {key: value for key, value in kwargs.items() if not (isinstance(value, list) and len(value) == 0)}
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


def _normalise_modality(value: Any, default: str = "unknown") -> str:
    text = str(value or default).lower()
    if "image" in text:
        return "image"
    if "video" in text:
        return "video"
    if "audio" in text:
        return "audio"
    return default


def _row_modality(row: dict[str, Any], default: str = "unknown") -> str:
    extra_info = _as_python(row.get("extra_info", {}))
    if isinstance(extra_info, dict) and extra_info.get("modality"):
        return _normalise_modality(extra_info["modality"], default)
    for key, modality in (("images", "image"), ("videos", "video"), ("audios", "audio")):
        if _normalise_media_list(row.get(key)):
            return modality
    return _normalise_modality(row.get("data_source"), default)


def _balanced_modality_sample_indices(
    dataframe: pd.DataFrame,
    max_samples: int,
    *,
    shuffle: bool,
    seed: int | None,
    default_modality: str,
) -> list[int]:
    rows = dataframe.to_dict(orient="records")
    indices_by_modality: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_modality[_row_modality(row, default_modality)].append(index)

    modalities = sorted(indices_by_modality)
    if not modalities:
        return []

    target = min(max_samples, len(dataframe))
    base_quota, remainder = divmod(target, len(modalities))
    quotas = {
        modality: min(base_quota + (1 if offset < remainder else 0), len(indices_by_modality[modality]))
        for offset, modality in enumerate(modalities)
    }

    leftover = target - sum(quotas.values())
    while leftover > 0:
        progressed = False
        for modality in modalities:
            if quotas[modality] >= len(indices_by_modality[modality]):
                continue
            quotas[modality] += 1
            leftover -= 1
            progressed = True
            if leftover == 0:
                break
        if not progressed:
            break

    rng = np.random.default_rng(seed) if shuffle else None
    selected: list[int] = []
    for modality in modalities:
        indices = indices_by_modality[modality]
        quota = quotas[modality]
        if quota <= 0:
            continue
        if shuffle:
            assert rng is not None
            selected.extend(rng.choice(indices, size=quota, replace=False).tolist())
        else:
            selected.extend(indices[:quota])
    return selected


class OfflineMLLMDPODataset(MultiTurnSFTDataset):
    """Build Qwen3-Omni offline DPO samples from Omni-Preference style rows.

    The dataset reads parquet/json/jsonl files containing a multimodal prompt,
    a preferred response, and a rejected response. Each row is converted into
    paired chosen/rejected model inputs with aligned image, video, or audio
    features, plus sample metadata needed by the offline DPO trainer.

    Args:
        data_files: Path or sequence of paths to parquet, json, or jsonl files.
        tokenizer: Unused tokenizer argument kept for compatibility with the
            common dataset factory signature.
        processor: Multimodal processor used by the Qwen3-Omni transform.
        config: Dataset config containing column names, multimodal transform
            kwargs, and source metadata.
        max_samples: Optional positive limit on the number of rows to load.

    Returns:
        A dataset whose ``__getitem__`` returns a dictionary containing paired
        chosen/rejected branch values and metadata, including
        ``sample_level_scores``, ``data_source``, ``reward_model``,
        ``modality``, and ``extra_info``. The collator expands those pairs into
        adjacent ``2 * B`` rows.
    """

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        if config is None:
            raise ValueError("OfflineMLLMDPODataset requires a data config.")
        if processor is None:
            raise ValueError("OfflineMLLMDPODataset requires a multimodal processor.")

        self.config = config
        self.processor = _prepare_qwen3_omni_processor(processor)
        self.tokenizer = tokenizer if tokenizer is not None else getattr(processor, "tokenizer", processor)
        self.pad_mode = DatasetPadMode(config.get("pad_mode", DatasetPadMode.RIGHT))

        self.truncation = config.get("truncation", "error")
        if self.truncation not in ("error", "left", "right"):
            raise ValueError(f"Unknown truncation method {self.truncation}")
        self.max_length = int(config.get("max_length", 1024))
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.balance_max_samples_by_modality = bool(config.get("balance_max_samples_by_modality", False))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")
        self.win_score_key = config.get("win_score_key", "win_score")
        self.lose_score_key = config.get("lose_score_key", "lose_score")
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

        if self.transform_kwargs.get("use_audio_in_video"):
            raise ValueError(
                "use_audio_in_video=True is not supported yet for Qwen3-Omni offline DPO preprocessing. "
                "Leave use_audio_in_video unset or set it to false."
            )

        base_transform = config.get("base_transform", "qwen3_omni_moe")
        if base_transform not in {"qwen3_omni_moe", "qwen2_5_omni"}:
            raise ValueError(
                f"Unsupported base_transform {base_transform!r}. Expected one of: 'qwen3_omni_moe', 'qwen2_5_omni'."
            )
        self.base_transform = process_qwen3_omni_sample

        self.parquet_files = _normalise_data_files(data_files)
        self._read_files_and_process()

    def _read_files_and_process(self):
        self.dataframe = _read_dataframe(self.parquet_files)
        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples is not None and self.max_samples > 0 and self.max_samples < total:
            if self.balance_max_samples_by_modality:
                indices = _balanced_modality_sample_indices(
                    self.dataframe,
                    int(self.max_samples),
                    shuffle=bool(self.shuffle),
                    seed=self.seed,
                    default_modality=self.data_source,
                )
            elif self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            selected_indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            self.dataframe = self.dataframe.iloc[selected_indices]
            sampling_mode = "balanced modality" if self.balance_max_samples_by_modality else "limited"
            print(f"selected {len(self.dataframe)} {sampling_mode} samples out of {total}")

        required = {self.prompt_key, self.chosen_key, self.rejected_key}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Offline MLLM DPO data is missing required columns: {sorted(missing)}")
        self.modalities = [_row_modality(row, self.data_source) for row in self.dataframe.to_dict(orient="records")]

    @staticmethod
    def _split_model_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        multi_modal_inputs: dict[str, Any] = {}
        for key, value in model_inputs.items():
            if key == "position_ids" and isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[0] == 1:
                value = value.squeeze(0).contiguous()
            if key == "position_ids" and isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[-2] == 1:
                value = value.squeeze(-2).contiguous()
            if key in _TEXT_MODEL_KEYS:
                res[key] = value
            else:
                multi_modal_inputs[key] = value

        if "labels" in res and "loss_mask" not in res:
            res["loss_mask"] = res["labels"].ne(IGNORE_INDEX)
        if multi_modal_inputs:
            res["multi_modal_inputs"] = multi_modal_inputs
        return res

    @staticmethod
    def _truncate_sequence_value(key: str, value: Any, max_length: int, truncation: str) -> Any:
        del key
        if not isinstance(value, torch.Tensor) or value.shape[-1] <= max_length:
            return value
        if truncation == "left":
            return value[..., -max_length:]
        return value[..., :max_length]

    def _prepare_branch_inputs(self, branch: dict[str, Any]) -> dict[str, Any]:
        input_ids = branch.get("input_ids")
        res = dict(branch)
        if not isinstance(input_ids, torch.Tensor):
            res["pad_mode"] = self.pad_mode.value
            return res

        sequence_length = input_ids.shape[-1]
        if sequence_length > self.max_length and self.truncation == "error":
            raise ValueError(f"sequence_length={sequence_length} is larger than self.max_length={self.max_length}")
        if sequence_length > self.max_length:
            for key in _TEXT_MODEL_KEYS:
                if key in res:
                    res[key] = self._truncate_sequence_value(key, res[key], self.max_length, self.truncation)
        return res

    def _build_preference_branch(self, sample: dict[str, Any], answer: Any) -> dict[str, Any]:
        return _build_preference_branch(sample, answer)

    def _process_preference_branch(self, sample: dict[str, Any], answer: Any) -> dict[str, Any]:
        branch_sample = self._build_preference_branch(sample, answer)
        model_inputs = self.base_transform(branch_sample, processor=self.processor, **self.transform_kwargs)[0]
        return self._prepare_branch_inputs(self._split_model_inputs(model_inputs))

    @staticmethod
    def _pair_branch_values(chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
        paired: dict[str, Any] = {}
        for key in chosen.keys() | rejected.keys():
            if key == "multi_modal_inputs":
                mm_keys = set(chosen.get(key, {}).keys()) | set(rejected.get(key, {}).keys())
                paired_mm = {}
                for mm_key in mm_keys:
                    paired_mm[mm_key] = [chosen.get(key, {}).get(mm_key), rejected.get(key, {}).get(mm_key)]
                if paired_mm:
                    paired[key] = paired_mm
            else:
                paired[key] = [chosen.get(key), rejected.get(key)]
        return paired

    def get_modality(self, item: int) -> str:
        return self.modalities[item]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe.iloc[item].to_dict()
        sample = {
            "prompt": row[self.prompt_key],
            "chosen": row[self.chosen_key],
            "rejected": row[self.rejected_key],
            "images": row.get("images"),
            "videos": row.get("videos"),
            "audios": row.get("audios"),
        }
        chosen = self._process_preference_branch(sample, row[self.chosen_key])
        rejected = self._process_preference_branch(sample, row[self.rejected_key])
        transformed = self._pair_branch_values(chosen, rejected)
        uid = str(row.get("uid") or uuid.uuid4())
        transformed["uid"] = [uid, uid]
        transformed["sample_level_scores"] = [
            torch.tensor([float(row.get(self.win_score_key, 1.0))], dtype=torch.float32),
            torch.tensor([float(row.get(self.lose_score_key, 0.0))], dtype=torch.float32),
        ]
        data_source = row.get("data_source") or self.data_source
        transformed["data_source"] = [data_source, data_source]
        reward_model = row.get("reward_model", {"style": "model", "ground_truth": row[self.chosen_key]})
        transformed["reward_model"] = [reward_model, reward_model]
        modality = self.get_modality(item)
        transformed["modality"] = [modality, modality]
        extra_info = _as_python(row.get("extra_info", {"index": int(item)}))
        if isinstance(extra_info, dict):
            extra_info = {**extra_info, "modality": modality}
        transformed["extra_info"] = [extra_info, extra_info]
        transformed["is_chosen"] = [True, False]
        return transformed


class ModalityGroupedBatchSampler(Sampler[int]):
    """Build same-modality batches for regular DataLoader sampling.

    ``StatefulDataLoader`` is configured with ``sampler=`` and ``batch_size=``,
    not ``batch_sampler=``. This sampler therefore yields individual indices,
    ordered as contiguous same-modality chunks of ``batch_size``. Each chunk
    samples a modality uniformly by default, or according to
    ``modality_sample_weights`` when provided, then samples rows from that
    modality with replacement by default. Validation can disable replacement to
    visit each row once while preserving same-modality batches.

    Args:
        data_source: Dataset that provides ``get_modality(index)``.
        dataset: Alias for ``data_source`` kept for compatibility with dataset
            factory arguments.
        data_config: Optional config used to infer ``batch_size`` from
            ``gen_batch_size`` or ``train_batch_size``.
        batch_size: Number of samples in each same-modality chunk.
        shuffle: Kept for compatibility. Ignored when ``replacement`` is false
            because validation should be deterministic.
        drop_last: Whether to drop incomplete same-modality chunks whose size
            is smaller than ``batch_size``. For weighted training batches this
            controls how many chunks are generated; for sequential validation
            batches this drops trailing partial chunks per modality.
        seed: Base random seed used for modality and row sampling.
        modality_sample_weights: Optional per-modality sampling weights.
        num_batches: Optional explicit number of chunks to generate per epoch.
        replacement: Whether to sample rows with replacement.

    Returns:
        A sampler whose iterator yields dataset indices arranged so each regular
        DataLoader batch contains samples from a single modality.
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
        replacement: bool = True,
    ):
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
        self.replacement = bool(replacement)
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

    def _build_sequential_batches(
        self,
        indices_by_modality: dict[str, list[int]],
        generator: torch.Generator,
    ) -> list[list[int]]:
        batches: list[list[int]] = []
        del generator
        for modality in sorted(indices_by_modality):
            indices = list(indices_by_modality[modality])
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if not batch:
                    continue
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        return batches

    def _build_batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices_by_modality = self._indices_by_modality()
        if not self.replacement:
            return self._build_sequential_batches(indices_by_modality, generator)
        return self._build_weighted_batches(indices_by_modality, generator)

    def __iter__(self):
        for batch in self._build_batches():
            yield from batch

    def __len__(self) -> int:
        return self._length


def _offline_mllm_dpo_collate_fn(features, pad_mode: DatasetPadMode | str | None = None):
    """Collate offline MLLM DPO samples into adjacent chosen/rejected rows.

    Each logical dataset row is a preference pair. The collator expands those
    pairs into a default batch layout of ``[chosen0, rejected0, chosen1,
    rejected1, ...]``, so tensor and non-tensor batch dimensions are ``2 * B``.
    With ``pad_mode=no_padding``, tensor values are represented as jagged nested
    tensors and non-tensor values follow ``SFTTensorCollator`` by stacking
    ``NonTensorData`` wrappers.

    Args:
        features (list[dict[str, Any]]): List of sample dictionaries produced by
            :class:`OfflineMLLMDPODataset`.

    Returns:
        dict[str, Any]: A batched dictionary where dense tensor keys map to
            :class:`torch.Tensor`. For ``pad_mode=no_padding``, tensor keys map
            to jagged nested tensors and non-tensor keys map to stacked
            ``NonTensorData`` wrappers.
    """
    if not features:
        return {}

    pad_mode = _infer_collate_pad_mode(features, pad_mode)
    features = _expand_paired_features(features)
    no_padding = pad_mode == DatasetPadMode.NO_PADDING

    modalities = {feature.get("modality") for feature in features}
    if len(modalities) != 1:
        raise ValueError(f"Offline MLLM DPO batches must contain a single modality, got {sorted(modalities)}")
    if no_padding:
        return _collate_no_padding_batch(features)

    tensor_keys = {
        key
        for feature in features
        for key, value in feature.items()
        if key != "multi_modal_inputs" and isinstance(value, torch.Tensor)
    }
    non_tensor_keys = {
        key
        for feature in features
        for key, value in feature.items()
        if key != "multi_modal_inputs" and not isinstance(value, torch.Tensor)
    }

    batch: dict[str, Any] = {}
    for key in sorted(tensor_keys):
        values = [feature.get(key) for feature in features]
        batch[key] = _collate_no_padding_values(key, values) if no_padding else _collate_tensor_values(key, values)
        if not no_padding and key == "position_ids" and batch[key].ndim == 4 and batch[key].shape[1] == 1:
            batch[key] = batch[key].squeeze(1).contiguous()
        if not no_padding and key == "position_ids" and batch[key].ndim == 4 and batch[key].shape[2] == 1:
            batch[key] = batch[key].squeeze(2).contiguous()
    if "labels" in batch and "loss_mask" not in batch:
        batch["loss_mask"] = batch["labels"].ne(IGNORE_INDEX)
    for key in sorted(non_tensor_keys):
        values = [feature.get(key) for feature in features]
        batch[key] = _collate_no_padding_values(key, values) if no_padding else _collate_non_tensor_values(values)

    multi_modal_inputs = _collate_multi_modal_inputs(features, no_padding=no_padding)
    if any(value is not None for value in multi_modal_inputs):
        batch["multi_modal_inputs"] = multi_modal_inputs
    return batch


def offline_mllm_dpo_collate_fn(features, pad_mode: DatasetPadMode | str | None = None):
    """Collate offline MLLM DPO samples, inferring right-padding vs no-padding from ``pad_mode``."""
    return _offline_mllm_dpo_collate_fn(features, pad_mode=pad_mode)
