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
"""Verify MiniCPM offline DPO parquet rows through real remote-code processor.

Loads ``processing_minicpmo.MiniCPMOProcessor`` via ``AutoProcessor``,
constructs :class:`OfflineMLLMDPODataset` batches from real parquet files,
and prints a readable summary for each collated batch.

Each batch summary separates **training** tensors (``input_ids``, ``labels``, ...)
from **processor** multimodal fields (``pixel_values``, ``audio_bounds``, ...).
It always prints ``label_alignment`` comparing parquet ``chosen``/``rejected``
text against label tokens, decoding the sequence with ``image_bound`` /
``audio_bounds`` spans replaced by ``[IMAGE#N]`` / ``[AUDIO#N]`` tags.

Example::

    python examples/dpo_trainer/minicpm/verify_minicpm_dataset.py \\
        --model-path openbmb/MiniCPM-o-4_5 \\
        --train-files ~/Omni-Preference/parquet_dpo/image/train.parquet \\
                      ~/Omni-Preference/parquet_dpo/audio/train.parquet \\
        --num-batches 4 \\
        --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

from verl_omni.utils.dataset.minicpm_transform import _MINICPM_AUDIO_SLOT, _MINICPM_IMAGE_SLOT, IGNORE_INDEX
from verl_omni.utils.dataset.offline_mllm_dpo_dataset import (
    ModalityGroupedBatchSampler,
    OfflineMLLMDPODataset,
    _answer_text,
    get_batch_modality,
    offline_mllm_dpo_collate_fn,
)

_DEFAULT_MM_CONFIGS = {
    "image_min_pixels": 3136,
    "image_max_pixels": 602112,
    "max_ratio": 200,
    "sample_rate": 16000,
    "max_slice_nums": 1,
}

# Keys produced by MiniCPMOProcessor (stored under multi_modal_inputs after split).
_PROCESSOR_MM_KEYS = frozenset(
    {
        "pixel_values",
        "image_sizes",
        "tgt_sizes",
        "image_bound",
        "audio_bounds",
        "spk_bounds",
        "audio_features",
        "audio_feature_lens",
    }
)

# Keys added by OfflineMLLMDPODataset / minicpm_transform for training & DPO.
_TRAINING_TENSOR_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "labels",
        "loss_mask",
        "image_mask",
        "video_mask",
        "audio_mask",
    }
)

_METADATA_KEYS = frozenset(
    {
        "uid",
        "modality",
        "extra_info",
        "data_source",
        "reward_model",
        "is_chosen",
        "sample_level_scores",
        "multi_modal_inputs",
    }
)

_PREVIEW_CHARS = 400


def _parse_train_files(values: Sequence[str]) -> list[str]:
    if not values:
        raise ValueError("At least one --train-files parquet path is required.")
    return [str(Path(path).expanduser()) for path in values]


def _load_processor(model_path: str, *, trust_remote_code: bool):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    processor_cls = type(processor).__name__
    if processor_cls != "MiniCPMOProcessor":
        print(
            f"Warning: expected MiniCPMOProcessor from remote code, got {processor_cls}.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"Loaded {processor_cls} from {model_path}", flush=True)
    return processor


def _load_tokenizer(model_path: str, processor, *, trust_remote_code: bool):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def _unwrap_non_tensor(value: Any) -> Any:
    if isinstance(value, NonTensorStack):
        return [_unwrap_non_tensor(item) for item in value]
    if isinstance(value, NonTensorData):
        return _unwrap_non_tensor(value.data)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_unwrap_non_tensor(item) for item in value.tolist()]
    return value


def _nested_lengths(value: torch.Tensor) -> list[int]:
    if not value.is_nested:
        return [int(value.shape[-1])]
    return [int(value[i].numel()) for i in range(value.size(0))]


def _summarize_tensor(name: str, value: torch.Tensor) -> dict[str, Any]:
    summary: dict[str, Any] = {"kind": "tensor", "dtype": str(value.dtype)}
    if value.is_nested:
        summary["layout"] = "nested/jagged"
        summary["batch_size"] = int(value.size(0))
        summary["seq_lengths"] = _nested_lengths(value)
        return summary

    summary["shape"] = list(value.shape)
    if value.numel() == 0:
        return summary

    if name in {"input_ids", "labels", "attention_mask", "loss_mask"} and value.ndim >= 1:
        flat = value.reshape(value.shape[0], -1) if value.ndim > 1 else value.unsqueeze(0)
        summary["seq_lengths"] = [int(row.numel()) for row in flat]
    if name == "labels" and value.numel() > 0:
        summary["supervised_tokens"] = int((value != IGNORE_INDEX).sum().item())
    return summary


def _summarize_mapping(name: str, mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _summarize_value(key, val) for key, val in sorted(mapping.items())}


def _summarize_value(name: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _summarize_tensor(name, value)
    if isinstance(value, dict):
        return _summarize_mapping(name, value)
    if isinstance(value, list | tuple):
        if not value:
            return []
        if all(isinstance(item, dict) for item in value):
            return [_summarize_mapping(f"{name}[{idx}]", item) for idx, item in enumerate(value)]
        return [_summarize_value(f"{name}[{idx}]", item) for idx, item in enumerate(value)]
    if isinstance(value, (str | int | float | bool)):
        return value
    return repr(value)


def _preview_text(text: str, limit: int = _PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_compare_text(text: str) -> str:
    return " ".join(str(text).split())


def _bound_tensors(value: Any) -> list[torch.Tensor]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def _placeholder_spans(multi_modal_inputs: dict[str, Any] | None) -> list[tuple[int, int, str]]:
    """Return token spans ``[start, end]`` (inclusive) replaced by placeholder tags when decoding."""

    if not multi_modal_inputs:
        return []

    spans: list[tuple[int, int, str]] = []
    for bound_idx, bound in enumerate(_bound_tensors(multi_modal_inputs.get("image_bound"))):
        if bound.numel() == 0:
            continue
        for start, end in bound.reshape(-1, 2).tolist():
            spans.append((int(start), int(end), f"IMAGE#{bound_idx}"))

    for bound_idx, bound in enumerate(_bound_tensors(multi_modal_inputs.get("audio_bounds"))):
        if bound.numel() == 0:
            continue
        for start, end in bound.reshape(-1, 2).tolist():
            spans.append((int(start), int(end), f"AUDIO#{bound_idx}"))

    for bound_idx, bound in enumerate(_bound_tensors(multi_modal_inputs.get("spk_bounds"))):
        if bound.numel() == 0:
            continue
        for start, end in bound.reshape(-1, 2).tolist():
            spans.append((int(start), int(end), f"SPK#{bound_idx}"))

    spans.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int, str]] = []
    for start, end, tag in spans:
        if end < start:
            continue
        if merged and start <= merged[-1][1] + 1:
            prev_start, prev_end, prev_tag = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), f"{prev_tag}+{tag}")
        else:
            merged.append((start, end, tag))
    return merged


def _decode_with_placeholder_spans(
    tokenizer,
    input_ids: torch.Tensor,
    spans: Sequence[tuple[int, int, str]],
) -> str:
    ids = input_ids.detach().cpu().reshape(-1).tolist()
    if not ids:
        return ""

    parts: list[str] = []
    cursor = 0
    for start, end, tag in spans:
        start = max(0, min(start, len(ids)))
        end = max(start, min(end, len(ids) - 1))
        if cursor < start:
            parts.append(tokenizer.decode(ids[cursor:start], skip_special_tokens=False))
        parts.append(f"[{tag}]")
        cursor = end + 1
    if cursor < len(ids):
        parts.append(tokenizer.decode(ids[cursor:], skip_special_tokens=False))
    return "".join(parts)


def _dataframe_row_from_extra_info(
    dataset: OfflineMLLMDPODataset,
    extra_info: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """Resolve a parquet row from collated ``extra_info`` without assuming positional ``index``."""

    debug: dict[str, Any] = {
        "extra_info_type": type(extra_info).__name__,
        "dataset_len": len(dataset.dataframe),
    }
    if not isinstance(extra_info, dict):
        debug["lookup_failure"] = "extra_info_is_not_dict"
        return None, debug

    debug["extra_info_keys"] = sorted(extra_info.keys())
    debug["index_candidates"] = {key: extra_info.get(key) for key in ("dataset_index", "index") if key in extra_info}

    dataframe = dataset.dataframe
    for key in ("dataset_index", "index"):
        raw_index = extra_info.get(key)
        if raw_index is None:
            continue
        index = int(raw_index)
        if 0 <= index < len(dataframe):
            debug["resolved_by"] = f"{key}->iloc"
            debug["resolved_index"] = index
            return dataframe.iloc[index], debug
        if index in dataframe.index:
            debug["resolved_by"] = f"{key}->loc"
            debug["resolved_index"] = index
            return dataframe.loc[index], debug
        debug.setdefault("out_of_bounds", []).append(
            {
                key: index,
                "dataframe_index_min": int(dataframe.index.min()),
                "dataframe_index_max": int(dataframe.index.max()),
            }
        )

    debug["lookup_failure"] = (
        "index_out_of_bounds: extra_info.index is often a global parquet sample id, "
        "not a positional row in the loaded dataframe; use dataset_index (injected by "
        "OfflineMLLMDPODataset.__getitem__) for lookup."
    )
    return None, debug


def _expected_branch_answer(
    dataset: OfflineMLLMDPODataset,
    extra_info: Any,
    *,
    is_chosen: bool,
    reward_model: Any = None,
) -> tuple[str | None, dict[str, Any]]:
    row, lookup_debug = _dataframe_row_from_extra_info(dataset, extra_info)
    key = dataset.chosen_key if is_chosen else dataset.rejected_key
    lookup_debug["answer_key"] = key

    if row is not None:
        text = _answer_text(row.get(key))
        if text:
            lookup_debug["answer_source"] = "dataframe"
            return text, lookup_debug
        lookup_debug["lookup_failure"] = f"dataframe row missing or empty {key!r}"

    if is_chosen and isinstance(reward_model, dict):
        text = _answer_text(reward_model.get("ground_truth"))
        if text:
            lookup_debug["answer_source"] = "reward_model.ground_truth"
            return text, lookup_debug

    if lookup_debug.get("lookup_failure") is None:
        lookup_debug["lookup_failure"] = f"could not resolve {key!r} from dataframe or reward_model"
    return None, lookup_debug


def _label_alignment_report(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    expected_answer: str | None,
    spans: Sequence[tuple[int, int, str]],
    lookup_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = input_ids.detach().cpu().reshape(-1)
    label_values = labels.detach().cpu().reshape(-1)
    supervised_mask = label_values.ne(IGNORE_INDEX)
    supervised_ids = ids[supervised_mask]

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_count = int((supervised_ids == unk_token_id).sum().item()) if unk_token_id is not None else 0
    supervised_count = int(supervised_ids.numel())

    decoded_labels = ""
    if supervised_count > 0:
        decoded_labels = tokenizer.decode(supervised_ids.tolist(), skip_special_tokens=False)

    report: dict[str, Any] = {
        "supervised_token_count": supervised_count,
        "supervised_unk_token_count": unk_count,
        "decoded_from_labels_preview": _preview_text(decoded_labels),
        "decoded_sequence_preview": _preview_text(_decode_with_placeholder_spans(tokenizer, ids, spans)),
        "placeholder_spans": [{"start": s, "end": e, "tag": t} for s, e, t in spans],
    }

    if expected_answer is None:
        report["expected_answer_preview"] = None
        report["alignment_status"] = "unknown_expected_answer"
        if lookup_debug:
            report["lookup_debug"] = lookup_debug
        return report

    expected_norm = _normalize_compare_text(expected_answer)
    decoded_norm = _normalize_compare_text(decoded_labels)
    report["expected_answer_preview"] = _preview_text(expected_answer)
    report["alignment_status"] = "ok"
    if supervised_count == 0:
        report["alignment_status"] = "no_supervised_tokens"
    elif unk_count == supervised_count:
        report["alignment_status"] = "all_unk_labels"
    elif unk_count > supervised_count // 2:
        report["alignment_status"] = "mostly_unk_labels"
    elif decoded_norm != expected_norm and expected_norm not in decoded_norm and decoded_norm not in expected_norm:
        report["alignment_status"] = "text_mismatch"

    expected_ids = tokenizer.encode(expected_answer, add_special_tokens=False)
    report["expected_token_count"] = len(expected_ids)
    report["label_token_ids_match_expected"] = supervised_ids.tolist() == expected_ids
    return report


def _batch_row_value(batch: dict[str, Any], key: str, row_idx: int) -> Any:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.is_nested:
        return value[row_idx]
    if isinstance(value, torch.Tensor):
        return value[row_idx]
    unwrapped = _unwrap_non_tensor(value)
    if isinstance(unwrapped, list | tuple):
        return unwrapped[row_idx]
    return unwrapped


def _decode_branch_text(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    spans: Sequence[tuple[int, int, str]] | None = None,
) -> dict[str, Any]:
    spans = list(spans or [])
    compact_text = _decode_with_placeholder_spans(tokenizer, input_ids, spans)
    result = {"decoded_sequence_preview": _preview_text(compact_text)}
    if labels is not None:
        label_ids = labels.detach().cpu().reshape(-1)
        supervised = label_ids[label_ids != IGNORE_INDEX]
        if supervised.numel() > 0:
            result["decoded_from_labels_preview"] = _preview_text(
                tokenizer.decode(supervised.tolist(), skip_special_tokens=False)
            )
    return result


def _print_batch_summary(
    batch_index: int,
    batch: dict[str, Any],
    *,
    dataset: OfflineMLLMDPODataset,
    tokenizer,
    show_input_text: bool,
) -> None:
    modality = get_batch_modality(batch)
    collated_rows = len(_unwrap_non_tensor(batch.get("extra_info", [])))
    logical_batch_size = collated_rows // 2

    header = {
        "batch_index": batch_index,
        "modality": modality,
        "logical_pairs": logical_batch_size,
        "collated_rows": collated_rows,
        "batch_keys": sorted(batch.keys()),
        "processor_image_slots_expected": modality == "image",
        "processor_audio_slots_expected": modality == "audio",
    }
    print(f"\n=== Batch {batch_index} ===", flush=True)
    print(json.dumps(header, indent=2, ensure_ascii=False), flush=True)

    training_summary: dict[str, Any] = {}
    other_tensor_summary: dict[str, Any] = {}
    for key, value in sorted(batch.items()):
        if key in _METADATA_KEYS or not isinstance(value, torch.Tensor):
            continue
        summary = _summarize_tensor(key, value)
        if key in _TRAINING_TENSOR_KEYS:
            training_summary[key] = summary
        else:
            other_tensor_summary[key] = summary

    print("training_tensor_fields:", flush=True)
    print(json.dumps(training_summary, indent=2, ensure_ascii=False), flush=True)
    if other_tensor_summary:
        print("other_tensor_fields:", flush=True)
        print(json.dumps(other_tensor_summary, indent=2, ensure_ascii=False), flush=True)

    extra_info = _unwrap_non_tensor(batch.get("extra_info"))
    data_source = _unwrap_non_tensor(batch.get("data_source"))
    is_chosen_flags = _unwrap_non_tensor(batch.get("is_chosen"))
    print("metadata:", flush=True)
    print(
        json.dumps(
            {
                "data_source": data_source,
                "extra_info": extra_info,
                "is_chosen": is_chosen_flags,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )

    multi_modal_inputs = batch.get("multi_modal_inputs")
    processor_summary: dict[str, Any] = {}
    if multi_modal_inputs is not None:
        per_row_keys: set[str] = set()
        for item in multi_modal_inputs:
            if isinstance(item, dict):
                per_row_keys.update(item.keys())
        processor_summary["present_processor_keys"] = sorted(per_row_keys & _PROCESSOR_MM_KEYS)
        processor_summary["rows"] = _summarize_value("multi_modal_inputs", multi_modal_inputs)

        print("processor_multi_modal_fields:", flush=True)
        print(json.dumps(processor_summary, indent=2, ensure_ascii=False), flush=True)

        if modality == "image":
            has_image_bound = any(
                isinstance(item, dict) and "image_bound" in item for item in multi_modal_inputs if item is not None
            )
            if not has_image_bound and "input_ids" in batch:
                print(
                    "warning: image batch missing image_bound in multi_modal_inputs; "
                    "check MiniCPM processor slot injection.",
                    flush=True,
                )
        if modality == "audio":
            has_audio_bounds = any(
                isinstance(item, dict) and "audio_bounds" in item for item in multi_modal_inputs if item is not None
            )
            if not has_audio_bounds and "input_ids" in batch:
                print(
                    "warning: audio batch missing audio_bounds in multi_modal_inputs; "
                    f"expected {_MINICPM_AUDIO_SLOT!r} slots with standalone audios.",
                    flush=True,
                )

    if "input_ids" in batch and "labels" in batch:
        input_ids = batch["input_ids"]
        num_rows = input_ids.size(0) if input_ids.is_nested else input_ids.shape[0]
        alignment_reports = []
        decoded_previews = []
        for row_idx in range(num_rows):
            row_input_ids = _batch_row_value(batch, "input_ids", row_idx)
            row_labels = _batch_row_value(batch, "labels", row_idx)
            row_extra_info = _batch_row_value(batch, "extra_info", row_idx)
            row_is_chosen = _batch_row_value(batch, "is_chosen", row_idx)
            row_reward_model = _batch_row_value(batch, "reward_model", row_idx)
            row_mm = None
            if isinstance(multi_modal_inputs, list | tuple) and row_idx < len(multi_modal_inputs):
                row_mm = multi_modal_inputs[row_idx]

            is_chosen = bool(row_is_chosen) if row_is_chosen is not None else row_idx % 2 == 0
            branch = "chosen" if is_chosen else "rejected"
            spans = _placeholder_spans(row_mm if isinstance(row_mm, dict) else None)
            expected_answer, lookup_debug = _expected_branch_answer(
                dataset,
                row_extra_info,
                is_chosen=is_chosen,
                reward_model=row_reward_model,
            )

            alignment_reports.append(
                {
                    "row_index": row_idx,
                    "branch": branch,
                    **_label_alignment_report(
                        tokenizer,
                        row_input_ids,
                        row_labels,
                        expected_answer=expected_answer,
                        spans=spans,
                        lookup_debug=lookup_debug,
                    ),
                }
            )
            if show_input_text:
                decoded_previews.append(
                    {
                        "row_index": row_idx,
                        "branch": branch,
                        **_decode_branch_text(
                            tokenizer,
                            row_input_ids,
                            row_labels,
                            spans=spans,
                        ),
                    }
                )

        print("label_alignment:", flush=True)
        print(json.dumps(alignment_reports, indent=2, ensure_ascii=False), flush=True)
        suspicious = [item for item in alignment_reports if item.get("alignment_status") != "ok"]
        if suspicious:
            print(
                f"warning: {len(suspicious)} row(s) with suspicious label alignment "
                f"(status != ok): {[item['alignment_status'] for item in suspicious]}",
                flush=True,
            )

        if show_input_text:
            print("decoded_previews:", flush=True)
            print(json.dumps(decoded_previews, indent=2, ensure_ascii=False), flush=True)


def _build_dataset(
    train_files: list[str],
    processor,
    tokenizer,
    *,
    max_length: int,
    truncation: str,
    mm_configs: dict[str, Any],
    max_samples: int,
):
    config = OmegaConf.create(
        {
            "base_transform": "minicpm",
            "pad_mode": "no_padding",
            "max_length": max_length,
            "truncation": truncation,
            "mm_configs": mm_configs,
        }
    )
    return OfflineMLLMDPODataset(
        train_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        max_samples=max_samples,
    )


def _iter_batches(
    dataset: OfflineMLLMDPODataset,
    *,
    batch_size: int,
    num_batches: int,
    seed: int,
    image_ratio: float,
    audio_ratio: float,
    pad_mode: str,
):
    sampler = ModalityGroupedBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        modality_sample_weights={"image": image_ratio, "audio": audio_ratio},
        num_batches=num_batches,
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda features: offline_mllm_dpo_collate_fn(features, pad_mode=pad_mode),
        num_workers=0,
    )
    yield from loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify MiniCPM offline DPO parquet batches.")
    default_model_path = os.environ.get("MINICPM_CACHE_DIR") or os.environ.get("MODEL_PATH") or "openbmb/MiniCPM-o-4_5"
    parser.add_argument(
        "--model-path",
        default=default_model_path,
        help="MiniCPM-o checkpoint or local snapshot containing processing_minicpmo.py. "
        "Defaults to MINICPM_CACHE_DIR, MODEL_PATH, or openbmb/MiniCPM-o-4_5.",
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        required=True,
        help="One or more parquet/json/jsonl files (same schema as OfflineMLLMDPODataset).",
    )
    parser.add_argument("--num-batches", type=int, default=2, help="Number of collated batches to print.")
    parser.add_argument("--batch-size", type=int, default=2, help="Preference pairs per batch.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncation", choices=("error", "left", "right"), default="right")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional cap on loaded rows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-ratio", type=float, default=1.0)
    parser.add_argument("--audio-ratio", type=float, default=1.0)
    parser.add_argument("--pad-mode", default="no_padding", choices=("no_padding", "right"))
    parser.add_argument(
        "--mm-configs",
        default="",
        help=f"JSON object overriding default mm_configs. Defaults: {json.dumps(_DEFAULT_MM_CONFIGS)}",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--show-input-text",
        action="store_true",
        help="Also print bound-aware decoded_previews (label_alignment is always printed).",
    )
    args = parser.parse_args(argv)

    train_files = _parse_train_files(args.train_files)
    mm_configs = dict(_DEFAULT_MM_CONFIGS)
    if args.mm_configs:
        mm_configs.update(json.loads(args.mm_configs))

    processor = _load_processor(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = _load_tokenizer(args.model_path, processor, trust_remote_code=args.trust_remote_code)

    print("train_files:", flush=True)
    for path in train_files:
        print(f"  - {path}", flush=True)
    print("mm_configs:", flush=True)
    print(json.dumps(mm_configs, indent=2, ensure_ascii=False), flush=True)
    print(
        "processor slot mapping:",
        json.dumps(
            {
                "image": _MINICPM_IMAGE_SLOT,
                "audio": _MINICPM_AUDIO_SLOT,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    dataset = _build_dataset(
        train_files,
        processor,
        tokenizer,
        max_length=args.max_length,
        truncation=args.truncation,
        mm_configs=mm_configs,
        max_samples=args.max_samples,
    )
    print(f"dataset_size={len(dataset)}", flush=True)

    for batch_index, batch in enumerate(
        _iter_batches(
            dataset,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed,
            image_ratio=args.image_ratio,
            audio_ratio=args.audio_ratio,
            pad_mode=args.pad_mode,
        )
    ):
        _print_batch_summary(
            batch_index,
            batch,
            dataset=dataset,
            tokenizer=tokenizer,
            show_input_text=args.show_input_text,
        )

    print(f"\nVerified {args.num_batches} batch(es) successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
