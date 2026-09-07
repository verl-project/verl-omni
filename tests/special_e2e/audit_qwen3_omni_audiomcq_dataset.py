#!/usr/bin/env python3
"""One-pass, resumable audit of AudioMCQ parquet audio through Qwen3-Omni preprocessing.

The input parquets and media are read-only.  All incremental output is written
to a caller-selected node-local directory so a large audit does not generate
high-frequency writes on shared storage.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def load_rows(data_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_audio_paths: set[str] = set()
    for data_file in data_files:
        table = pq.read_table(data_file)
        split = data_file.stem
        for row_index, row in enumerate(table.to_pylist()):
            audio_paths = row.get("audios") or []
            if len(audio_paths) != 1:
                raise ValueError(f"{data_file} row {row_index} has {len(audio_paths)} audio paths; expected one")
            audio_path = os.fspath(audio_paths[0])
            if audio_path in seen_audio_paths:
                raise ValueError(f"duplicate audio path across input parquets: {audio_path}")
            seen_audio_paths.add(audio_path)
            rows.append(
                {
                    "audio_path": audio_path,
                    "data_file": str(data_file),
                    "split": split,
                    "row_index": row_index,
                    "prompt": row["prompt"],
                    "extra_info": row.get("extra_info") or {},
                }
            )
    # Keep directory traversal stable and reduce random metadata access.
    rows.sort(key=lambda row: row["audio_path"])
    return rows


def materialize_audio_message(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(row["prompt"])
    audio_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content_list = []
        for segment in filter(None, re.split(r"(<audio>)", content)):
            if segment == "<audio>":
                if audio_count:
                    raise ValueError("more than one <audio> placeholder")
                content_list.append({"type": "audio", "audio": row["audio_path"]})
                audio_count += 1
            else:
                content_list.append({"type": "text", "text": segment})
        message["content"] = content_list
    if audio_count != 1:
        raise ValueError(f"expected one <audio> placeholder, found {audio_count}")
    return messages


def qwen3_omni_vllm_audio_token_count(audio_feature_frames: int) -> int:
    """Mirror vLLM's Qwen3-Omni audio placeholder expansion."""
    input_lengths_leave = audio_feature_frames % 100
    feature_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (audio_feature_frames // 100) * 13


def audit_row(
    processor,
    row: dict[str, Any],
    *,
    max_prompt_length: int,
    max_response_length: int,
    max_model_length: int,
) -> dict[str, Any]:
    messages = materialize_audio_message(row)
    raw_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    audio_config = OmegaConf.create({"audio_sampling_rate": sampling_rate, "audio_max_items": 1})
    waveform = RLHFDataset._load_local_audio(row["audio_path"], audio_config)
    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[raw_prompt],
        audio=[waveform],
        mm_processor_kwargs={"sampling_rate": sampling_rate},
    )
    input_features = model_inputs.get("input_features")
    feature_attention_mask = model_inputs.get("feature_attention_mask")
    if input_features is None or feature_attention_mask is None:
        raise RuntimeError("processor dropped input_features or feature_attention_mask")
    feature_lengths = feature_attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()
    if len(feature_lengths) != 1 or int(feature_lengths[0]) <= 0:
        raise RuntimeError(f"invalid audio feature lengths: {feature_lengths}")
    prompt_length = len(normalize_token_ids(model_inputs["input_ids"]))
    audio_feature_frames = int(feature_lengths[0])
    vllm_audio_tokens = qwen3_omni_vllm_audio_token_count(audio_feature_frames)
    # The processor-side prompt contains one audio placeholder. vLLM replaces
    # it with the post-convolution audio token sequence before enforcing
    # max_model_len.
    vllm_expanded_prompt_tokens = prompt_length - 1 + vllm_audio_tokens
    return {
        "status": "valid",
        "target_sampling_rate": sampling_rate,
        "audio_sample_count": int(waveform.size),
        "audio_duration_seconds": float(waveform.size / sampling_rate),
        "audio_sha256": hashlib.sha256(np.ascontiguousarray(waveform).tobytes()).hexdigest(),
        "prompt_tokens": prompt_length,
        "audio_feature_frames": audio_feature_frames,
        "vllm_audio_tokens": vllm_audio_tokens,
        "vllm_expanded_prompt_tokens": vllm_expanded_prompt_tokens,
        "input_features_shape": list(input_features.shape),
        "over_prompt_budget": prompt_length > max_prompt_length,
        "over_model_budget": vllm_expanded_prompt_tokens + max_response_length > max_model_length,
    }


def read_completed(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    completed: set[str] = set()
    records: list[dict[str, Any]] = []
    if not path.exists():
        return completed, records
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] ignoring incomplete checkpoint line {line_number}", flush=True)
                break
            completed.add(record["audio_path"])
            records.append(record)
    return completed, records


def percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), quantile))


def build_summary(
    records: list[dict[str, Any]],
    *,
    expected: int,
    elapsed_seconds: float,
    max_response_length: int,
    max_model_length: int,
) -> dict[str, Any]:
    statuses = Counter(record["status"] for record in records)
    valid = [record for record in records if record["status"] == "valid"]
    prompt_lengths = [record["prompt_tokens"] for record in valid]
    feature_lengths = [record["audio_feature_frames"] for record in valid]
    vllm_prompt_lengths = [
        record.get(
            "vllm_expanded_prompt_tokens",
            record["prompt_tokens"] - 1 + qwen3_omni_vllm_audio_token_count(record["audio_feature_frames"]),
        )
        for record in valid
    ]
    durations = [record["audio_duration_seconds"] for record in valid]
    errors = Counter(record.get("error_type", "") for record in records if record["status"] != "valid")
    return {
        "complete": len(records) == expected,
        "expected_rows": expected,
        "audited_rows": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "error_type_counts": dict(sorted(errors.items())),
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": {
            "min": min(prompt_lengths) if prompt_lengths else None,
            "p50": percentile(prompt_lengths, 50),
            "p95": percentile(prompt_lengths, 95),
            "p99": percentile(prompt_lengths, 99),
            "max": max(prompt_lengths) if prompt_lengths else None,
        },
        "audio_feature_frames": {
            "min": min(feature_lengths) if feature_lengths else None,
            "p50": percentile(feature_lengths, 50),
            "p95": percentile(feature_lengths, 95),
            "p99": percentile(feature_lengths, 99),
            "max": max(feature_lengths) if feature_lengths else None,
        },
        "vllm_expanded_prompt_tokens": {
            "min": min(vllm_prompt_lengths) if vllm_prompt_lengths else None,
            "p50": percentile(vllm_prompt_lengths, 50),
            "p95": percentile(vllm_prompt_lengths, 95),
            "p99": percentile(vllm_prompt_lengths, 99),
            "max": max(vllm_prompt_lengths) if vllm_prompt_lengths else None,
        },
        "audio_duration_seconds": {
            "min": min(durations) if durations else None,
            "p50": percentile(durations, 50),
            "p95": percentile(durations, 95),
            "p99": percentile(durations, 99),
            "max": max(durations) if durations else None,
        },
        "over_prompt_budget": sum(bool(record.get("over_prompt_budget")) for record in valid),
        "over_model_budget": sum(
            prompt_length + max_response_length > max_model_length for prompt_length in vllm_prompt_lengths
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    torch.set_num_threads(1)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (output_dir / "audit.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another AudioMCQ audit is already using {output_dir}") from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()
    checkpoint_path = output_dir / "assets.partial.jsonl"
    final_path = output_dir / "assets.jsonl"
    summary_path = output_dir / "summary.json"
    invalid_path = output_dir / "invalid_assets.jsonl"
    if final_path.exists():
        raise FileExistsError(f"completed audit already exists: {final_path}")

    rows = load_rows([Path(path).expanduser().resolve(strict=True) for path in args.data_file])
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        rows = rows[: args.max_rows]
    completed, records = read_completed(checkpoint_path)
    unexpected = completed - {row["audio_path"] for row in rows}
    if unexpected:
        raise ValueError(f"checkpoint contains {len(unexpected)} paths absent from current inputs")
    print(f"[info] rows={len(rows)} resumed={len(records)} output={output_dir}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    started = time.monotonic()
    pending_since_flush = 0
    with checkpoint_path.open("a", encoding="utf-8") as output:
        for row in rows:
            if row["audio_path"] in completed:
                continue
            base = {
                "audio_path": row["audio_path"],
                "data_file": row["data_file"],
                "split": row["split"],
                "row_index": row["row_index"],
                "source_index": row["extra_info"].get("index"),
                "source_dataset": row["extra_info"].get("source_dataset"),
                "source_id": row["extra_info"].get("source_id"),
            }
            try:
                result = {
                    **base,
                    **audit_row(
                        processor,
                        row,
                        max_prompt_length=args.max_prompt_length,
                        max_response_length=args.max_response_length,
                        max_model_length=args.max_model_length,
                    ),
                }
            except Exception as error:  # Keep scanning so one bad asset cannot hide later defects.
                result = {
                    **base,
                    "status": "invalid",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            records.append(result)
            pending_since_flush += 1
            if pending_since_flush >= args.progress_every:
                output.flush()
                pending_since_flush = 0
                invalid_count = sum(record["status"] != "valid" for record in records)
                print(f"[progress] audited={len(records)}/{len(rows)} invalid={invalid_count}", flush=True)
        output.flush()

    summary = build_summary(
        records,
        expected=len(rows),
        elapsed_seconds=time.monotonic() - started,
        max_response_length=args.max_response_length,
        max_model_length=args.max_model_length,
    )
    if not summary["complete"]:
        atomic_json_dump(summary, summary_path)
        raise RuntimeError(f"audit incomplete: {summary}")
    with invalid_path.open("w", encoding="utf-8") as output:
        for record in records:
            if record["status"] != "valid":
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    checkpoint_path.replace(final_path)
    atomic_json_dump(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
