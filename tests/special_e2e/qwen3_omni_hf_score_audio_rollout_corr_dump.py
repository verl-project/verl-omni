#!/usr/bin/env python3
"""HF-score one audio-conditioned fixed response dumped by verl.

This is the AudioMCQ counterpart of the text-only rollout-correlation scorer.
It reconstructs the processor inputs from one parquet row, verifies that the
processor prompt IDs match the unpadded dumped prompt, and then compares HF
Thinker log-probabilities with the dumped Megatron actor/ref values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def _emit(record: dict, output_file: str | None) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _load_record(path: str, row: int) -> dict:
    matches = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "rollout_corr_sample" and int(record.get("row", -1)) == row:
                matches.append(record)
    if not matches:
        raise ValueError(f"No rollout_corr_sample row={row} found in {path}")
    # The later record includes ref_log_probs after reference scoring.
    return next((record for record in reversed(matches) if record.get("ref_log_probs") is not None), matches[-1])


def _load_example(path: str, row: int) -> dict:
    table = pq.read_table(path, columns=["prompt", "audios"])
    if not 0 <= row < table.num_rows:
        raise IndexError(f"dataset row {row} is outside [0, {table.num_rows})")
    example = table.slice(row, 1).to_pylist()[0]
    return {"prompt": example["prompt"], "audios": example.get("audios") or []}


def _materialize_audio_messages(example: dict) -> list[dict]:
    messages = copy.deepcopy(example["prompt"])
    audios = example["audios"]
    audio_offset = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        items = []
        for segment in filter(None, re.split(r"(<audio>)", content)):
            if segment != "<audio>":
                items.append({"type": "text", "text": segment})
                continue
            if audio_offset >= len(audios):
                raise ValueError("prompt has more <audio> placeholders than audio paths")
            items.append({"type": "audio", "audio": audios[audio_offset]})
            audio_offset += 1
        message["content"] = items
    if audio_offset != len(audios):
        raise ValueError(f"prompt left {len(audios) - audio_offset} audio paths unused")
    return messages


def _prepare_audio_inputs(processor, example: dict) -> tuple[list[int], dict, dict]:
    if len(example["audios"]) != 1:
        raise ValueError(f"AudioMCQ parity requires exactly one audio, got {len(example['audios'])}")
    messages = _materialize_audio_messages(example)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    audio_config = OmegaConf.create({"audio_sampling_rate": sampling_rate, "audio_max_items": 1})
    audios = [RLHFDataset._load_local_audio(example["audios"][0], audio_config)]
    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[text],
        audio=audios,
        mm_processor_kwargs={"sampling_rate": sampling_rate},
    )
    input_features = model_inputs.get("input_features")
    feature_attention_mask = model_inputs.get("feature_attention_mask")
    if input_features is None or feature_attention_mask is None:
        raise RuntimeError("processor did not return audio features and their attention mask")
    audio_feature_lengths = feature_attention_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(audio_feature_lengths <= 0):
        raise RuntimeError(f"invalid audio feature lengths: {audio_feature_lengths.tolist()}")
    prompt_ids = normalize_token_ids(model_inputs["input_ids"])
    audio = np.ascontiguousarray(audios[0])
    audit = {
        "audio_path": example["audios"][0],
        "audio_sampling_rate": sampling_rate,
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "input_features_shape": list(input_features.shape),
        "feature_attention_mask_shape": list(feature_attention_mask.shape),
        "audio_feature_lengths": [int(value) for value in audio_feature_lengths.tolist()],
    }
    kwargs = {
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "audio_feature_lengths": audio_feature_lengths,
    }
    return prompt_ids, kwargs, audit


def _compare(left: list[float], right: list[float], mask: list[int]) -> dict:
    pairs = [
        (float(a), float(b))
        for a, b, keep in zip(left, right, mask, strict=True)
        if keep and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return {"paired_count": len(pairs), "pearson_corr": None, "abs_diff_mean": None, "abs_diff_max": None}
    lhs = torch.tensor([a for a, _ in pairs], dtype=torch.float64)
    rhs = torch.tensor([b for _, b in pairs], dtype=torch.float64)
    corr = torch.corrcoef(torch.stack([lhs, rhs]))[0, 1]
    diffs = (lhs - rhs).abs()
    return {
        "paired_count": len(pairs),
        "pearson_corr": float(corr.item()),
        "abs_diff_mean": float(diffs.mean().item()),
        "abs_diff_max": float(diffs.max().item()),
    }


def _dtype(value: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--dataset-row", type=int, default=0)
    parser.add_argument("--dump-row", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--min-correlation", type=float, default=0.99)
    parser.add_argument("--output-file")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    # Register the top-level Qwen3-Omni class as a thinker-only causal LM.
    import verl_omni.models.transformers.qwen3_omni_thinker  # noqa: F401

    record = _load_record(args.dump_jsonl, args.dump_row)
    example = _load_example(args.data_file, args.dataset_row)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids, audio_kwargs, audio_audit = _prepare_audio_inputs(processor, example)

    responses = [int(value) for value in record["responses"]]
    response_mask = [int(value) for value in record["response_mask"]]
    dumped_ids = [int(value) for value in record["input_ids"]]
    dumped_mask = [int(value) for value in record["attention_mask"]]
    response_start = len(dumped_ids) - len(responses)
    dumped_prompt_ids = [
        token for token, keep in zip(dumped_ids[:response_start], dumped_mask[:response_start], strict=True) if keep
    ]
    if dumped_prompt_ids != prompt_ids:
        raise ValueError(
            "processor prompt IDs do not match the dumped unpadded prompt: "
            f"processor={len(prompt_ids)} dumped={len(dumped_prompt_ids)}"
        )

    input_ids = torch.tensor([prompt_ids + responses], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "dtype": _dtype(args.dtype),
        "attn_implementation": args.attn_implementation,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    input_ids = input_ids.to(first_device)
    attention_mask = attention_mask.to(first_device)
    audio_kwargs = {
        name: value.to(first_device, dtype=model_dtype) if name == "input_features" else value.to(first_device)
        for name, value in audio_kwargs.items()
    }

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **audio_kwargs,
        )
        logits = output.logits.float()
        prompt_len = len(prompt_ids)
        target_ids = input_ids[0, prompt_len : prompt_len + len(responses)]
        pred_logits = logits[0, prompt_len - 1 : prompt_len - 1 + len(responses)]
        hf_log_probs = pred_logits.log_softmax(dim=-1).gather(1, target_ids.unsqueeze(1)).squeeze(1)

    hf = [float(value) for value in hf_log_probs.cpu().tolist()]
    actor = [float(value) for value in record["actor_old_log_probs"]]
    rollout = [float(value) for value in record["rollout_log_probs"]]
    ref = [float(value) for value in record.get("ref_log_probs") or []]
    actor_comparison = _compare(hf, actor, response_mask)
    result = {
        "event": "hf_audio_rollout_corr_score",
        "dump_jsonl": args.dump_jsonl,
        "dump_row": args.dump_row,
        "data_file": args.data_file,
        "dataset_row": args.dataset_row,
        "model_path": args.model_path,
        "prompt_len": len(prompt_ids),
        "prompt_ids_match": True,
        "response_len": len(responses),
        "valid_tokens": sum(response_mask),
        "audio": audio_audit,
        "hf_vs_actor_old": actor_comparison,
        "hf_vs_rollout": _compare(hf, rollout, response_mask),
        "hf_vs_ref": _compare(hf, ref, response_mask) if ref else None,
        "min_correlation_required": args.min_correlation,
        "status": (
            "PASS"
            if actor_comparison["pearson_corr"] is not None
            and actor_comparison["pearson_corr"] >= args.min_correlation
            else "FAIL"
        ),
        "sample": [
            {
                "position": index,
                "token_id": responses[index],
                "mask": response_mask[index],
                "hf": hf[index],
                "actor_old": actor[index],
                "rollout": rollout[index],
                **({"ref": ref[index]} if ref else {}),
            }
            for index in range(len(responses))
        ],
    }
    _emit(result, args.output_file)
    if args.strict and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
