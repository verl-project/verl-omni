#!/usr/bin/env python3
import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def _emit(record: dict, output_file: str | None) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load_examples(data_file: str, row_indices: list[int]) -> list[dict]:
    table = pq.read_table(data_file)
    examples = []
    for row_index in row_indices:
        row = table.slice(row_index, 1).to_pylist()[0]
        examples.append(
            {
                "row_index": row_index,
                "prompt": row["prompt"],
                "images": row.get("images") or [],
                "audios": row.get("audios") or [],
                "ground_truth": row.get("reward_model", {}).get("ground_truth"),
            }
        )
    return examples


def _materialize_messages(example: dict) -> list[dict]:
    """Replace image/audio placeholders with processor-ready content items."""
    messages = copy.deepcopy(example["prompt"])
    images = example["images"]
    audios = example["audios"]
    image_offset = 0
    audio_offset = 0
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            continue
        content_list = []
        for segment in filter(None, re.split(r"(<image>|<audio>)", content)):
            if segment not in {"<image>", "<audio>"}:
                content_list.append({"type": "text", "text": segment})
                continue
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError(f"row {example['row_index']} has fewer images than <image> placeholders")
                image = images[image_offset]
                if isinstance(image, dict) and image.get("bytes") is not None:
                    image = Image.open(BytesIO(image["bytes"])).convert("RGB")
                elif isinstance(image, Image.Image):
                    image = image.convert("RGB")
                else:
                    raise TypeError(f"Unsupported Geo3K image type: {type(image)}")
                content_list.append({"type": "image", "image": image})
                image_offset += 1
            else:
                if audio_offset >= len(audios):
                    raise ValueError(f"row {example['row_index']} has fewer audios than <audio> placeholders")
                content_list.append({"type": "audio", "audio": audios[audio_offset]})
                audio_offset += 1
        message["content"] = content_list
    if image_offset != len(images):
        raise ValueError(f"row {example['row_index']} has unused images: {len(images) - image_offset}")
    if audio_offset != len(audios):
        raise ValueError(f"row {example['row_index']} has unused audios: {len(audios) - audio_offset}")
    return messages


def _prepare_prompt(processor, example: dict) -> tuple[list[int], dict, dict, dict]:
    """Mirror AgentLoop preprocessing and retain raw vLLM multimodal payloads."""
    if example["images"] and example["audios"]:
        raise NotImplementedError("The standalone probe does not support mixed image-audio prompts")
    if not example["images"] and not example["audios"]:
        prompt_ids = normalize_token_ids(
            processor.apply_chat_template(example["prompt"], tokenize=True, add_generation_prompt=True)
        )
        return prompt_ids, {}, {}, {"image_count": 0, "audio_count": 0}

    messages = _materialize_messages(example)
    raw_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if example["images"]:
        from qwen_vl_utils import process_vision_info

        image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        if videos:
            raise NotImplementedError("This Qwen3-Omni AR probe supports images only, not video")
        model_inputs = build_multimodal_processor_inputs(processor, text=[raw_prompt], images=images)
        prompt_ids = normalize_token_ids(model_inputs["input_ids"])
        image_grid_thw = model_inputs.get("image_grid_thw")
        return prompt_ids, {"image": images}, {}, {
            "image_count": len(images or []),
            "audio_count": 0,
            "image_grid_thw": None if image_grid_thw is None else image_grid_thw.tolist(),
            "pixel_values_shape": list(model_inputs["pixel_values"].shape),
        }

    if len(example["audios"]) != 1:
        raise ValueError(f"AudioMCQ standalone probe requires exactly one audio, got {len(example['audios'])}")
    sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    audio_config = OmegaConf.create({"audio_sampling_rate": sampling_rate, "audio_max_items": 1})
    audios = [RLHFDataset._load_local_audio(audio_path, audio_config) for audio_path in example["audios"]]
    mm_processor_kwargs = {"sampling_rate": sampling_rate}
    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[raw_prompt],
        audio=audios,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    input_features = model_inputs.get("input_features")
    feature_attention_mask = model_inputs.get("feature_attention_mask")
    if input_features is None or feature_attention_mask is None:
        raise RuntimeError("Qwen3-Omni processor dropped AudioMCQ input_features or feature_attention_mask")
    audio_feature_lengths = feature_attention_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(audio_feature_lengths <= 0):
        raise RuntimeError(f"Qwen3-Omni processor produced invalid audio lengths: {audio_feature_lengths.tolist()}")
    prompt_ids = normalize_token_ids(model_inputs["input_ids"])
    return prompt_ids, {"audio": audios}, mm_processor_kwargs, {
        "image_count": 0,
        "audio_count": len(audios),
        "audio_sampling_rate": sampling_rate,
        "audio_sample_counts": [int(audio.size) for audio in audios],
        "audio_sha256": [hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest() for audio in audios],
        "input_features_shape": list(input_features.shape),
        "feature_attention_mask_shape": list(feature_attention_mask.shape),
        "audio_feature_lengths": audio_feature_lengths.tolist(),
    }


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _extract_sampled_values(token_ids: list[int], output_logprobs) -> tuple[list[float | None], list[int | None]]:
    values: list[float | None] = []
    ranks: list[int | None] = []
    rows = list(output_logprobs or [])
    for idx, token_id in enumerate(token_ids):
        row = rows[idx] if idx < len(rows) else None
        if row is None or token_id not in row:
            values.append(None)
            ranks.append(None)
            continue
        entry = row[token_id]
        values.append(_safe_float(getattr(entry, "logprob", None)))
        rank = getattr(entry, "rank", None)
        ranks.append(None if rank is None else int(rank))
    return values, ranks


def _summarize_logprobs(token_ids: list[int], output_logprobs, sample_limit: int) -> dict:
    values = []
    ranks = []
    samples = []
    missing = 0
    duplicate_sampled = 0
    rows = list(output_logprobs or [])
    for idx, token_id in enumerate(token_ids):
        row = rows[idx] if idx < len(rows) else None
        if row is None or token_id not in row:
            missing += 1
            samples.append({"i": idx, "token_id": int(token_id), "missing": True})
            continue
        entry = row[token_id]
        logprob = _safe_float(getattr(entry, "logprob", None))
        rank = getattr(entry, "rank", None)
        if logprob is not None:
            values.append(logprob)
        if rank is not None:
            ranks.append(int(rank))
        row_keys = list(row.keys())
        if row_keys.count(token_id) > 1:
            duplicate_sampled += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "i": idx,
                    "token_id": int(token_id),
                    "logprob": None if logprob is None else round(logprob, 6),
                    "rank": None if rank is None else int(rank),
                    "row_keys": [int(x) for x in row_keys[:8]],
                    "row_values": [
                        round(float(getattr(row[x], "logprob", float("nan"))), 6)
                        for x in row_keys[:8]
                    ],
                }
            )

    count = len(values)
    zero_count = sum(1 for x in values if x == 0.0)
    near_zero_count = sum(1 for x in values if abs(x) < 1e-6)
    rank1_count = sum(1 for x in ranks if x == 1)
    return {
        "token_count": len(token_ids),
        "logprob_rows": len(rows),
        "valid_logprob_count": count,
        "missing_sampled_count": missing,
        "duplicate_sampled_key_count": duplicate_sampled,
        "mean": None if not values else sum(values) / count,
        "min": None if not values else min(values),
        "max": None if not values else max(values),
        "zero_fraction": None if not values else zero_count / count,
        "near_zero_fraction": None if not values else near_zero_count / count,
        "rank1_fraction": None if not ranks else rank1_count / len(ranks),
        "sample": samples,
    }


def _compare_logprobs(left: list[float | None], right: list[float | None]) -> dict:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right, strict=False)
        if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
    ]
    if not pairs:
        return {
            "paired_count": 0,
            "abs_diff_mean": None,
            "abs_diff_max": None,
            "signed_diff_mean": None,
        }
    diffs = [a - b for a, b in pairs]
    abs_diffs = [abs(x) for x in diffs]
    return {
        "paired_count": len(pairs),
        "abs_diff_mean": sum(abs_diffs) / len(abs_diffs),
        "abs_diff_max": max(abs_diffs),
        "signed_diff_mean": sum(diffs) / len(diffs),
    }


async def _run(args: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm_omni.entrypoints import AsyncOmni

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    examples = _load_examples(args.data_file, args.row_indices)
    engine = AsyncOmni(
        model=args.model_path,
        trust_remote_code=True,
        stage_configs_path=args.stage_config,
        max_model_len=args.max_model_len,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.init_timeout,
    )
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ignore_eos=args.ignore_eos,
        seed=args.seed,
        logprobs=args.logprobs,
    )
    scoring_sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        prompt_logprobs=args.prompt_logprobs,
        logprobs=None,
    )

    _emit(
        {
            "event": "probe_start",
            "model_path": args.model_path,
            "stage_config": args.stage_config,
            "data_file": args.data_file,
            "row_indices": args.row_indices,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "vllm_use_v1": os.getenv("VLLM_USE_V1"),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "ignore_eos": args.ignore_eos,
            "logprobs": args.logprobs,
            "score_generated": args.score_generated,
            "prompt_logprobs": args.prompt_logprobs,
            "concurrency": args.concurrency,
        },
        args.output_file,
    )

    try:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def _run_one_example(example: dict) -> None:
            async with semaphore:
                await _run_one_example_unlocked(example)

        async def _run_one_example_unlocked(example: dict) -> None:
            prompt_ids, multi_modal_data, mm_processor_kwargs, multimodal_summary = _prepare_prompt(processor, example)
            prompt = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}
            if mm_processor_kwargs:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs
            request_id = f"standalone_logprob_{example['row_index']}_{uuid4().hex[:8]}"
            final = None
            async for output in engine.generate(
                prompt=prompt,
                sampling_params=sampling,
                request_id=request_id,
            ):
                final = output
            if final is None or final.request_output is None:
                raise RuntimeError(f"No vLLM-Omni output for {request_id}")
            completion = final.request_output.outputs[0]
            token_ids = list(completion.token_ids)
            summary = _summarize_logprobs(
                token_ids=token_ids,
                output_logprobs=completion.logprobs,
                sample_limit=args.sample_limit,
            )
            generate_values, _ = _extract_sampled_values(token_ids, completion.logprobs)
            score_summary = None
            score_compare = None
            if args.score_generated and token_ids:
                full_prompt_ids = prompt_ids + token_ids
                if len(full_prompt_ids) + 1 > args.max_model_len:
                    score_summary = {
                        "skipped": True,
                        "reason": "prompt_plus_generated_exceeds_max_model_len",
                        "full_prompt_len": len(full_prompt_ids),
                        "max_model_len": args.max_model_len,
                    }
                else:
                    score_request_id = f"{request_id}_score"
                    score_final = None
                    score_prompt = {"prompt_token_ids": full_prompt_ids, "multi_modal_data": multi_modal_data}
                    if mm_processor_kwargs:
                        score_prompt["mm_processor_kwargs"] = mm_processor_kwargs
                    async for score_output in engine.generate(
                        prompt=score_prompt,
                        sampling_params=scoring_sampling,
                        request_id=score_request_id,
                    ):
                        score_final = score_output
                    if score_final is None or score_final.request_output is None:
                        raise RuntimeError(f"No vLLM-Omni score output for {score_request_id}")
                    prompt_logprobs = score_final.request_output.prompt_logprobs
                    scored_prompt_len = len(score_final.request_output.prompt_token_ids or [])
                    score_start = scored_prompt_len - len(token_ids)
                    if score_start < 0:
                        raise RuntimeError(
                            f"Score prompt is shorter than generated completion: {scored_prompt_len} < {len(token_ids)}"
                        )
                    score_diagnostics = {
                        "request_prompt_token_ids_len": scored_prompt_len,
                        "prompt_logprobs_is_none": prompt_logprobs is None,
                        "prompt_logprobs_len": None if prompt_logprobs is None else len(prompt_logprobs),
                        "prompt_logprobs_non_none_rows": (
                            None if prompt_logprobs is None else sum(row is not None for row in prompt_logprobs)
                        ),
                        "completion_start": score_start,
                    }
                    score_rows = []
                    for i in range(len(token_ids)):
                        pos = score_start + i
                        row = None
                        if prompt_logprobs is not None and pos < len(prompt_logprobs):
                            row = prompt_logprobs[pos]
                        score_rows.append(row)
                    score_summary = _summarize_logprobs(
                        token_ids=token_ids,
                        output_logprobs=score_rows,
                        sample_limit=args.sample_limit,
                    )
                    score_summary["diagnostics"] = score_diagnostics
                    score_values, _ = _extract_sampled_values(token_ids, score_rows)
                    score_compare = _compare_logprobs(generate_values, score_values)
            text = processor.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            _emit(
                {
                    "event": "row_result",
                    "request_id": request_id,
                    "row_index": example["row_index"],
                    "ground_truth": example["ground_truth"],
                    "prompt_len": len(prompt_ids),
                    "multimodal": multimodal_summary,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "token_ids": token_ids,
                    "token_ids_head": token_ids[: args.sample_limit],
                    "text_head": text[: args.print_chars],
                    "logprob_summary": summary,
                    "teacher_forcing_logprob_summary": score_summary,
                    "generate_vs_teacher_forcing": score_compare,
                },
                args.output_file,
            )
        await asyncio.gather(*(_run_one_example(example) for example in examples))
    finally:
        for name in ("shutdown", "close"):
            fn = getattr(engine, name, None)
            if fn is None:
                continue
            result = fn()
            if hasattr(result, "__await__"):
                await result
            break

    _emit({"event": "probe_done"}, args.output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone vLLM-Omni AR logprob probe.")
    parser.add_argument(
        "--model-path",
        default="/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate",
    )
    parser.add_argument(
        "--stage-config",
        default=str(
            Path(__file__).resolve().parent / "qwen3_omni_thinker_only_tp4_full_async_no_sleep_raw_logprobs.yaml"
        ),
    )
    parser.add_argument(
        "--data-file",
        default="/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet",
    )
    parser.add_argument("--row-indices", default="0")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--stage-init-timeout", type=int, default=1800)
    parser.add_argument("--init-timeout", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--prompt-logprobs", type=int, default=0)
    parser.add_argument("--score-generated", dest="score_generated", action="store_true", default=True)
    parser.add_argument("--no-score-generated", dest="score_generated", action="store_false")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--print-chars", type=int, default=1200)
    parser.add_argument("--output-file")
    args = parser.parse_args()
    args.row_indices = [int(x) for x in args.row_indices.split(",") if x.strip()]
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text("", encoding="utf-8")
    return args


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))
