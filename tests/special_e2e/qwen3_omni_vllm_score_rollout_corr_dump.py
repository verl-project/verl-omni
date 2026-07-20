#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from transformers import AutoTokenizer


def _emit(record: dict, output_file: str | None) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load_records(path: str, limit: int) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "rollout_corr_sample":
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No rollout_corr_sample records found in {path}")
    return records


def _finite_pairs(left: list[float | None], right: list[float | None], mask: list[int]) -> list[tuple[float, float]]:
    pairs = []
    for a, b, keep in zip(left, right, mask):
        if not keep or a is None or b is None:
            continue
        a = float(a)
        b = float(b)
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    return pairs


def _corr(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = [x for x, _ in pairs]
    right = [y for _, y in pairs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    if left_var == 0 or right_var == 0:
        return None
    cov = sum((x - left_mean) * (y - right_mean) for x, y in pairs)
    return cov / math.sqrt(left_var * right_var)


def _stats(values: list[float | None], mask: list[int]) -> dict:
    kept = []
    missing = 0
    for value, keep in zip(values, mask):
        if not keep:
            continue
        if value is None:
            missing += 1
            continue
        value = float(value)
        if math.isfinite(value):
            kept.append(value)
        else:
            missing += 1
    if not kept:
        return {
            "count": 0,
            "missing_count": missing,
            "mean": None,
            "min": None,
            "max": None,
            "zero_fraction": None,
            "near_zero_fraction": None,
        }
    return {
        "count": len(kept),
        "missing_count": missing,
        "mean": sum(kept) / len(kept),
        "min": min(kept),
        "max": max(kept),
        "zero_fraction": sum(1 for x in kept if x == 0.0) / len(kept),
        "near_zero_fraction": sum(1 for x in kept if abs(x) < 1e-6) / len(kept),
    }


def _compare(name: str, left_name: str, left: list[float | None], right: list[float | None], mask: list[int]) -> dict:
    pairs = _finite_pairs(left, right, mask)
    prefix = f"{left_name}_vs_{name}"
    if not pairs:
        return {
            f"{prefix}/paired_count": 0,
            f"{prefix}/abs_diff_mean": None,
            f"{prefix}/abs_diff_max": None,
            f"{prefix}/signed_diff_mean": None,
            f"{prefix}/mult_prob_error_mean": None,
            f"{prefix}/mult_prob_error_max": None,
            f"{prefix}/corr": None,
        }
    diffs = [a - b for a, b in pairs]
    abs_diffs = [abs(x) for x in diffs]
    mult_errors = [math.exp(min(x, 80.0)) for x in abs_diffs]
    return {
        f"{prefix}/paired_count": len(pairs),
        f"{prefix}/abs_diff_mean": sum(abs_diffs) / len(abs_diffs),
        f"{prefix}/abs_diff_max": max(abs_diffs),
        f"{prefix}/signed_diff_mean": sum(diffs) / len(diffs),
        f"{prefix}/mult_prob_error_mean": sum(mult_errors) / len(mult_errors),
        f"{prefix}/mult_prob_error_max": max(mult_errors),
        f"{prefix}/corr": _corr(pairs),
    }


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _unpadded_prompt(record: dict) -> tuple[list[int], int]:
    response_len = len(record["responses"])
    response_start = len(record["input_ids"]) - response_len
    prompt_ids = record["input_ids"][:response_start]
    prompt_mask = record["attention_mask"][:response_start]
    unpadded = [int(tok) for tok, keep in zip(prompt_ids, prompt_mask) if int(keep) == 1]
    if not unpadded:
        raise ValueError(f"row={record.get('row')} has no unpadded prompt tokens")
    return unpadded, response_start


def _extract_prompt_logprob(prompt_logprobs, token_id: int, position: int) -> float | None:
    if prompt_logprobs is None or position >= len(prompt_logprobs):
        return None
    row = prompt_logprobs[position]
    if row is None or token_id not in row:
        return None
    return _safe_float(getattr(row[token_id], "logprob", None))


async def _run(args: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm_omni.entrypoints import AsyncOmni

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    records = _load_records(args.dump_jsonl, args.record_limit)
    engine = AsyncOmni(
        model=args.model_path,
        trust_remote_code=True,
        stage_configs_path=args.stage_config,
        max_model_len=args.max_model_len,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.init_timeout,
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
            "event": "vllm_rollout_corr_score_start",
            "dump_jsonl": args.dump_jsonl,
            "model_path": args.model_path,
            "stage_config": args.stage_config,
            "record_count": len(records),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "vllm_use_v1": os.getenv("VLLM_USE_V1"),
            "max_model_len": args.max_model_len,
            "prompt_logprobs": args.prompt_logprobs,
        },
        args.output_file,
    )

    try:
        for record in records:
            prompt_ids, padded_response_start = _unpadded_prompt(record)
            responses = [int(x) for x in record["responses"]]
            response_mask = [int(x) for x in record["response_mask"]]
            full_ids = prompt_ids + responses
            if len(full_ids) + 1 > args.max_model_len:
                _emit(
                    {
                        "event": "vllm_rollout_corr_score_row",
                        "row": record["row"],
                        "skipped": True,
                        "reason": "prompt_plus_response_exceeds_max_model_len",
                        "unpadded_prompt_len": len(prompt_ids),
                        "response_len": len(responses),
                        "max_model_len": args.max_model_len,
                    },
                    args.output_file,
                )
                continue

            request_id = f"rollout_corr_score_{record['row']}_{uuid4().hex[:8]}"
            final = None
            async for output in engine.generate(
                prompt={"prompt_token_ids": full_ids},
                sampling_params=scoring_sampling,
                request_id=request_id,
            ):
                final = output
            if final is None or final.request_output is None:
                raise RuntimeError(f"No vLLM-Omni score output for {request_id}")
            prompt_logprobs = final.request_output.prompt_logprobs
            vllm_tf = [
                _extract_prompt_logprob(prompt_logprobs, token_id, len(prompt_ids) + i)
                for i, token_id in enumerate(responses)
            ]
            rollout = [float(x) for x in record["rollout_log_probs"]]
            actor_old = [float(x) for x in record["actor_old_log_probs"]]
            ref = [float(x) for x in record.get("ref_log_probs", [])]
            decoded = tokenizer.decode(
                [tok for tok, keep in zip(responses, response_mask) if keep][: args.sample_limit],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            result = {
                "event": "vllm_rollout_corr_score_row",
                "row": record["row"],
                "request_id": request_id,
                "padded_input_len": len(record["input_ids"]),
                "padded_response_start": padded_response_start,
                "unpadded_prompt_len": len(prompt_ids),
                "response_len": len(responses),
                "valid_tokens": sum(response_mask),
                "decoded_valid_head": decoded,
                "vllm_teacher_forcing_stats": _stats(vllm_tf, response_mask),
                "rollout_stats": _stats(rollout, response_mask),
                "actor_old_stats": _stats(actor_old, response_mask),
                "vllm_tf_vs_rollout": _compare("rollout", "vllm_tf", vllm_tf, rollout, response_mask),
                "vllm_tf_vs_actor_old": _compare("actor_old", "vllm_tf", vllm_tf, actor_old, response_mask),
                "sample": [
                    {
                        "i": i,
                        "token_id": responses[i],
                        "mask": response_mask[i],
                        "vllm_tf": None if vllm_tf[i] is None else round(float(vllm_tf[i]), 6),
                        "rollout": round(rollout[i], 6),
                        "actor_old": round(actor_old[i], 6),
                        **({"ref": round(ref[i], 6)} if ref else {}),
                    }
                    for i in range(min(args.sample_limit, len(responses)))
                ],
            }
            if ref:
                result["ref_stats"] = _stats(ref, response_mask)
                result["vllm_tf_vs_ref"] = _compare("ref", "vllm_tf", vllm_tf, ref, response_mask)
            _emit(result, args.output_file)
    finally:
        for name in ("shutdown", "close"):
            fn = getattr(engine, name, None)
            if fn is None:
                continue
            result = fn()
            if hasattr(result, "__await__"):
                await result
            break

    _emit({"event": "vllm_rollout_corr_score_done"}, args.output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM-Omni-score fixed rollout-corr samples dumped from verl.")
    parser.add_argument("--dump-jsonl", required=True)
    parser.add_argument(
        "--model-path",
        default="/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate",
    )
    parser.add_argument(
        "--stage-config",
        default=str(Path(__file__).resolve().parent / "qwen3_omni_thinker_only_tp4_full_async_no_sleep_raw_logprobs.yaml"),
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--stage-init-timeout", type=int, default=1800)
    parser.add_argument("--init-timeout", type=int, default=1800)
    parser.add_argument("--prompt-logprobs", type=int, default=0)
    parser.add_argument("--record-limit", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--output-file")
    args = parser.parse_args()
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text("", encoding="utf-8")
    return args


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))
