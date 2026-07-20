#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _emit(record: dict, output_file: str | None) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load_rows(jsonl_file: str) -> tuple[dict, list[dict]]:
    start = {}
    rows = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "probe_start":
                start = record
            elif record.get("event") == "row_result":
                rows.append(record)
    if not rows:
        raise ValueError(f"No row_result records found in {jsonl_file}")
    return start, rows


def _load_prompt(data_file: str, row_index: int) -> list[dict]:
    table = pq.read_table(data_file)
    row = table.slice(row_index, 1).to_pylist()[0]
    return row["prompt"]


def _finite_stats(values: list[float]) -> dict:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    if not finite:
        return {
            "valid_logprob_count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "zero_fraction": None,
            "near_zero_fraction": None,
        }
    return {
        "valid_logprob_count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
        "zero_fraction": sum(1 for x in finite if x == 0.0) / len(finite),
        "near_zero_fraction": sum(1 for x in finite if abs(x) < 1e-6) / len(finite),
    }


def _compare(left: list[float], right: list[float]) -> dict:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if math.isfinite(float(a)) and math.isfinite(float(b))
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


def _extract_vllm_values(row: dict, key: str) -> list[float]:
    sample = (row.get(key) or {}).get("sample") or []
    values = []
    for item in sample:
        value = item.get("logprob")
        if value is not None:
            values.append(float(value))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score vLLM-generated Qwen3-Omni tokens with HF thinker.")
    parser.add_argument("--jsonl-file", required=True)
    parser.add_argument(
        "--model-path",
        default="/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate",
    )
    parser.add_argument(
        "--data-file",
        default="/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--output-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, rows = _load_rows(args.jsonl_file)
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text("", encoding="utf-8")

    # Register Qwen3-Omni thinker-only forward with AutoModelForCausalLM.
    import verl_omni.models.transformers.qwen3_omni_thinker  # noqa: F401

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = args.dtype
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float32":
        dtype = torch.float32

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    _emit(
        {
            "event": "hf_score_start",
            "jsonl_file": args.jsonl_file,
            "source_probe_start": start,
            "model_path": args.model_path,
            "data_file": args.data_file,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        args.output_file,
    )

    for row in rows:
        token_ids = row.get("token_ids")
        if token_ids is None:
            token_ids = row.get("token_ids_head")
        token_ids = [int(x) for x in token_ids]
        prompt = _load_prompt(args.data_file, int(row["row_index"]))
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        input_ids_list = prompt_ids + token_ids
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        first_device = next(model.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)

        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits.float()
            target_ids = input_ids[0, len(prompt_ids) : len(prompt_ids) + len(token_ids)]
            pred_logits = logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(token_ids), :]
            hf_logprobs = pred_logits.log_softmax(dim=-1).gather(1, target_ids.unsqueeze(1)).squeeze(1)
        hf_values = [float(x) for x in hf_logprobs.detach().cpu().tolist()]

        vllm_values = _extract_vllm_values(row, "logprob_summary")
        vllm_score_values = _extract_vllm_values(row, "teacher_forcing_logprob_summary")
        _emit(
            {
                "event": "hf_score_row",
                "row_index": row["row_index"],
                "prompt_len": len(prompt_ids),
                "token_count": len(token_ids),
                "hf_logprob_summary": {
                    **_finite_stats(hf_values),
                    "sample": [
                        {
                            "i": i,
                            "token_id": int(token_ids[i]),
                            "logprob": round(float(hf_values[i]), 6),
                        }
                        for i in range(min(args.sample_limit, len(hf_values)))
                    ],
                },
                "hf_vs_vllm_generate": _compare(hf_values, vllm_values),
                "hf_vs_vllm_teacher_forcing": _compare(hf_values, vllm_score_values),
                "vllm_generate_mean": (row.get("logprob_summary") or {}).get("mean"),
                "vllm_teacher_forcing_mean": (row.get("teacher_forcing_logprob_summary") or {}).get("mean"),
            },
            args.output_file,
        )

    _emit({"event": "hf_score_done"}, args.output_file)


if __name__ == "__main__":
    main()
