#!/usr/bin/env python3
"""Compare HF and Megatron target-logit/LSE probes by unpadded token hash."""

import argparse
import glob
import json
import math
from pathlib import Path


def _load_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _summary(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        return {"paired_count": 0, "abs_diff_mean": None, "abs_diff_max": None, "pearson_corr": None}
    diffs = [a - b for a, b in zip(left, right)]
    abs_diffs = [abs(value) for value in diffs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    corr = None
    if left_var and right_var:
        covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
        corr = covariance / math.sqrt(left_var * right_var)
    return {
        "paired_count": len(left),
        "abs_diff_mean": sum(abs_diffs) / len(abs_diffs),
        "abs_diff_max": max(abs_diffs),
        "signed_diff_mean": sum(diffs) / len(diffs),
        "pearson_corr": corr,
    }


def _vector_summary(megatron: list[dict] | None, hf: list[dict], hf_key: str) -> dict | None:
    if megatron is None or len(megatron) != len(hf):
        return None
    hf_stats = [component.get(hf_key) for component in hf]
    if any(stat is None for stat in hf_stats):
        return None
    result = {
        "sum": _summary([stat["sum"] for stat in megatron], [stat["sum"] for stat in hf_stats]),
        "square_sum": _summary(
            [stat["square_sum"] for stat in megatron], [stat["square_sum"] for stat in hf_stats]
        ),
    }
    megatron_head = [value for stat in megatron for value in stat["head"]]
    hf_head = [value for stat in hf_stats for value in stat["head"]]
    result["head"] = _summary(megatron_head, hf_head)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-jsonl", required=True)
    parser.add_argument("--megatron-vocab-glob", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    hf_by_hash = {}
    for record in _load_jsonl(args.hf_jsonl):
        if record.get("event") == "hf_rollout_corr_score_row" and "input_ids_sha256" in record:
            hf_by_hash[record["input_ids_sha256"]] = record

    result_rows = []
    for path in sorted(glob.glob(args.megatron_vocab_glob)):
        for record in _load_jsonl(path):
            for audit in record.get("response_score_audit", []):
                hf = hf_by_hash.get(audit["input_ids_sha256"])
                row = {
                    "event": "megatron_hf_logprob_component_comparison",
                    "source_file": path,
                    "rank": record.get("rank"),
                    "forward_count": record.get("count"),
                    "input_ids_sha256": audit["input_ids_sha256"],
                    "input_len": audit["input_len"],
                    "response_len": audit["response_len"],
                    "matched_hf": hf is not None,
                }
                if hf is not None:
                    hf_components = hf["components"]
                    row["label_tokens_match"] = audit["label_token_ids"] == [
                        component["token_id"] for component in hf_components
                    ]
                    row["target_logit"] = _summary(
                        audit["target_logits"], [component["hf_target_logit"] for component in hf_components]
                    )
                    row["logsumexp"] = _summary(
                        audit["logsumexp"], [component["hf_logsumexp"] for component in hf_components]
                    )
                    row["logprob"] = _summary(
                        audit["manual_log_probs"], [component["hf_logprob"] for component in hf_components]
                    )
                    if audit.get("manual_lm_head_target_logits") is not None:
                        row["megatron_manual_lm_head_vs_output"] = _summary(
                            audit["manual_lm_head_target_logits"], audit["target_logits"]
                        )
                        hf_manual = [component.get("hf_manual_lm_head_target_logit") for component in hf_components]
                        if all(value is not None for value in hf_manual):
                            row["megatron_manual_lm_head_vs_hf_manual"] = _summary(
                                audit["manual_lm_head_target_logits"], hf_manual
                            )
                            row["hf_manual_lm_head_vs_output"] = _summary(
                                hf_manual, [component["hf_target_logit"] for component in hf_components]
                            )
                    lm_head_weight = _vector_summary(
                        audit.get("lm_head_weight"), hf_components, "hf_lm_head_weight"
                    )
                    if lm_head_weight is not None:
                        row["lm_head_weight"] = lm_head_weight
                    input_embedding = _vector_summary(
                        audit.get("input_embedding"), hf_components, "hf_input_embedding"
                    )
                    if input_embedding is not None:
                        row["input_embedding"] = input_embedding
                        hf_input_token_ids = [component.get("input_token_id") for component in hf_components]
                        # Older HF component dumps omitted this field even though
                        # their vector summaries are usable. Preserve those
                        # comparisons instead of failing the whole parity run.
                        row["input_tokens_match"] = (
                            audit.get("input_token_ids") == hf_input_token_ids
                            if all(token_id is not None for token_id in hf_input_token_ids)
                            else None
                        )
                    pre_lm_hidden = _vector_summary(
                        audit.get("pre_lm_hidden"), hf_components, "hf_pre_lm_hidden"
                    )
                    if pre_lm_hidden is not None:
                        row["pre_lm_hidden"] = pre_lm_hidden
                    pre_final_norm_hidden = _vector_summary(
                        audit.get("pre_final_norm_hidden"), hf_components, "hf_pre_final_norm_hidden"
                    )
                    if pre_final_norm_hidden is not None:
                        row["pre_final_norm_hidden"] = pre_final_norm_hidden
                    final_norm_weight = audit.get("final_norm_weight")
                    if final_norm_weight is not None:
                        row["final_norm_weight"] = _vector_summary(
                            [final_norm_weight] * len(hf_components),
                            hf_components,
                            "hf_final_norm_weight",
                        )
                result_rows.append(row)

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(result_rows)} component comparison rows to {output}")


if __name__ == "__main__":
    main()
