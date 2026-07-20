#!/usr/bin/env python3
"""Locate the first Megatron/HF decoder activation mismatch by response token."""

import argparse
import glob
import json
import math
from collections import defaultdict
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
        "pearson_corr": corr,
    }


def _compare_stats(megatron: list[dict], hf: list[dict]) -> dict:
    result = {}
    for field in ("sum", "square_sum"):
        result[field] = _summary([row[field] for row in megatron], [row[field] for row in hf])
    megatron_head = [value for row in megatron for value in row["head"]]
    hf_head = [value for row in hf for value in row["head"]]
    result["head"] = _summary(megatron_head, hf_head)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-jsonl", required=True)
    parser.add_argument("--megatron-decoder-glob", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--first-mismatch-threshold", type=float, default=1e-2)
    return parser.parse_args()


def main():
    args = parse_args()
    hf = {}
    for record in _load_jsonl(args.hf_jsonl):
        if record.get("event") != "hf_rollout_corr_score_row":
            continue
        for component in record.get("decoder_component_audit", []):
            key = (
                record["input_ids_sha256"],
                component["layer"],
                component["component"],
                component["response_index"],
            )
            hf[key] = component

    aggregate = defaultdict(lambda: {"megatron": [], "hf": []})
    per_sample_aggregate = defaultdict(lambda: {"megatron": [], "hf": []})
    token_alignment = {"matched": 0, "mismatched": 0, "missing": 0}
    topology = []
    for path in sorted(glob.glob(args.megatron_decoder_glob)):
        for record in _load_jsonl(path):
            if record.get("event") != "megatron_decoder_component_audit":
                continue
            # R2 emits RECORD and REPLAY_FORWARD. Keep the baseline in the
            # HF parity report; a dedicated report compares the two passes.
            replay_action = record.get("router_replay_action")
            if replay_action is not None and replay_action != "record":
                continue
            topology.append(record.get("topology", {}))
            for component in record.get("decoder_component_audit", []):
                key = (
                    component["input_ids_sha256"],
                    component["layer"],
                    component["component"],
                    component["response_index"],
                )
                hf_component = hf.get(key)
                if hf_component is None:
                    continue
                # Response indices are intentionally used because Megatron's
                # BSHD path compacts padding. When both writers carry a token
                # id, require it to agree before treating the activations as a
                # numerical pair.
                megatron_token = component.get("input_token_id")
                hf_token = hf_component.get("input_token_id")
                if megatron_token is None or hf_token is None:
                    token_alignment["missing"] += 1
                elif megatron_token != hf_token:
                    token_alignment["mismatched"] += 1
                    continue
                else:
                    token_alignment["matched"] += 1
                aggregate[(component["layer"], component["component"])]["megatron"].append(component["stats"])
                aggregate[(component["layer"], component["component"])]["hf"].append(hf_component["stats"])
                per_sample_aggregate[
                    (component["input_ids_sha256"], component["layer"], component["component"])
                ]["megatron"].append(component["stats"])
                per_sample_aggregate[
                    (component["input_ids_sha256"], component["layer"], component["component"])
                ]["hf"].append(hf_component["stats"])

    summaries = []
    for (layer, component), values in sorted(aggregate.items()):
        comparison = _compare_stats(values["megatron"], values["hf"])
        summaries.append(
            {
                "layer": layer,
                "component": component,
                "comparison": comparison,
                "matched_tokens": len(values["hf"]),
            }
        )

    first_mismatch = next(
        (
            summary
            for summary in summaries
            if summary["comparison"]["head"]["abs_diff_max"] is not None
            and summary["comparison"]["head"]["abs_diff_max"] > args.first_mismatch_threshold
        ),
        None,
    )
    sample_summaries = []
    for (sample_sha, layer, component), values in sorted(per_sample_aggregate.items()):
        comparison = _compare_stats(values["megatron"], values["hf"])
        sample_summaries.append(
            {
                "input_ids_sha256": sample_sha,
                "layer": layer,
                "component": component,
                "comparison": comparison,
                "matched_tokens": len(values["hf"]),
            }
        )
    first_mismatch_by_sample = {}
    for summary in sample_summaries:
        sample_sha = summary["input_ids_sha256"]
        if sample_sha in first_mismatch_by_sample:
            continue
        if (
            summary["comparison"]["head"]["abs_diff_max"] is not None
            and summary["comparison"]["head"]["abs_diff_max"] > args.first_mismatch_threshold
        ):
            first_mismatch_by_sample[sample_sha] = summary
    output = {
        "event": "megatron_hf_decoder_component_comparison",
        "hf_jsonl": args.hf_jsonl,
        "megatron_decoder_glob": args.megatron_decoder_glob,
        "topology": topology[:1],
        "token_alignment": token_alignment,
        "summary_count": len(summaries),
        "first_mismatch_threshold": args.first_mismatch_threshold,
        "first_mismatch": first_mismatch,
        "first_mismatch_by_sample": first_mismatch_by_sample,
        "summaries": summaries,
        "sample_summaries": sample_summaries,
    }
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote decoder component comparison to {path}")


if __name__ == "__main__":
    main()
