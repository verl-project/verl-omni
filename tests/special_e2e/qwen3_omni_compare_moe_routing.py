#!/usr/bin/env python3
"""Compare response-token MoE top-k routing captured by HF and Megatron."""

import argparse
import glob
import json
from collections import defaultdict


def _load_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _key(sample: str, entry: dict):
    return sample, int(entry["layer"]), int(entry["response_index"])


def _route(entry: dict):
    route = {int(expert): float(prob) for expert, prob in zip(entry["expert_ids"], entry["expert_probs"])}
    total = sum(route.values())
    # MCore's `moe_router_pre_softmax=False` normalizes over selected experts,
    # while the HF gate hook exposes probabilities normalized over all experts.
    # Normalize both top-k views before comparing their combination weights.
    return {expert: prob / total for expert, prob in route.items()} if total else route


def _stats_values(stats: dict | None) -> list[float]:
    if not isinstance(stats, dict):
        return []
    values = [stats.get("sum"), stats.get("square_sum"), *(stats.get("head") or [])]
    return [float(value) for value in values if value is not None]


def _top_logits(entry: dict) -> dict[int, float]:
    return {
        int(expert): float(value)
        for expert, value in zip(
            entry.get("router_logit_top_expert_ids", []), entry.get("router_logit_top_values", [])
        )
    }


def _weight_fingerprint_delta(left: dict, right: dict) -> dict:
    left_anchors = {tuple(item["index"]): float(item["value"]) for item in left.get("anchors", [])}
    right_anchors = {tuple(item["index"]): float(item["value"]) for item in right.get("anchors", [])}
    common = sorted(set(left_anchors) & set(right_anchors))
    anchor_diffs = [abs(left_anchors[index] - right_anchors[index]) for index in common]
    return {
        "shape_equal": left.get("shape") == right.get("shape"),
        "sum_abs_diff": abs(float(left["sum"]) - float(right["sum"])),
        "square_sum_abs_diff": abs(float(left["square_sum"]) - float(right["square_sum"])),
        "anchor_abs_diff_max": max(anchor_diffs) if anchor_diffs else None,
        "anchor_abs_diff_mean": sum(anchor_diffs) / len(anchor_diffs) if anchor_diffs else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-jsonl", required=True)
    parser.add_argument("--megatron-decoder-glob", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    hf = {}
    hf_weights = {}
    for record in _load_jsonl(args.hf_jsonl):
        if record.get("event") != "hf_rollout_corr_score_row":
            continue
        sample = record["input_ids_sha256"]
        for entry in record.get("moe_router_audit", []):
            hf[_key(sample, entry)] = entry
        for entry in record.get("moe_router_weight_audit", []):
            hf_weights.setdefault(int(entry["layer"]), entry)

    aggregates = defaultdict(lambda: {"matched": 0, "route_exact": 0, "prob_abs": []})
    per_sample = defaultdict(lambda: {"matched": 0, "route_exact": 0, "prob_abs": []})
    stages = defaultdict(
        lambda: {
            "matched": 0,
            "router_input_abs": [],
            "router_logits_abs": [],
            "router_raw_topk_margin_abs": [],
            "logit_top_exact": 0,
            "logit_top_overlap": [],
            "logit_top_value_abs": [],
        }
    )
    mcore_weights = {}
    disagreements = []
    alignment = {"matched": 0, "missing": 0, "token_mismatched": 0}
    for path in sorted(glob.glob(args.megatron_decoder_glob)):
        for record in _load_jsonl(path):
            if record.get("event") != "megatron_decoder_component_audit":
                continue
            replay_action = record.get("router_replay_action")
            if replay_action is not None and replay_action != "record":
                continue
            for entry in record.get("moe_router_weight_audit", []):
                mcore_weights.setdefault(int(entry["layer"]), entry)
            for entry in record.get("moe_router_audit", []):
                key = _key(entry["input_ids_sha256"], entry)
                hf_entry = hf.get(key)
                if hf_entry is None:
                    alignment["missing"] += 1
                    continue
                if entry.get("input_token_id") != hf_entry.get("input_token_id"):
                    alignment["token_mismatched"] += 1
                    continue
                alignment["matched"] += 1
                layer_key = key[1]
                sample_key = (key[0], layer_key)
                mcore_route = _route(entry)
                hf_route = _route(hf_entry)
                exact = set(mcore_route) == set(hf_route)
                for values in (aggregates[layer_key], per_sample[sample_key]):
                    values["matched"] += 1
                    values["route_exact"] += int(exact)
                    if exact:
                        values["prob_abs"].extend(
                            abs(mcore_route[expert] - hf_route[expert]) for expert in mcore_route
                        )
                stage = stages[layer_key]
                stage["matched"] += 1
                for field, bucket in (
                    ("router_input_stats", "router_input_abs"),
                    ("router_logits_stats", "router_logits_abs"),
                ):
                    left = _stats_values(entry.get(field))
                    right = _stats_values(hf_entry.get(field))
                    if len(left) == len(right):
                        stage[bucket].extend(abs(a - b) for a, b in zip(left, right))
                left_margin = entry.get("router_raw_topk_margin")
                right_margin = hf_entry.get("router_raw_topk_margin")
                if left_margin is not None and right_margin is not None:
                    stage["router_raw_topk_margin_abs"].append(abs(float(left_margin) - float(right_margin)))
                mcore_logits = _top_logits(entry)
                hf_logits = _top_logits(hf_entry)
                if mcore_logits and hf_logits:
                    mcore_ids, hf_ids = set(mcore_logits), set(hf_logits)
                    stage["logit_top_exact"] += int(mcore_ids == hf_ids)
                    stage["logit_top_overlap"].append(len(mcore_ids & hf_ids) / len(mcore_ids | hf_ids))
                    if mcore_ids == hf_ids:
                        stage["logit_top_value_abs"].extend(
                            abs(mcore_logits[expert] - hf_logits[expert]) for expert in mcore_ids
                        )
                if not exact and len(disagreements) < 32:
                    disagreements.append(
                        {
                            "input_ids_sha256": key[0],
                            "layer": layer_key,
                            "response_index": key[2],
                            "megatron_expert_ids": sorted(mcore_route),
                            "hf_expert_ids": sorted(hf_route),
                        }
                    )

    def _summary(key, values):
        probability_diffs = values["prob_abs"]
        return {
            **key,
            "matched_tokens": values["matched"],
            "route_exact_fraction": (
                values["route_exact"] / values["matched"] if values["matched"] else None
            ),
            "prob_abs_diff_mean": sum(probability_diffs) / len(probability_diffs) if probability_diffs else None,
            "prob_abs_diff_max": max(probability_diffs) if probability_diffs else None,
        }

    def _mean(values):
        return sum(values) / len(values) if values else None

    stage_summaries = []
    for layer, values in sorted(stages.items()):
        summary = {
                "layer": layer,
                "matched_tokens": values["matched"],
                "router_input_abs_diff_mean": _mean(values["router_input_abs"]),
                "router_input_abs_diff_max": max(values["router_input_abs"]) if values["router_input_abs"] else None,
                "router_logits_abs_diff_mean": _mean(values["router_logits_abs"]),
                "router_logits_abs_diff_max": max(values["router_logits_abs"]) if values["router_logits_abs"] else None,
                "router_logit_top16_exact_fraction": (
                    values["logit_top_exact"] / values["matched"] if values["matched"] else None
                ),
                "router_logit_top16_jaccard_mean": _mean(values["logit_top_overlap"]),
                "router_logit_top16_value_abs_diff_mean": _mean(values["logit_top_value_abs"]),
                "router_logit_top16_value_abs_diff_max": (
                    max(values["logit_top_value_abs"]) if values["logit_top_value_abs"] else None
                ),
        }
        if values["router_raw_topk_margin_abs"]:
            summary.update(
                {
                    "router_raw_topk_margin_abs_diff_mean": _mean(values["router_raw_topk_margin_abs"]),
                    "router_raw_topk_margin_abs_diff_max": max(values["router_raw_topk_margin_abs"]),
                }
            )
        stage_summaries.append(summary)

    weight_summaries = [
        {"layer": layer, **_weight_fingerprint_delta(mcore_weights[layer], hf_weights[layer])}
        for layer in sorted(set(mcore_weights) & set(hf_weights))
    ]

    result = {
        "event": "megatron_hf_moe_router_comparison",
        "token_alignment": alignment,
        "summaries": [_summary({"layer": layer}, values) for layer, values in sorted(aggregates.items())],
        "sample_summaries": [
            _summary({"input_ids_sha256": sample, "layer": layer}, values)
            for (sample, layer), values in sorted(per_sample.items())
        ],
        "stage_summaries": stage_summaries,
        "weight_summaries": weight_summaries,
        "route_disagreements": disagreements,
    }
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
