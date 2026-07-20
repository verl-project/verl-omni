#!/usr/bin/env python3
"""Compare R2 RECORD and REPLAY_FORWARD activation audits from one scorer run."""

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
        corr = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right)) / math.sqrt(
            left_var * right_var
        )
    return {
        "paired_count": len(left),
        "abs_diff_mean": sum(abs_diffs) / len(abs_diffs),
        "abs_diff_max": max(abs_diffs),
        "pearson_corr": corr,
    }


def _component_key(entry: dict) -> tuple:
    return (
        entry.get("input_ids_sha256"),
        int(entry["layer"]),
        entry["component"],
        int(entry["response_index"]),
    )


def _router_key(entry: dict) -> tuple:
    return (
        entry.get("input_ids_sha256"),
        int(entry["layer"]),
        int(entry["response_index"]),
    )


def _moe_mlp_stage_key(entry: dict) -> tuple:
    return (int(entry["layer"]), entry["stage"])


def _metadata_key(entry: dict) -> tuple:
    return (int(entry["layer"]), entry["phase"])


def _fingerprint_matches(left, right) -> bool | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    if "sha256" not in left or "sha256" not in right:
        return None
    return left.get("shape") == right.get("shape") and left["sha256"] == right["sha256"]


def _route(entry: dict) -> tuple[int, ...]:
    return tuple(sorted(int(expert) for expert in entry.get("expert_ids", [])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-decoder-glob", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = defaultdict(dict)
    action_counts = defaultdict(int)
    for path in sorted(glob.glob(args.megatron_decoder_glob)):
        for record in _load_jsonl(path):
            if record.get("event") != "megatron_decoder_component_audit":
                continue
            action = record.get("router_replay_action")
            if action not in {"record", "replay_forward"}:
                continue
            rank = int(record.get("rank", -1))
            # One action is expected once per rank. Keep the first complete
            # audit if a retry appends another event to the same debug file.
            records[(rank, action)].setdefault("components", record.get("decoder_component_audit", []))
            records[(rank, action)].setdefault("routers", record.get("moe_router_audit", []))
            records[(rank, action)].setdefault("moe_mlp_stages", record.get("moe_mlp_stage_audit", []))
            records[(rank, action)].setdefault("replay_metadata", record.get("moe_replay_metadata_audit", []))
            action_counts[action] += 1

    components = defaultdict(lambda: {"record": [], "replay": []})
    routes = defaultdict(lambda: {"record": [], "replay": []})
    moe_mlp_stages = defaultdict(lambda: {"record": [], "replay": []})
    metadata_checks = defaultdict(list)
    replay_target_map_checks = defaultdict(list)
    matched_ranks = []
    for rank in sorted({rank for rank, _ in records}):
        recorded = records.get((rank, "record"))
        replayed = records.get((rank, "replay_forward"))
        if recorded is None or replayed is None:
            continue
        matched_ranks.append(rank)
        record_components = {_component_key(entry): entry for entry in recorded["components"]}
        replay_components = {_component_key(entry): entry for entry in replayed["components"]}
        for key in sorted(set(record_components) & set(replay_components)):
            before, after = record_components[key], replay_components[key]
            if before.get("input_token_id") != after.get("input_token_id"):
                continue
            values = components[(key[1], key[2])]
            values["record"].append(before["stats"])
            values["replay"].append(after["stats"])

        record_routes = {_router_key(entry): entry for entry in recorded["routers"]}
        replay_routes = {_router_key(entry): entry for entry in replayed["routers"]}
        for key in sorted(set(record_routes) & set(replay_routes)):
            before, after = record_routes[key], replay_routes[key]
            if before.get("input_token_id") != after.get("input_token_id"):
                continue
            values = routes[key[1]]
            values["record"].append(before)
            values["replay"].append(after)

        record_moe_mlp_stages = {_moe_mlp_stage_key(entry): entry for entry in recorded["moe_mlp_stages"]}
        replay_moe_mlp_stages = {_moe_mlp_stage_key(entry): entry for entry in replayed["moe_mlp_stages"]}
        for key in sorted(set(record_moe_mlp_stages) & set(replay_moe_mlp_stages)):
            before, after = record_moe_mlp_stages[key], replay_moe_mlp_stages[key]
            values = moe_mlp_stages[key]
            values["record"].append(before["stats"])
            values["replay"].append(after["stats"])

        record_metadata = {_metadata_key(entry): entry for entry in recorded["replay_metadata"]}
        replay_metadata = {_metadata_key(entry): entry for entry in replayed["replay_metadata"]}
        for key in sorted(set(record_metadata) & set(replay_metadata)):
            before, after = record_metadata[key], replay_metadata[key]
            phase = key[1]
            common_fields = (
                ("input", "input", "input"),
                ("routing_map", "routing_map", "routing_map"),
                ("per_expert", "per_expert", "per_expert"),
            )
            if phase == "route":
                common_fields += (("routing_probs", "routing_probs", "routing_probs"),)
                common_fields += (("recorded_topk_to_target_topk", "recorded_topk", "target_topk"),)
            elif phase == "dispatcher_preprocess":
                common_fields += (
                    ("tokens_per_expert", "tokens_per_expert", "tokens_per_expert"),
                    ("input_splits", "input_splits", "input_splits"),
                    ("output_splits", "output_splits", "output_splits"),
                    ("output_splits_tp", "output_splits_tp", "output_splits_tp"),
                    ("local_permutation", "local_permutation", "local_permutation"),
                )
            for check, before_key, after_key in common_fields:
                matches = _fingerprint_matches(before.get(before_key), after.get(after_key))
                if matches is not None:
                    metadata_checks[(key[0], phase, check)].append((rank, matches))
            if phase == "route" and after.get("route_vs_target_map_mismatch_count") is not None:
                replay_target_map_checks[(key[0], phase)].append(
                    (rank, int(after["route_vs_target_map_mismatch_count"]))
                )

    component_summaries = []
    for (layer, component), values in sorted(components.items()):
        component_summaries.append(
            {
                "layer": layer,
                "component": component,
                "sum": _summary(
                    [float(entry["sum"]) for entry in values["record"]],
                    [float(entry["sum"]) for entry in values["replay"]],
                ),
                "square_sum": _summary(
                    [float(entry["square_sum"]) for entry in values["record"]],
                    [float(entry["square_sum"]) for entry in values["replay"]],
                ),
                "head": _summary(
                    [float(item) for entry in values["record"] for item in entry.get("head", [])],
                    [float(item) for entry in values["replay"] for item in entry.get("head", [])],
                ),
            }
        )

    route_summaries = []
    for layer, values in sorted(routes.items()):
        exact = [
            _route(before) == _route(after) for before, after in zip(values["record"], values["replay"], strict=True)
        ]
        margin_pairs = [
            (float(before["router_raw_topk_margin"]), float(after["router_raw_topk_margin"]))
            for before, after in zip(values["record"], values["replay"], strict=True)
            if before.get("router_raw_topk_margin") is not None and after.get("router_raw_topk_margin") is not None
        ]
        route_summaries.append(
            {
                "layer": layer,
                "matched_tokens": len(exact),
                "route_exact_fraction": sum(exact) / len(exact) if exact else None,
                "raw_topk_margin": _summary(
                    [before for before, _ in margin_pairs],
                    [after for _, after in margin_pairs],
                ),
            }
        )

    moe_mlp_stage_summaries = []
    for (layer, stage), values in sorted(moe_mlp_stages.items()):
        moe_mlp_stage_summaries.append(
            {
                "layer": layer,
                "stage": stage,
                "sum": _summary(
                    [float(entry["sum"]) for entry in values["record"]],
                    [float(entry["sum"]) for entry in values["replay"]],
                ),
                "square_sum": _summary(
                    [float(entry["square_sum"]) for entry in values["record"]],
                    [float(entry["square_sum"]) for entry in values["replay"]],
                ),
                "abs_max": _summary(
                    [float(entry["abs_max"]) for entry in values["record"]],
                    [float(entry["abs_max"]) for entry in values["replay"]],
                ),
                "head": _summary(
                    [float(item) for entry in values["record"] for item in entry.get("head", [])],
                    [float(item) for entry in values["replay"] for item in entry.get("head", [])],
                ),
            }
        )

    metadata_summaries = []
    for (layer, phase, check), values in sorted(metadata_checks.items()):
        mismatched_ranks = [rank for rank, matches in values if not matches]
        metadata_summaries.append(
            {
                "layer": layer,
                "phase": phase,
                "check": check,
                "compared_ranks": len(values),
                "exact_fraction": sum(matches for _, matches in values) / len(values),
                "mismatched_ranks": mismatched_ranks,
            }
        )

    replay_target_map_summaries = []
    for (layer, phase), values in sorted(replay_target_map_checks.items()):
        mismatch_counts = [count for _, count in values]
        replay_target_map_summaries.append(
            {
                "layer": layer,
                "phase": phase,
                "compared_ranks": len(values),
                "zero_mismatch_fraction": sum(count == 0 for count in mismatch_counts) / len(mismatch_counts),
                "mismatch_count_max": max(mismatch_counts),
                "mismatch_count_mean": sum(mismatch_counts) / len(mismatch_counts),
                "nonzero_mismatch_ranks": [rank for rank, count in values if count],
            }
        )

    result = {
        "event": "megatron_router_replay_audit_comparison",
        "action_event_counts": dict(action_counts),
        "matched_ranks": matched_ranks,
        "component_summaries": component_summaries,
        "route_summaries": route_summaries,
        "moe_mlp_stage_summaries": moe_mlp_stage_summaries,
        "metadata_summaries": metadata_summaries,
        "replay_target_map_summaries": replay_target_map_summaries,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote Megatron R2 record/replay comparison to {output}")


if __name__ == "__main__":
    main()
