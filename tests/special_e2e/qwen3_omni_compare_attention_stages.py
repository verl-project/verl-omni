#!/usr/bin/env python3
"""Compare the first-layer Megatron/HF attention-stage audit."""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from qwen3_omni_compare_decoder_components import _compare_stats, _load_jsonl


def _execution_signature(item: dict) -> dict:
    """Keep dispatch-relevant fields while excluding process-local storage addresses."""
    layouts = {}
    for name in ("query_layout", "key_layout", "value_layout"):
        layout = item.get(name, {})
        layouts[name] = {
            key: value
            for key, value in layout.items()
            if key not in {"storage_data_ptr", "storage_nbytes"}
        }
    return {
        "core_attention_type": item.get("core_attention_type"),
        "path": item.get("path"),
        "has_packed_seq_params": item.get("has_packed_seq_params"),
        "has_attention_bias": item.get("has_attention_bias"),
        "has_inference_context": item.get("has_inference_context"),
        "layouts": layouts,
        "qkv_aliases": item.get("qkv_aliases"),
        "response_mask": item.get("response_mask"),
        "runtime": item.get("runtime"),
    }


def _flatten(value, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(child, child_prefix))
        return flattened
    return {prefix: json.dumps(value, ensure_ascii=True, sort_keys=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-jsonl", required=True)
    parser.add_argument("--megatron-decoder-glob", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    hf = {}
    for record in _load_jsonl(args.hf_jsonl):
        if record.get("event") != "hf_rollout_corr_score_row":
            continue
        for item in record.get("attention_stage_audit", []):
            hf[(record["input_ids_sha256"], item["layer"], item["stage"], item["tp_rank"], item["response_index"])] = item

    aggregate = defaultdict(lambda: {"megatron": [], "hf": []})
    per_sample_aggregate = defaultdict(lambda: {"megatron": [], "hf": []})
    alignment = {"matched": 0, "mismatched": 0, "missing": 0}
    stages_seen = set()
    execution_by_sample_layer = defaultdict(list)
    for path in sorted(glob.glob(args.megatron_decoder_glob)):
        for record in _load_jsonl(path):
            if record.get("event") != "megatron_decoder_component_audit":
                continue
            replay_action = record.get("router_replay_action")
            if replay_action is not None and replay_action != "record":
                continue
            for item in record.get("attention_execution_audit", []):
                execution_by_sample_layer[(item["input_ids_sha256"], item["layer"])].append(
                    {
                        "rank": record["rank"],
                        "tp_rank": record["tp_rank"],
                        "signature": _execution_signature(item),
                    }
                )
            for item in record.get("attention_stage_audit", []):
                stages_seen.add(item["stage"])
                key = (item["input_ids_sha256"], item["layer"], item["stage"], item["tp_rank"], item["response_index"])
                expected = hf.get(key)
                if expected is None:
                    alignment["missing"] += 1
                    continue
                if expected["input_token_id"] != item["input_token_id"]:
                    alignment["mismatched"] += 1
                    continue
                alignment["matched"] += 1
                aggregate[(item["layer"], item["stage"])]["megatron"].append(item["stats"])
                aggregate[(item["layer"], item["stage"])]["hf"].append(expected["stats"])
                per_sample_aggregate[(item["input_ids_sha256"], item["layer"], item["stage"])][
                    "megatron"
                ].append(item["stats"])
                per_sample_aggregate[(item["input_ids_sha256"], item["layer"], item["stage"])][
                    "hf"
                ].append(expected["stats"])

    summaries = [
        {
            "layer": layer,
            "stage": stage,
            "matched_tokens": len(values["hf"]),
            "comparison": _compare_stats(values["megatron"], values["hf"]),
        }
        for (layer, stage), values in sorted(aggregate.items())
    ]
    sample_summaries = [
        {
            "input_ids_sha256": sample_sha,
            "layer": layer,
            "stage": stage,
            "matched_tokens": len(values["hf"]),
            "comparison": _compare_stats(values["megatron"], values["hf"]),
        }
        for (sample_sha, layer, stage), values in sorted(per_sample_aggregate.items())
    ]
    execution_differences = []
    for (sample_sha, layer), entries in sorted(execution_by_sample_layer.items()):
        field_values = defaultdict(dict)
        for entry in entries:
            rank_label = f"rank{entry['rank']}/tp{entry['tp_rank']}"
            for field, value in _flatten(entry["signature"]).items():
                field_values[field][rank_label] = value
        differing_fields = {
            field: values
            for field, values in sorted(field_values.items())
            if len(set(values.values())) > 1
        }
        execution_differences.append(
            {
                "input_ids_sha256": sample_sha,
                "layer": layer,
                "rank_signatures": entries,
                "cross_rank_differences": differing_fields,
            }
        )
    Path(args.output_file).write_text(
        json.dumps(
            {
                "event": "megatron_hf_attention_stage_comparison",
                "stages_seen_in_megatron": sorted(stages_seen),
                "token_alignment": alignment,
                "summaries": summaries,
                "sample_summaries": sample_summaries,
                "attention_execution_audit": execution_differences,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
