#!/usr/bin/env python3
"""Summarize the strict Megatron actor-versus-HF logprob parity gate."""

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-jsonl", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-logprob-corr", type=float, default=0.99)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with Path(args.component_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    valid_rows = [
        row
        for row in rows
        if row.get("matched_hf")
        and row.get("label_tokens_match")
        and isinstance(row.get("logprob", {}).get("pearson_corr"), (int, float))
        and math.isfinite(row["logprob"]["pearson_corr"])
    ]
    correlations = [float(row["logprob"]["pearson_corr"]) for row in valid_rows]
    abs_diffs = [float(row["logprob"]["abs_diff_mean"]) for row in valid_rows]
    minimum = min(correlations) if correlations else None
    result = {
        "event": "megatron_hf_parity_gate",
        "component_jsonl": args.component_jsonl,
        "min_logprob_corr_required": args.min_logprob_corr,
        "comparison_rows": len(rows),
        "valid_rows": len(valid_rows),
        "unique_inputs": len({row.get("input_ids_sha256") for row in valid_rows}),
        "logprob_corr_min": minimum,
        "logprob_corr_mean": sum(correlations) / len(correlations) if correlations else None,
        "logprob_abs_diff_mean_max": max(abs_diffs) if abs_diffs else None,
        "status": "PASS" if minimum is not None and minimum >= args.min_logprob_corr else "FAIL",
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if args.strict and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
