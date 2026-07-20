#!/usr/bin/env python3
"""Locate the first R2 router-index mismatch across pack/scatter boundaries."""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


STAGE_PAIRS = (
    ("packed", "record_packed", "replay_input_packed"),
    ("sp_global", "record_sp_gather", "replay_sp_gather"),
    ("sp_local", "record_local", "replay_sp_scatter"),
    ("target", "record_local", "replay_target"),
)


def _load_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _flatten(values):
    if isinstance(values, list):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def _coordinate(flat_index: int, shape: list[int]) -> list[int]:
    coordinate = []
    for size in reversed(shape):
        coordinate.append(flat_index % size)
        flat_index //= size
    return list(reversed(coordinate))


def _first_mismatch(left: dict, right: dict) -> dict | None:
    if left["shape"] != right["shape"]:
        return {"reason": "shape", "record_shape": left["shape"], "replay_shape": right["shape"]}
    left_values = list(_flatten(left.get("values", [])))
    right_values = list(_flatten(right.get("values", [])))
    if len(left_values) != len(right_values):
        return {"reason": "value_length", "record_length": len(left_values), "replay_length": len(right_values)}
    for flat_index, (before, after) in enumerate(zip(left_values, right_values, strict=True)):
        if before != after:
            return {
                "reason": "value",
                "coordinate": _coordinate(flat_index, left["shape"]),
                "record_value": before,
                "replay_value": after,
            }
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = defaultdict(dict)
    for path in sorted(glob.glob(args.trace_glob)):
        for record in _load_jsonl(path):
            if record.get("event") != "megatron_router_replay_index_trace":
                continue
            key = (int(record["rank"]), int(record["layer"]), record["stage"])
            records.setdefault(key, record)

    summaries = []
    for label, record_stage, replay_stage in STAGE_PAIRS:
        comparisons = []
        for rank, layer in sorted({(rank, layer) for rank, layer, _ in records}):
            before = records.get((rank, layer, record_stage))
            after = records.get((rank, layer, replay_stage))
            if before is None or after is None:
                continue
            # The trace producer preserves each boundary's native tensor dtype.
            # Router indices can therefore be int64 on record and uint8 after
            # replay scatter even when every index value is identical. Compare
            # semantic values; the producer hashes remain useful diagnostics.
            first_mismatch = _first_mismatch(before, after)
            exact = first_mismatch is None
            comparisons.append(
                {
                    "rank": rank,
                    "layer": layer,
                    "exact": exact,
                    "first_mismatch": first_mismatch,
                }
            )
        if comparisons:
            summaries.append(
                {
                    "boundary": label,
                    "compared": len(comparisons),
                    "exact_fraction": sum(item["exact"] for item in comparisons) / len(comparisons),
                    "mismatches": [item for item in comparisons if not item["exact"]],
                }
            )

    result = {
        "event": "megatron_router_replay_index_trace_comparison",
        "trace_files": len(glob.glob(args.trace_glob)),
        "boundary_summaries": summaries,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote router replay index trace comparison to {output}")


if __name__ == "__main__":
    main()
