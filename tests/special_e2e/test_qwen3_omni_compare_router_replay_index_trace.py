import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen3_omni_compare_router_replay_index_trace.py")


def _trace(action: str, stage: str, values: list[list[int]]) -> dict:
    flat = bytes(item for row in values for item in row)
    import hashlib

    return {
        "event": "megatron_router_replay_index_trace",
        "action": action,
        "stage": stage,
        "layer": 1,
        "rank": 3,
        "shape": [len(values), len(values[0])],
        "dtype": "torch.uint8",
        "sha256": hashlib.sha256(flat).hexdigest(),
        "values": values,
    }


def test_index_trace_comparator_reports_first_target_mismatch(tmp_path):
    source = tmp_path / "trace.rank3.jsonl"
    records = [
        _trace("record", "record_local", [[1, 2], [3, 4]]),
        _trace("record", "record_sp_gather", [[1, 2], [3, 4]]),
        _trace("record", "record_packed", [[1, 2], [3, 4]]),
        _trace("replay", "replay_input_packed", [[1, 2], [3, 4]]),
        _trace("replay", "replay_sp_gather", [[1, 2], [3, 4]]),
        _trace("replay", "replay_sp_scatter", [[1, 2], [3, 5]]),
        _trace("replay", "replay_target", [[1, 2], [3, 5]]),
    ]
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-glob",
            str(tmp_path / "trace.rank*.jsonl"),
            "--output-file",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    summaries = {entry["boundary"]: entry for entry in result["boundary_summaries"]}
    assert summaries["packed"]["exact_fraction"] == 1.0
    assert summaries["sp_global"]["exact_fraction"] == 1.0
    assert summaries["sp_local"]["mismatches"][0]["first_mismatch"] == {
        "reason": "value",
        "coordinate": [1, 1],
        "record_value": 4,
        "replay_value": 5,
    }


def test_index_trace_comparator_ignores_dtype_only_hash_difference(tmp_path):
    source = tmp_path / "trace.rank3.jsonl"
    record = _trace("record", "record_local", [[1, 2], [3, 4]])
    replay = _trace("replay", "replay_sp_scatter", [[1, 2], [3, 4]])
    record["dtype"] = "torch.int64"
    record["sha256"] = "hash-of-int64-storage"
    replay["dtype"] = "torch.uint8"
    replay["sha256"] = "hash-of-uint8-storage"
    source.write_text("\n".join(json.dumps(item) for item in [record, replay]) + "\n", encoding="utf-8")
    output = tmp_path / "comparison.json"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--trace-glob", str(tmp_path / "trace.rank*.jsonl"), "--output-file", str(output)],
        check=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    summaries = {entry["boundary"]: entry for entry in result["boundary_summaries"]}
    assert summaries["sp_local"]["exact_fraction"] == 1.0
    assert summaries["sp_local"]["mismatches"] == []
