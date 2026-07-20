import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen3_omni_compare_megatron_router_replay.py")


def _component(action: str, sum_value: float) -> dict:
    topk_key = "recorded_topk" if action == "record" else "target_topk"
    return {
        "event": "megatron_decoder_component_audit",
        "rank": 3,
        "router_replay_action": action,
        "decoder_component_audit": [
            {
                "input_ids_sha256": "sample",
                "layer": 1,
                "component": "post_attention_residual",
                "response_index": 0,
                "input_token_id": 42,
                "stats": {"sum": sum_value, "square_sum": 2.0, "head": [0.1, 0.2]},
            }
        ],
        "moe_router_audit": [
            {
                "input_ids_sha256": "sample",
                "layer": 1,
                "response_index": 0,
                "input_token_id": 42,
                "expert_ids": [1, 3],
                "router_raw_topk_margin": 0.02,
            }
        ],
        "moe_mlp_stage_audit": [
            {
                "layer": 1,
                "stage": "dispatch_output:0",
                "stats": {"sum": sum_value, "square_sum": 2.0, "abs_max": 1.0, "head": [0.1, 0.2]},
            }
        ],
        "moe_replay_metadata_audit": [
            {
                "layer": 1,
                "phase": "route",
                "input": {"shape": [2, 2], "sha256": "input"},
                "routing_probs": {"shape": [2, 2], "sha256": "probs"},
                "routing_map": {"shape": [2, 2], "sha256": "map"},
                "per_expert": {"shape": [2], "sha256": "counts"},
                topk_key: {"shape": [2, 1], "sha256": "topk"},
                "route_vs_target_map_mismatch_count": 0,
            }
        ],
    }


def test_compare_megatron_router_replay_pairs_record_and_replay(tmp_path):
    source = tmp_path / "decoder.rank3.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in (_component("record", 1.0), _component("replay_forward", 1.25))
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "replay.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--megatron-decoder-glob",
            str(tmp_path / "decoder.rank*.jsonl"),
            "--output-file",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["matched_ranks"] == [3]
    assert result["route_summaries"] == [
        {
            "layer": 1,
            "matched_tokens": 1,
            "route_exact_fraction": 1.0,
            "raw_topk_margin": {
                "paired_count": 1,
                "abs_diff_mean": 0.0,
                "abs_diff_max": 0.0,
                "pearson_corr": None,
            },
        }
    ]
    assert result["moe_mlp_stage_summaries"][0]["stage"] == "dispatch_output:0"
    assert result["moe_mlp_stage_summaries"][0]["sum"]["abs_diff_max"] == 0.25
    assert {
        (entry["phase"], entry["check"], entry["exact_fraction"])
        for entry in result["metadata_summaries"]
    } >= {("route", "recorded_topk_to_target_topk", 1.0)}
    assert result["replay_target_map_summaries"] == [
        {
            "layer": 1,
            "phase": "route",
            "compared_ranks": 1,
            "zero_mismatch_fraction": 1.0,
            "mismatch_count_max": 0,
            "mismatch_count_mean": 0.0,
            "nonzero_mismatch_ranks": [],
        }
    ]
