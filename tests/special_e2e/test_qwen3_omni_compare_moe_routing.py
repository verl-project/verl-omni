import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen3_omni_compare_moe_routing.py")


def _row(sample, layer, response_index, token, expert_ids, expert_probs, **extra):
    row = {
        "input_ids_sha256": sample,
        "layer": layer,
        "response_index": response_index,
        "input_token_id": token,
        "expert_ids": expert_ids,
        "expert_probs": expert_probs,
    }
    row.update(extra)
    return row


def test_compare_moe_routing_reports_exact_routes_and_disagreements(tmp_path):
    sample = "sample-a"
    hf = tmp_path / "hf.jsonl"
    hf.write_text(
        json.dumps(
            {
                "event": "hf_rollout_corr_score_row",
                "input_ids_sha256": sample,
                "moe_router_audit": [
                    _row(sample, 4, 0, 42, [3, 1], [0.7, 0.3]),
                    _row(sample, 4, 1, 43, [2, 1], [0.6, 0.4]),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    megatron = tmp_path / "megatron.rank0.jsonl"
    megatron.write_text(
        json.dumps(
            {
                "event": "megatron_decoder_component_audit",
                "moe_router_audit": [
                    _row(sample, 4, 0, 42, [1, 3], [0.3, 0.7]),
                    _row(sample, 4, 1, 43, [2, 5], [0.6, 0.4]),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hf-jsonl",
            str(hf),
            "--megatron-decoder-glob",
            str(tmp_path / "megatron.rank*.jsonl"),
            "--output-file",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["token_alignment"] == {"matched": 2, "missing": 0, "token_mismatched": 0}
    assert result["summaries"] == [
        {
            "layer": 4,
            "matched_tokens": 2,
            "route_exact_fraction": 0.5,
            "prob_abs_diff_mean": 0.0,
            "prob_abs_diff_max": 0.0,
        }
    ]
    assert result["route_disagreements"][0]["megatron_expert_ids"] == [2, 5]


def test_compare_moe_routing_normalizes_hf_full_softmax_over_selected_experts(tmp_path):
    sample = "sample-b"
    hf = tmp_path / "hf.jsonl"
    hf.write_text(
        json.dumps(
            {
                "event": "hf_rollout_corr_score_row",
                "input_ids_sha256": sample,
                "moe_router_audit": [_row(sample, 4, 0, 42, [3, 1], [0.14, 0.06])],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    megatron = tmp_path / "megatron.rank0.jsonl"
    megatron.write_text(
        json.dumps(
            {
                "event": "megatron_decoder_component_audit",
                "moe_router_audit": [_row(sample, 4, 0, 42, [1, 3], [0.3, 0.7])],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hf-jsonl",
            str(hf),
            "--megatron-decoder-glob",
            str(tmp_path / "megatron.rank*.jsonl"),
            "--output-file",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))["summaries"][0]
    assert summary["prob_abs_diff_mean"] < 1e-12
    assert summary["prob_abs_diff_max"] < 1e-12


def test_compare_moe_routing_reports_router_boundary_and_weight_fingerprints(tmp_path):
    sample = "sample-c"
    stats = {"sum": 1.0, "square_sum": 2.0, "head": [0.25, -0.5]}
    logits = {"sum": 3.0, "square_sum": 5.0, "head": [1.0, 2.0]}
    fingerprint = {
        "layer": 4,
        "shape": [2, 2],
        "sum": 1.0,
        "square_sum": 2.0,
        "anchors": [{"index": [0, 0], "value": 0.5}],
    }
    extra = {
        "router_input_stats": stats,
        "router_logits_stats": logits,
        "router_logit_top_expert_ids": [3, 1],
        "router_logit_top_values": [2.0, 1.0],
    }
    hf = tmp_path / "hf.jsonl"
    hf.write_text(
        json.dumps(
            {
                "event": "hf_rollout_corr_score_row",
                "input_ids_sha256": sample,
                "moe_router_audit": [_row(sample, 4, 0, 42, [3, 1], [0.7, 0.3], **extra)],
                "moe_router_weight_audit": [fingerprint],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    megatron = tmp_path / "megatron.rank0.jsonl"
    megatron.write_text(
        json.dumps(
            {
                "event": "megatron_decoder_component_audit",
                "moe_router_audit": [_row(sample, 4, 0, 42, [1, 3], [0.3, 0.7], **extra)],
                "moe_router_weight_audit": [fingerprint],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hf-jsonl",
            str(hf),
            "--megatron-decoder-glob",
            str(tmp_path / "megatron.rank*.jsonl"),
            "--output-file",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["stage_summaries"] == [
        {
            "layer": 4,
            "matched_tokens": 1,
            "router_input_abs_diff_mean": 0.0,
            "router_input_abs_diff_max": 0.0,
            "router_logits_abs_diff_mean": 0.0,
            "router_logits_abs_diff_max": 0.0,
            "router_logit_top16_exact_fraction": 1.0,
            "router_logit_top16_jaccard_mean": 1.0,
            "router_logit_top16_value_abs_diff_mean": 0.0,
            "router_logit_top16_value_abs_diff_max": 0.0,
        }
    ]
    assert result["weight_summaries"] == [
        {
            "layer": 4,
            "shape_equal": True,
            "sum_abs_diff": 0.0,
            "square_sum_abs_diff": 0.0,
            "anchor_abs_diff_max": 0.0,
            "anchor_abs_diff_mean": 0.0,
        }
    ]


def test_compare_moe_routing_reports_raw_topk_margin_delta(tmp_path):
    sample = "sample-margin"
    hf = tmp_path / "hf.jsonl"
    megatron = tmp_path / "megatron.rank0.jsonl"
    hf.write_text(
        json.dumps(
            {
                "event": "hf_rollout_corr_score_row",
                "input_ids_sha256": sample,
                "moe_router_audit": [
                    _row(sample, 1, 0, 42, [3, 1], [0.7, 0.3], router_raw_topk_margin=0.02)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    megatron.write_text(
        json.dumps(
            {
                "event": "megatron_decoder_component_audit",
                "moe_router_audit": [
                    _row(sample, 1, 0, 42, [3, 1], [0.7, 0.3], router_raw_topk_margin=0.05)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hf-jsonl",
            str(hf),
            "--megatron-decoder-glob",
            str(tmp_path / "megatron.rank*.jsonl"),
            "--output-file",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))["stage_summaries"][0]
    assert abs(summary["router_raw_topk_margin_abs_diff_mean"] - 0.03) < 1e-12
    assert abs(summary["router_raw_topk_margin_abs_diff_max"] - 0.03) < 1e-12
