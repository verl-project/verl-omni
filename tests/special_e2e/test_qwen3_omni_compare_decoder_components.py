import json
import subprocess
import sys
from pathlib import Path


def _component(layer: int, token_id: int) -> dict:
    return {
        "layer": layer,
        "component": "layer_input",
        "response_index": 0,
        "model_position": 7,
        "input_ids_sha256": "sample",
        "input_token_id": token_id,
        "stats": {"sum": 1.0, "square_sum": 2.0, "head": [0.1, 0.2]},
    }


def test_decoder_component_compare_rejects_misaligned_tokens(tmp_path):
    script = Path(__file__).with_name("qwen3_omni_compare_decoder_components.py")
    hf_path = tmp_path / "hf.jsonl"
    decoder_path = tmp_path / "decoder.rank0.jsonl"
    output_path = tmp_path / "comparison.json"
    hf_path.write_text(
        json.dumps(
            {
                "event": "hf_rollout_corr_score_row",
                "input_ids_sha256": "sample",
                "decoder_component_audit": [_component(1, 42), _component(2, 43)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    decoder_path.write_text(
        json.dumps(
            {
                "event": "megatron_decoder_component_audit",
                "topology": {},
                "decoder_component_audit": [_component(1, 42), _component(2, 99)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--hf-jsonl",
            str(hf_path),
            "--megatron-decoder-glob",
            str(tmp_path / "decoder.rank*.jsonl"),
            "--output-file",
            str(output_path),
        ],
        check=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["token_alignment"] == {"matched": 1, "mismatched": 1, "missing": 0}
    assert [(item["layer"], item["component"]) for item in result["summaries"]] == [(1, "layer_input")]
