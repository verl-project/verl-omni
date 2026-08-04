# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the Qwen3-Omni MiniCPM-o judge evaluation script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "examples" / "dpo_trainer" / "qwen3_omni" / "compare_ref_vs_trained_minicpm_judge.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("compare_ref_vs_trained_minicpm_judge", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


judge_mod = _load_script_module()


def test_build_multimodal_content_requires_local_media_files(tmp_path):
    image_path = tmp_path / "sample.png"
    video_path = tmp_path / "sample.mp4"
    audio_path = tmp_path / "sample.wav"
    image_path.write_bytes(b"fake-image")
    video_path.write_bytes(b"fake-video")
    audio_path.write_bytes(b"fake-audio")
    sample = judge_mod.EvalSample(
        data_file="data.jsonl",
        index=0,
        uid="sample-0",
        modality="image",
        prompt_text="Describe the content.",
        media={
            "images": [str(image_path)],
            "videos": [str(video_path)],
            "audios": [str(audio_path)],
        },
        raw_prompt=[],
    )

    content = judge_mod.build_multimodal_content(sample)

    assert [item["type"] for item in content] == ["image_url", "video_url", "audio_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert content[2]["audio_url"]["url"].startswith("data:audio/")


def test_build_multimodal_content_fails_when_declared_media_is_missing():
    sample = judge_mod.EvalSample(
        data_file="data.jsonl",
        index=0,
        uid="sample-0",
        modality="video",
        prompt_text="Describe the clip.",
        media={"images": [], "videos": [], "audios": []},
        raw_prompt=[],
    )

    with pytest.raises(ValueError, match="has no videos media"):
        judge_mod.build_multimodal_content(sample)


def test_parse_judge_response_maps_randomized_labels_back_to_models():
    response = json.dumps(
        {
            "A": {
                "overall_score": 8,
                "fluency": 8,
                "relevance": 8,
                "accuracy": 7,
                "reasoning_quality": 8,
                "safety": 9,
            },
            "B": {
                "overall_score": 6,
                "fluency": 6,
                "relevance": 6,
                "accuracy": 6,
                "reasoning_quality": 6,
                "safety": 6,
            },
            "winner": "A",
            "rationale": "A is better grounded.",
        }
    )

    result = judge_mod.parse_judge_response(response, {"A": "trained", "B": "reference"})

    assert result.winner == "trained"
    assert result.trained_score == 8
    assert result.reference_score == 6
    assert result.trained_dimension_scores["accuracy"] == 7


def test_run_uses_mock_generation_and_judge_endpoints(tmp_path, monkeypatch):
    data_path = tmp_path / "eval.jsonl"
    rows = [
        {
            "uid": "row-0",
            "data_source": "omni_preference/image",
            "prompt": [{"role": "user", "content": "<image>What is shown?"}],
            "images": ["https://example.com/image.png"],
            "chosen": "good",
            "rejected": "bad",
        },
        {
            "uid": "row-1",
            "data_source": "omni_preference/audio",
            "prompt": [{"role": "user", "content": "<audio>What is said?"}],
            "audios": ["https://example.com/audio.wav"],
            "chosen": "good",
            "rejected": "bad",
        },
    ]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    calls = []

    def fake_wait_for_server(*_args, **_kwargs):
        return None

    def fake_post(_router_address, payload):
        calls.append(payload)
        model = payload["model"]
        if model == "reference":
            return {"choices": [{"message": {"content": "reference answer"}}]}
        if model == "trained":
            return {"choices": [{"message": {"content": "trained answer"}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "A": {"overall_score": 4},
                                "B": {"overall_score": 9},
                                "winner": "B",
                                "rationale": "B is better.",
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(judge_mod, "wait_for_server", fake_wait_for_server)
    monkeypatch.setattr(judge_mod, "_post_chat_completion", fake_post)

    args = SimpleNamespace(
        data_files=[str(data_path)],
        output_jsonl=str(tmp_path / "out.jsonl"),
        max_samples=-1,
        seed=1,
        model_path="base",
        adapter_path="adapter",
        generation_router_address="127.0.0.1:8000",
        trained_generation_router_address=None,
        reference_model_name="reference",
        trained_model_name="trained",
        launch_generation_server=False,
        generation_server_host="127.0.0.1",
        generation_server_port=8000,
        generation_server_command=judge_mod.DEFAULT_QWEN_SERVER_COMMAND,
        generation_max_tokens=16,
        generation_temperature=0.0,
        generation_top_p=1.0,
        judge_router_address="127.0.0.1:8001",
        judge_model="judge",
        launch_judge_server=False,
        judge_server_host="127.0.0.1",
        judge_server_port=8001,
        judge_server_command=judge_mod.DEFAULT_JUDGE_SERVER_COMMAND,
        judge_max_tokens=32,
        judge_temperature=0.0,
        server_timeout_s=1.0,
    )

    summary = judge_mod.run(args)

    assert summary["overall"]["total"] == 2
    assert summary["overall"]["trained_wins"] == 1
    assert summary["overall"]["reference_wins"] == 1
    assert summary["by_modality"]["image"]["total"] == 1
    assert summary["by_modality"]["audio"]["total"] == 1
    assert Path(args.output_jsonl).read_text(encoding="utf-8").count("\n") == 2
    assert len(calls) == 6
