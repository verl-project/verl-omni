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

"""Regression coverage for the unified LingBot offline prompt-rewrite driver.

``rewrite_prompts.py`` merges the former vLLM and transformers drivers behind a
single ``--backend`` switch, so these tests exercise the shared IO/record layer
and the vLLM HTTP backend without importing the optional 27B model stack.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT_DIR = _REPO_ROOT / "examples" / "flowgrpo_trainer" / "lingbot_video"


def _load_script(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_DIR / f"{module_name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_script("rewrite_prompts")


def test_prompt_loading_keeps_chinese_and_validates_shards(driver, tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "an astronaut walking\n一只猫在云中飞行\nan astronaut walking\n\n",
        encoding="utf-8",
    )

    # Duplicate lines and blank lines are dropped; Chinese prompts are preserved.
    assert driver._load_prompts(str(prompts), num_shards=1, shard=0) == [
        "an astronaut walking",
        "一只猫在云中飞行",
    ]
    assert driver._load_prompts(str(prompts), num_shards=2, shard=1) == ["一只猫在云中飞行"]
    with pytest.raises(ValueError, match="num_shards"):
        driver._load_prompts(str(prompts), num_shards=0, shard=0)
    with pytest.raises(ValueError, match="shard"):
        driver._load_prompts(str(prompts), num_shards=1, shard=1)


def test_already_done_reads_prompt_raw_for_resume(driver, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    # A valid record, a blank line, and three lines the resume must skip without
    # crashing: a non-JSON line (JSONDecodeError), a JSON object missing the key
    # (KeyError), and a valid-JSON non-object (TypeError on subscription).
    out.write_text(
        json.dumps({"prompt_raw": "done prompt", "caption": {}})
        + "\n\nnot json\n"
        + json.dumps({"caption": {}})
        + "\n"
        + json.dumps([1, 2, 3])
        + "\n",
        encoding="utf-8",
    )
    assert driver._already_done(str(out)) == {"done prompt"}
    assert driver._already_done(str(tmp_path / "missing.jsonl")) == set()


def test_build_record_matches_prepare_structured_captions_schema(driver) -> None:
    args = SimpleNamespace(duration=5.4, data_source="dance_grpo/hpsv3")
    caption = {"comprehensive_description": "a cat", "camera_info": "static"}
    record = driver.build_record("a cat", caption, args)
    assert record == {
        "prompt_raw": "a cat",
        "caption": caption,
        "duration": 5,  # rounded to int
        "data_source": "dance_grpo/hpsv3",
        "reward_model": {"style": "model", "ground_truth": "a cat"},
    }


def test_process_one_treats_non_dict_caption_as_empty(driver) -> None:
    args = SimpleNamespace(mode="t2v", duration=5.0, data_source="src")

    class DictRewriter:
        def rewrite(self, prompt, mode, duration):
            return {"json": {"comprehensive_description": prompt}}

    class PlainRewriter:
        def rewrite(self, prompt, mode, duration):
            return {"json": "not structured"}

    record = driver._process_one(DictRewriter(), "a dog", args)
    assert record["caption"] == {"comprehensive_description": "a dog"}
    # A non-dict step-2 output is counted as "empty", never written.
    assert driver._process_one(PlainRewriter(), "a dog", args) is None


def test_vllm_backend_preserves_official_base_then_lora_selection(driver) -> None:
    requests = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "rewritten"}}]}

    class Session:
        def post(self, url: str, *, json: dict, timeout: float) -> Response:
            requests.append((url, json, timeout))
            return Response()

    backend = driver.VLLMBackend("http://rewriter:8137", base_model="base", lora_model="rewriter", timeout=12)
    backend._local.session = Session()

    assert backend.generate("expand", image=None, use_lora=False) == "rewritten"
    assert backend.generate("map", image=None, use_lora=True) == "rewritten"
    # step1 EXPAND selects the base model id, step2 MAP selects the LoRA id.
    assert [payload[1]["model"] for payload in requests] == ["base", "rewriter"]
    assert all(payload[1]["temperature"] == 0.0 for payload in requests)
    assert all(payload[1]["chat_template_kwargs"] == {"enable_thinking": False} for payload in requests)
    assert all(payload[2] == 12 for payload in requests)


def test_vllm_backend_rejects_image_inputs(driver) -> None:
    backend = driver.VLLMBackend("http://rewriter:8137")
    with pytest.raises(NotImplementedError, match="text-only"):
        backend.generate("map", image=object(), use_lora=True)
