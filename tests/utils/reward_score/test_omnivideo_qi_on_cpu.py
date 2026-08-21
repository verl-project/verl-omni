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

import asyncio
import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def _load_reward_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/omnivideo_qi.py"
    spec = importlib.util.spec_from_file_location("omnivideo_qi_reward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reward_module_supports_unregistered_dynamic_import():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/omnivideo_qi.py"
    spec = importlib.util.spec_from_file_location("omnivideo_qi_dynamic_reward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module.compute_score)


omnivideo_qi = _load_reward_module()

VALID_RESPONSE = """<time>1.0-2.5</time><caption>A person opens the door.</caption>
<time>4.0-5.5</time><caption>The person enters the room.</caption>
<thinking>Let me think. The two events show an arrival.</thinking>
<answer>B</answer>"""


def test_parse_response_accepts_strict_ordered_groundings():
    parsed = omnivideo_qi.parse_response(VALID_RESPONSE)

    assert parsed.format_valid
    assert [(pair.start, pair.end) for pair in parsed.pairs] == [(1.0, 2.5), (4.0, 5.5)]
    assert parsed.answer == "B"


@pytest.mark.parametrize(
    "response",
    [
        VALID_RESPONSE.replace("4.0-5.5", "2.0-5.5"),
        VALID_RESPONSE.replace("1.0-2.5", "1-2.5"),
        VALID_RESPONSE.replace(
            "<thinking>Let me think. The two events show an arrival.</thinking>", "<thinking></thinking>"
        ),
        f"prefix {VALID_RESPONSE}",
    ],
)
def test_parse_response_rejects_non_strict_formats(response):
    assert not omnivideo_qi.parse_response(response).format_valid


def test_segment_decode_uses_killable_ffmpeg_subprocess(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_path = Path(command[-1].replace("%04d", "0001"))
        Image.new("RGB", (16, 12), color="red").save(output_path)

    vision_process = SimpleNamespace(
        fetch_video=lambda *args, **kwargs: pytest.fail("torchvision decoding cannot enforce a hard timeout")
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(vision_process=vision_process))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_process)
    monkeypatch.setitem(
        sys.modules,
        "verl_omni.utils.reward_score.reward_utils",
        SimpleNamespace(pil_image_to_base64=lambda image: f"data:image/png;base64,{image.size}"),
    )
    monkeypatch.setattr(omnivideo_qi, "_ffmpeg_executable", lambda: "/opt/ffmpeg", raising=False)
    monkeypatch.setattr(omnivideo_qi, "subprocess", SimpleNamespace(run=fake_run), raising=False)
    monkeypatch.setenv("OMNIVIDEO_QI_DECODE_TIMEOUT", "12.5")

    frames = omnivideo_qi._load_segment_frames(
        "/data/example.mp4",
        omnivideo_qi.GroundingPair(1.0, 2.5, "caption"),
        max_frames=8,
    )

    assert len(frames) == 1
    command, kwargs = calls[0]
    assert command[:2] == ["/opt/ffmpeg", "-hide_banner"]
    assert command[command.index("-threads") + 1] == "1"
    assert command[command.index("-frames:v") + 1] == "8"
    assert kwargs["timeout"] == 12.5


def test_short_segment_decode_samples_a_midpoint_frame(monkeypatch):
    def fake_run(command, **kwargs):
        del kwargs
        if "-t" not in command:
            output_path = Path(command[-1].replace("%04d", "0001"))
            Image.new("RGB", (16, 12), color="red").save(output_path)

    monkeypatch.setitem(
        sys.modules,
        "verl_omni.utils.reward_score.reward_utils",
        SimpleNamespace(pil_image_to_base64=lambda image: f"data:image/png;base64,{image.size}"),
    )
    monkeypatch.setattr(omnivideo_qi, "_ffmpeg_executable", lambda: "/opt/ffmpeg")
    monkeypatch.setattr(omnivideo_qi.subprocess, "run", fake_run)

    frames = omnivideo_qi._load_segment_frames(
        "/data/example.mp4",
        omnivideo_qi.GroundingPair(0.0, 0.1, "caption"),
        max_frames=8,
    )

    assert len(frames) == 1


@pytest.mark.asyncio
async def test_choice_answer_still_receives_outcome_reward_without_valid_grounding(monkeypatch):
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_URL", raising=False)

    result = await omnivideo_qi.compute_score(
        solution_str="<answer>C</answer>",
        ground_truth="C. kitchen",
        extra_info={"is_multiple_choice": True},
    )

    assert result["score"] == 1.0
    assert result["format"] == 0.0
    assert result["answer"] == 1.0
    assert result["intent"] == 0.0


@pytest.mark.asyncio
async def test_compute_score_implements_qi_equation(monkeypatch):
    config = omnivideo_qi.JudgeConfig(url="http://judge/v1/chat/completions", model="judge")
    monkeypatch.setattr(omnivideo_qi, "_judge_config", lambda *args: config)
    monkeypatch.setattr(omnivideo_qi, "_load_segment_frames", lambda *args: ["data:image/png;base64,frame"])

    async def fake_chat_complete(session, config, content, system_prompt):
        del session, config, content
        if system_prompt == omnivideo_qi.CONSISTENCY_SYSTEM_PROMPT:
            return 0.8, "consistent"
        if system_prompt == omnivideo_qi.COMPLETENESS_SYSTEM_PROMPT:
            return 0.6, "complete"
        raise AssertionError("multiple-choice answer should not call the answer judge")

    monkeypatch.setattr(omnivideo_qi, "_chat_complete", fake_chat_complete)

    result = await omnivideo_qi.compute_score(
        solution_str=VALID_RESPONSE,
        ground_truth="B. entering the room",
        extra_info={
            "is_multiple_choice": True,
            "question": "What happens?",
            "video_path": "/data/example.mp4",
        },
    )

    assert result["format"] == 1.0
    assert result["answer"] == 1.0
    assert result["consistency"] == pytest.approx(0.8)
    assert result["completeness"] == pytest.approx(0.6)
    assert result["intent"] == pytest.approx(0.7)
    assert result["score"] == pytest.approx(2.7)


@pytest.mark.asyncio
async def test_compute_score_bounds_concurrent_video_decodes_per_worker(monkeypatch):
    config = omnivideo_qi.JudgeConfig(url="http://judge/v1/chat/completions", model="judge")
    monkeypatch.setattr(omnivideo_qi, "_judge_config", lambda *args: config)
    monkeypatch.setattr(
        omnivideo_qi,
        "_SEGMENT_DECODE_SEMAPHORE",
        threading.BoundedSemaphore(2),
        raising=False,
    )
    lock = threading.Lock()
    active_decodes = 0
    peak_decodes = 0

    def fake_load_segment_frames(*args):
        nonlocal active_decodes, peak_decodes
        with lock:
            active_decodes += 1
            peak_decodes = max(peak_decodes, active_decodes)
        time.sleep(0.02)
        with lock:
            active_decodes -= 1
        return ["data:image/png;base64,frame"]

    async def fake_chat_complete(*args):
        return 1.0, "ok"

    monkeypatch.setattr(omnivideo_qi, "_load_segment_frames", fake_load_segment_frames)
    monkeypatch.setattr(omnivideo_qi, "_chat_complete", fake_chat_complete)

    await asyncio.gather(
        *[
            omnivideo_qi.compute_score(
                solution_str=VALID_RESPONSE,
                ground_truth="B. entering the room",
                extra_info={"is_multiple_choice": True, "video_path": f"/data/video-{index}.mp4"},
            )
            for index in range(4)
        ]
    )

    assert peak_decodes == 2


def test_judge_config_normalizes_openai_base_url(monkeypatch):
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_URL", "http://judge:8000/v1")
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_MODEL", "local-judge")

    config = omnivideo_qi._judge_config(reward_model_tokenizer=SimpleNamespace(name_or_path="ignored"))

    assert config.url == "http://judge:8000/v1/chat/completions"
    assert config.model == "local-judge"


def test_judge_config_uses_resource_efficient_default(monkeypatch):
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_URL", "http://judge:8000/v1")
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_MODEL", raising=False)

    config = omnivideo_qi._judge_config()

    assert config.model == "Qwen/Qwen3-VL-8B-Instruct"


def test_judge_config_uses_bounded_default_timeout(monkeypatch):
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_URL", "http://judge:8000/v1")
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_TIMEOUT", raising=False)

    config = omnivideo_qi._judge_config()

    assert config.timeout_seconds == 60.0


def test_judge_config_uses_colocated_reward_router(monkeypatch):
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_URL", raising=False)
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_MODEL", raising=False)

    config = omnivideo_qi._judge_config(
        reward_router_address="reward-router:9000",
        reward_model_tokenizer=SimpleNamespace(name_or_path="/models/qwen3-vl-30b"),
    )

    assert config.url == "http://reward-router:9000/v1/chat/completions"
    assert config.model == "/models/qwen3-vl-30b"
