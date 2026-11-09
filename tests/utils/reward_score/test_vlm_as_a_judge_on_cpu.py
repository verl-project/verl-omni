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
"""CPU tests for the unified VLM-as-a-judge scorers."""

import hashlib
import inspect
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from verl.utils.import_utils import load_extern_object

from verl_omni.utils.reward_score import genrm_ocr, reward_utils, unified_reward, vlm_as_a_judge

REPO_ROOT = Path(__file__).parents[3]
STUB_BASE64 = "data:image/png;base64,STUB"
OCR_MODEL_OUTPUT = "hello"
UNIFIED_MODEL_OUTPUT = "Alignment Score (1-5): 4\nCoherence Score (1-5): 5\nStyle Score (1-5): 3"
OCR_USER_PROMPT = (
    "Please output only the text content from the image without any additional descriptions or formatting."
)
UNIFIED_USER_TEXT = "a cat"
UNIFIED_EXPECTED_PROMPT = (
    "You are presented with a generated image and its associated text caption. "
    "Your task is to analyze the image across multiple dimensions in relation to the caption. "
    "Specifically:\n"
    "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is "
    "(absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [a cat]"
)

# Captured from origin/main (017658d) by patching the historical transport with the
# same stub image and model name. The lengths and digests pin the exact JSON bytes.
OCR_EXPECTED_REQUEST = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": STUB_BASE64}},
                {"type": "text", "text": OCR_USER_PROMPT},
            ],
        },
    ],
    "model": "judge-model",
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 4096,
}
OCR_EXPECTED_LENGTH = 392
OCR_EXPECTED_SHA256 = "84335dbe74bc935ef5992cfdee9dd8f93537d7aff6c13aa156a6665cf17f40b1"

UNIFIED_EXPECTED_REQUEST = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": STUB_BASE64}},
                {"type": "text", "text": UNIFIED_EXPECTED_PROMPT},
            ],
        }
    ],
    "model": "judge-model",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}
UNIFIED_EXPECTED_LENGTH = 970
UNIFIED_EXPECTED_SHA256 = "953236e1e0424539986d3a2373761362958218ba89f774e54a86a6fd678b8a6b"

OCR_SCORERS = [
    pytest.param(vlm_as_a_judge.compute_score_ocr, id="shared"),
    pytest.param(genrm_ocr.compute_score_ocr, id="legacy"),
]
UNIFIED_SCORERS = [
    pytest.param(vlm_as_a_judge.compute_score_unified_reward, id="shared"),
    pytest.param(unified_reward.compute_score_unified_reward, id="legacy"),
]
EXTERNAL_SCORERS = [
    pytest.param("verl_omni/utils/reward_score/vlm_as_a_judge.py", "compute_score_ocr", id="shared-ocr-file"),
    pytest.param(
        "verl_omni/utils/reward_score/vlm_as_a_judge.py",
        "compute_score_unified_reward",
        id="shared-unified-file",
    ),
    pytest.param("verl_omni/utils/reward_score/genrm_ocr.py", "compute_score_ocr", id="legacy-ocr-file"),
    pytest.param(
        "verl_omni/utils/reward_score/unified_reward.py",
        "compute_score_unified_reward",
        id="legacy-unified-file",
    ),
]


@pytest.fixture(autouse=True)
def _stub_image_encoding(monkeypatch):
    monkeypatch.setattr(reward_utils, "pil_image_to_base64", lambda image: STUB_BASE64)


def _image() -> torch.Tensor:
    return torch.zeros(3, 8, 8, dtype=torch.uint8)


def _video() -> torch.Tensor:
    return torch.zeros(2, 3, 8, 8, dtype=torch.uint8)


def _capture(monkeypatch, *, sessions: list | None = None) -> list[dict]:
    requests: list[dict] = []

    async def transport(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        if sessions is not None:
            sessions.append(session)
        content = UNIFIED_MODEL_OUTPUT if chat_complete_request["messages"][0]["role"] == "user" else OCR_MODEL_OUTPUT
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(vlm_as_a_judge, "_post_chat_completion", transport)
    return requests


async def _score_ocr(
    scorer,
    *,
    user_prompt=None,
    model_name="judge-model",
    sampling_params=None,
    extra_info=None,
    solution_image=None,
):
    kwargs = {}
    if user_prompt is not None:
        kwargs["user_prompt"] = user_prompt
    if sampling_params is not None:
        kwargs["sampling_params"] = sampling_params
    return await scorer(
        data_source="test",
        solution_image=_image() if solution_image is None else solution_image,
        ground_truth="hello",
        extra_info=extra_info or {"media_kind": "image"},
        reward_router_address="judge:8000",
        model_name=model_name,
        **kwargs,
    )


async def _score_unified(
    scorer,
    *,
    user_prompt=None,
    model_name="judge-model",
    sampling_params=None,
    extra_info=None,
    solution_image=None,
):
    kwargs = {}
    if user_prompt is not None:
        kwargs["user_prompt"] = user_prompt
    if sampling_params is not None:
        kwargs["sampling_params"] = sampling_params
    return await scorer(
        data_source="test",
        solution_image=_image() if solution_image is None else solution_image,
        ground_truth=UNIFIED_USER_TEXT,
        extra_info=extra_info or {"media_kind": "image"},
        reward_router_address="judge:8000",
        model_name=model_name,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", OCR_SCORERS)
async def test_ocr_default_request_matches_main_bytes(monkeypatch, scorer):
    requests = _capture(monkeypatch)

    result = await _score_ocr(scorer)

    encoded = json.dumps(requests[0]).encode("utf-8")
    assert requests[0] == OCR_EXPECTED_REQUEST
    assert len(encoded) == OCR_EXPECTED_LENGTH
    assert hashlib.sha256(encoded).hexdigest() == OCR_EXPECTED_SHA256
    assert result == {"score": 1.0, "genrm_response": OCR_MODEL_OUTPUT}


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", UNIFIED_SCORERS)
async def test_unified_default_request_matches_main_bytes(monkeypatch, scorer):
    requests = _capture(monkeypatch)

    result = await _score_unified(scorer)

    encoded = json.dumps(requests[0]).encode("utf-8")
    assert requests[0] == UNIFIED_EXPECTED_REQUEST
    assert len(encoded) == UNIFIED_EXPECTED_LENGTH
    assert hashlib.sha256(encoded).hexdigest() == UNIFIED_EXPECTED_SHA256
    assert result["score"] == pytest.approx(0.75)
    assert result["raw_score"] == pytest.approx(4.0)
    assert result["reward_extra_info"] == {}


@pytest.mark.asyncio
async def test_ocr_runs_one_request_per_frame_without_shared_session(monkeypatch):
    sessions = []
    _capture(monkeypatch, sessions=sessions)

    result = await _score_ocr(
        vlm_as_a_judge.compute_score_ocr,
        solution_image=_video(),
        extra_info={"media_kind": "video", "fps": 8, "frame_interval": 1},
    )

    assert sessions == [None, None]
    assert result["score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_unified_reuses_one_session_for_all_frames(monkeypatch):
    sessions = []
    _capture(monkeypatch, sessions=sessions)

    result = await _score_unified(
        vlm_as_a_judge.compute_score_unified_reward,
        solution_image=_video(),
        extra_info={"media_kind": "video", "fps": 8},
    )

    assert len(sessions) == 2
    assert sessions[0] is not None
    assert sessions[0] is sessions[1]
    assert result["score"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_generic_frame_scorer_builds_the_shared_request(monkeypatch):
    requests = _capture(monkeypatch)

    response = await vlm_as_a_judge._score_vlm_frame(
        _image(),
        "transcribe this",
        router_address="judge:8000",
        model_name="judge-model",
        sampling_params={"temperature": 0.1},
        system_prompt="You are terse.",
    )

    assert response == OCR_MODEL_OUTPUT
    assert requests == [
        {
            "messages": [
                {"role": "system", "content": "You are terse."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": STUB_BASE64}},
                        {"type": "text", "text": "transcribe this"},
                    ],
                },
            ],
            "model": "judge-model",
            "temperature": 0.1,
        }
    ]


def test_single_transport_and_scorer_implementation():
    assert not hasattr(vlm_as_a_judge, "_chat_complete")
    assert not hasattr(vlm_as_a_judge, "_chat_complete_with_session")
    assert not hasattr(vlm_as_a_judge, "_compute_score_ocr_impl")
    assert not hasattr(vlm_as_a_judge, "_compute_score_unified_reward_impl")
    assert inspect.iscoroutinefunction(vlm_as_a_judge._post_chat_completion)

    assert genrm_ocr.compute_score_ocr is vlm_as_a_judge.compute_score_ocr
    assert unified_reward.compute_score_unified_reward is vlm_as_a_judge.compute_score_unified_reward


def test_public_signatures_are_unchanged():
    ocr_parameters = [
        ("data_source", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("solution_image", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("ground_truth", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("extra_info", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("reward_router_address", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("reward_model_tokenizer", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("model_name", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("sampling_params", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("user_prompt", inspect.Parameter.KEYWORD_ONLY, None),
    ]
    unified_parameters = list(ocr_parameters)

    for function in (vlm_as_a_judge.compute_score_ocr, genrm_ocr.compute_score_ocr):
        signature = inspect.signature(function)
        actual = [(parameter.name, parameter.kind, parameter.default) for parameter in signature.parameters.values()]
        assert actual == ocr_parameters
        assert "system_prompt" not in signature.parameters
        assert "prompt" not in signature.parameters
    for function in (vlm_as_a_judge.compute_score_unified_reward, unified_reward.compute_score_unified_reward):
        signature = inspect.signature(function)
        actual = [(parameter.name, parameter.kind, parameter.default) for parameter in signature.parameters.values()]
        assert actual == unified_parameters
        assert "system_prompt" not in signature.parameters
        assert "prompt" not in signature.parameters

    assert inspect.signature(vlm_as_a_judge.compute_score_ocr) == inspect.signature(genrm_ocr.compute_score_ocr)
    assert inspect.signature(vlm_as_a_judge.compute_score_unified_reward) == inspect.signature(
        unified_reward.compute_score_unified_reward
    )


def test_sampling_params_merge_defaults_without_mutating_them():
    assert vlm_as_a_judge._sampling_params() == vlm_as_a_judge.DEFAULT_SAMPLING_PARAMS
    assert vlm_as_a_judge._sampling_params({"temperature": 0.2, "seed": 9}) == {
        **vlm_as_a_judge.DEFAULT_SAMPLING_PARAMS,
        "temperature": 0.2,
        "seed": 9,
    }
    assert vlm_as_a_judge.DEFAULT_SAMPLING_PARAMS == {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 4096,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", OCR_SCORERS)
async def test_ocr_sampling_params_override_defaults(monkeypatch, scorer):
    requests = _capture(monkeypatch)

    await _score_ocr(scorer, sampling_params={"temperature": 0.2, "max_tokens": 17, "seed": 9})

    assert requests[0] == {
        **OCR_EXPECTED_REQUEST,
        "temperature": 0.2,
        "max_tokens": 17,
        "seed": 9,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", OCR_SCORERS)
async def test_ocr_none_judge_response_scores_as_empty(monkeypatch, scorer):
    async def transport(router_address, chat_complete_request, session=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    monkeypatch.setattr(vlm_as_a_judge, "_post_chat_completion", transport)

    result = await _score_ocr(scorer)

    assert result == {"score": 0.0, "genrm_response": ""}


@pytest.mark.asyncio
async def test_genrm_ocr_environment_variables_are_ignored(monkeypatch):
    requests = _capture(monkeypatch)
    monkeypatch.setenv("GENRM_OCR_TEMPERATURE", "0.0")
    monkeypatch.setenv("GENRM_OCR_TOP_P", "1.0")
    monkeypatch.setenv("GENRM_OCR_MAX_TOKENS", "123")
    monkeypatch.setenv("GENRM_OCR_SEED", "777")

    await _score_ocr(vlm_as_a_judge.compute_score_ocr)

    assert requests[0] == OCR_EXPECTED_REQUEST


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", UNIFIED_SCORERS)
async def test_unified_sampling_params_override_defaults(monkeypatch, scorer):
    requests = _capture(monkeypatch)

    await _score_unified(scorer, sampling_params={"temperature": 0.9, "max_tokens": 17})

    assert requests[0]["temperature"] == 0.9
    assert requests[0]["top_p"] == 1.0
    assert requests[0]["max_tokens"] == 17
    assert "seed" not in requests[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", OCR_SCORERS)
async def test_ocr_user_prompt_literal_and_package(monkeypatch, scorer):
    requests = _capture(monkeypatch)

    await _score_ocr(scorer, user_prompt="Transcribe only the largest word.")
    await _score_ocr(
        scorer,
        user_prompt="pkg://tests.utils.reward_score.test_vlm_as_a_judge_on_cpu.build_ocr_prompt",
    )

    assert requests[0]["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert requests[0]["messages"][1]["content"][1]["text"] == "Transcribe only the largest word."
    assert requests[1]["messages"][1]["content"][1]["text"] == "Transcribe every visible word."


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", UNIFIED_SCORERS)
async def test_unified_user_prompt_literal_and_package(monkeypatch, scorer):
    requests = _capture(monkeypatch)

    await _score_unified(scorer, user_prompt="Score {caption} in one line. Caption: {caption}")
    await _score_unified(
        scorer,
        user_prompt="pkg://tests.utils.reward_score.test_vlm_as_a_judge_on_cpu.build_unified_prompt",
    )

    assert requests[0]["messages"][0]["content"][1]["text"] == ("Score a cat in one line. Caption: a cat")
    assert requests[1]["messages"][0]["content"][1]["text"] == "Judge the image against: a cat"


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", OCR_SCORERS)
async def test_ocr_default_model_path_does_not_depend_on_real_home(monkeypatch, tmp_path, scorer):
    requests = _capture(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    await _score_ocr(scorer, model_name=None)

    assert requests[0]["model"] == str(tmp_path / "models/tiny-random/qwen3-vl")


@pytest.mark.asyncio
async def test_unified_default_model_path_is_unchanged(monkeypatch):
    requests = _capture(monkeypatch)

    await _score_unified(vlm_as_a_judge.compute_score_unified_reward, model_name=None)

    assert requests[0]["model"] == "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b"


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", OCR_SCORERS + UNIFIED_SCORERS)
async def test_scorers_do_not_emit_deprecation_warnings(monkeypatch, scorer):
    _capture(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if scorer.__name__ == "compute_score_ocr":
            await _score_ocr(scorer)
        else:
            await _score_unified(scorer)

    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


@pytest.mark.parametrize(
    "module_path, object_name",
    [
        ("pkg://verl_omni.utils.reward_score.vlm_as_a_judge", "compute_score_ocr"),
        ("pkg://verl_omni.utils.reward_score.vlm_as_a_judge", "compute_score_unified_reward"),
        ("pkg://verl_omni.utils.reward_score.genrm_ocr", "compute_score_ocr"),
        ("pkg://verl_omni.utils.reward_score.unified_reward", "compute_score_unified_reward"),
    ],
)
def test_pkg_entrypoints_are_callable(module_path, object_name):
    scorer = load_extern_object(module_path, object_name)

    assert inspect.iscoroutinefunction(scorer)


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path, object_name", EXTERNAL_SCORERS)
async def test_file_entrypoints_are_callable(monkeypatch, module_path, object_name):
    monkeypatch.chdir(REPO_ROOT)
    scorer = load_extern_object("file://" + module_path, object_name)
    requests = []

    async def transport(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        content = UNIFIED_MODEL_OUTPUT if chat_complete_request["messages"][0]["role"] == "user" else "hello"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    assert inspect.iscoroutinefunction(scorer)
    # The file-path loader returns only the object; its module globals expose the
    # shared transport used by both the new module and the compatibility wrappers.
    monkeypatch.setitem(scorer.__globals__, "_post_chat_completion", transport)
    if object_name == "compute_score_ocr":
        await scorer(
            data_source="test",
            solution_image=_image(),
            ground_truth="hello",
            extra_info={"media_kind": "image"},
            reward_router_address="judge:8000",
            model_name="judge-model",
        )
    else:
        await scorer(
            data_source="test",
            solution_image=_image(),
            ground_truth=UNIFIED_USER_TEXT,
            extra_info={"media_kind": "image"},
            reward_router_address="judge:8000",
            model_name="judge-model",
        )
    assert requests


def build_ocr_prompt() -> str:
    return "Transcribe every visible word."


def build_unified_prompt(caption: str) -> str:
    return f"Judge the image against: {caption}"
