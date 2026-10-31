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
import importlib.util
import inspect
import json
import os
import random
import sys
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

QWEN_EXPECTED_PROMPT_LENGTH = 805
QWEN_EXPECTED_PROMPT_SHA256 = "46e95eb1834f909f0198d45acbfd2c7e9076feca3f927ac656b4771e66ba0eb1"
QWEN_EXPECTED_REQUEST_LENGTH = 1078
QWEN_EXPECTED_REQUEST_SHA256 = "7639b5dbe90f2d2480e90e9f83a49615f760ffe265dbf37f758d45639ed9660a"
QWEN_VALID_RESPONSE = (
    '{"A": {"overall_score": 7, "fluency": 7, "relevance": 8, "accuracy": 7, '
    '"reasoning_quality": 6, "safety": 9}, "B": {"overall_score": 5, "fluency": 5, '
    '"relevance": 5, "accuracy": 5, "reasoning_quality": 5, "safety": 5}, '
    '"winner": "A", "rationale": "more specific"}'
)

OCR_SCORERS = [
    pytest.param(vlm_as_a_judge, "compute_score_ocr", "_chat_complete", id="new"),
    pytest.param(genrm_ocr, "compute_score_ocr", "_chat_complete", id="legacy"),
]
UNIFIED_SCORERS = [
    pytest.param(vlm_as_a_judge, "compute_score_unified_reward", "_chat_complete_with_session", id="new"),
    pytest.param(unified_reward, "compute_score_unified_reward", "_chat_complete", id="legacy"),
]
EXTERNAL_SCORERS = [
    pytest.param(
        "verl_omni/utils/reward_score/vlm_as_a_judge.py",
        "compute_score_ocr",
        "_chat_complete",
        id="new-ocr-file",
    ),
    pytest.param(
        "verl_omni/utils/reward_score/vlm_as_a_judge.py",
        "compute_score_unified_reward",
        "_chat_complete_with_session",
        id="new-unified-file",
    ),
    pytest.param(
        "verl_omni/utils/reward_score/genrm_ocr.py",
        "compute_score_ocr",
        "_chat_complete",
        id="legacy-ocr-file",
    ),
    pytest.param(
        "verl_omni/utils/reward_score/unified_reward.py",
        "compute_score_unified_reward",
        "_chat_complete",
        id="legacy-unified-file",
    ),
]


@pytest.fixture(autouse=True)
def _clean_sampling_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("GENRM_OCR_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _stub_image_encoding(monkeypatch):
    monkeypatch.setattr(reward_utils, "pil_image_to_base64", lambda image: STUB_BASE64)


def _image() -> torch.Tensor:
    return torch.zeros(3, 8, 8, dtype=torch.uint8)


def _video() -> torch.Tensor:
    return torch.zeros(2, 3, 8, 8, dtype=torch.uint8)


def _capture(monkeypatch, owner, attribute: str, *, sessions: list | None = None) -> list[dict]:
    requests: list[dict] = []

    async def transport(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        if sessions is not None:
            sessions.append(session)
        content = UNIFIED_MODEL_OUTPUT if chat_complete_request["messages"][0]["role"] == "user" else "hello"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(owner, attribute, transport)
    return requests


async def _score_ocr(module, *, user_prompt=None, model_name="judge-model", sampling_params=None, extra_info=None):
    kwargs = {}
    if user_prompt is not None:
        kwargs["user_prompt"] = user_prompt
    if sampling_params is not None:
        kwargs["sampling_params"] = sampling_params
    return await module.compute_score_ocr(
        data_source="test",
        solution_image=_image(),
        ground_truth="hello",
        extra_info=extra_info or {"media_kind": "image"},
        reward_router_address="judge:8000",
        model_name=model_name,
        **kwargs,
    )


async def _score_unified(module, *, user_prompt=None, model_name="judge-model", extra_info=None):
    kwargs = {}
    if user_prompt is not None:
        kwargs["user_prompt"] = user_prompt
    return await module.compute_score_unified_reward(
        data_source="test",
        solution_image=_image(),
        ground_truth=UNIFIED_USER_TEXT,
        extra_info=extra_info or {"media_kind": "image"},
        reward_router_address="judge:8000",
        model_name=model_name,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("module, function_name, transport_name", OCR_SCORERS)
async def test_ocr_default_request_matches_baseline_bytes(monkeypatch, module, function_name, transport_name):
    requests = _capture(monkeypatch, module, transport_name)

    result = await _score_ocr(module)

    encoded = json.dumps(requests[0]).encode("utf-8")
    assert requests[0] == OCR_EXPECTED_REQUEST
    assert len(encoded) == OCR_EXPECTED_LENGTH
    assert hashlib.sha256(encoded).hexdigest() == OCR_EXPECTED_SHA256
    assert result == {"score": 1.0, "genrm_response": OCR_MODEL_OUTPUT}


@pytest.mark.asyncio
@pytest.mark.parametrize("module, function_name, transport_name", UNIFIED_SCORERS)
async def test_unified_default_request_matches_baseline_bytes(monkeypatch, module, function_name, transport_name):
    requests = _capture(monkeypatch, module, transport_name)

    result = await _score_unified(module)

    encoded = json.dumps(requests[0]).encode("utf-8")
    assert requests[0] == UNIFIED_EXPECTED_REQUEST
    assert len(encoded) == UNIFIED_EXPECTED_LENGTH
    assert hashlib.sha256(encoded).hexdigest() == UNIFIED_EXPECTED_SHA256
    assert result["score"] == pytest.approx(0.75)
    assert result["raw_score"] == pytest.approx(4.0)
    assert result["reward_extra_info"] == {}


@pytest.mark.asyncio
async def test_unified_reuses_one_session_for_all_frames(monkeypatch):
    sessions = []
    _capture(monkeypatch, vlm_as_a_judge, "_chat_complete_with_session", sessions=sessions)

    result = await vlm_as_a_judge.compute_score_unified_reward(
        data_source="test",
        solution_image=_video(),
        ground_truth=UNIFIED_USER_TEXT,
        extra_info={"media_kind": "video", "fps": 8},
        reward_router_address="judge:8000",
        model_name="judge-model",
    )

    assert len(sessions) == 2
    assert sessions[0] is not None
    assert sessions[0] is sessions[1]
    assert result["score"] == pytest.approx(0.75)


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
    unified_parameters = [parameter for parameter in ocr_parameters if parameter[0] != "sampling_params"]

    for module in (vlm_as_a_judge, genrm_ocr):
        signature = inspect.signature(module.compute_score_ocr)
        actual = [(parameter.name, parameter.kind, parameter.default) for parameter in signature.parameters.values()]
        assert actual == ocr_parameters
        assert "system_prompt" not in signature.parameters
        assert "prompt" not in signature.parameters
    for module in (vlm_as_a_judge, unified_reward):
        signature = inspect.signature(module.compute_score_unified_reward)
        actual = [(parameter.name, parameter.kind, parameter.default) for parameter in signature.parameters.values()]
        assert actual == unified_parameters
        assert "system_prompt" not in signature.parameters
        assert "prompt" not in signature.parameters

    assert inspect.signature(vlm_as_a_judge.compute_score_ocr) == inspect.signature(genrm_ocr.compute_score_ocr)
    assert inspect.signature(vlm_as_a_judge.compute_score_unified_reward) == inspect.signature(
        unified_reward.compute_score_unified_reward
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy, implementation_name, function_name",
    [
        (genrm_ocr, "_compute_score_ocr_impl", "compute_score_ocr"),
        (unified_reward, "_compute_score_unified_reward_impl", "compute_score_unified_reward"),
    ],
)
async def test_legacy_wrappers_delegate_to_new_implementation(monkeypatch, legacy, implementation_name, function_name):
    calls = []

    async def fake_implementation(*args, **kwargs):
        calls.append((args, kwargs))
        return {"score": 1.0}

    monkeypatch.setattr(vlm_as_a_judge, implementation_name, fake_implementation)

    result = await getattr(legacy, function_name)(
        data_source="test",
        solution_image=_image(),
        ground_truth="hello",
        extra_info={"media_kind": "image"},
        reward_router_address="judge:8000",
        model_name="judge-model",
    )

    assert result == {"score": 1.0}
    assert calls[0][1]["transport"] is legacy._chat_complete


@pytest.mark.asyncio
async def test_legacy_transport_delegates_to_new_transport(monkeypatch):
    requests = []

    async def transport(router_address, chat_complete_request):
        requests.append(chat_complete_request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

    monkeypatch.setattr(vlm_as_a_judge, "_chat_complete", transport)

    result = await _score_ocr(genrm_ocr)

    assert result["score"] == 1.0
    assert requests[0] == OCR_EXPECTED_REQUEST


@pytest.mark.asyncio
@pytest.mark.parametrize("module, function_name, transport_name", OCR_SCORERS)
async def test_ocr_user_prompt_literal_and_package(monkeypatch, module, function_name, transport_name):
    requests = _capture(monkeypatch, module, transport_name)

    await _score_ocr(module, user_prompt="Transcribe only the largest word.")
    await _score_ocr(
        module,
        user_prompt="pkg://tests.utils.reward_score.test_vlm_as_a_judge_on_cpu.build_ocr_prompt",
    )

    assert requests[0]["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert requests[0]["messages"][1]["content"][1]["text"] == "Transcribe only the largest word."
    assert requests[1]["messages"][1]["content"][1]["text"] == "Transcribe every visible word."


@pytest.mark.asyncio
@pytest.mark.parametrize("module, function_name, transport_name", UNIFIED_SCORERS)
async def test_unified_user_prompt_literal_and_package(monkeypatch, module, function_name, transport_name):
    requests = _capture(monkeypatch, module, transport_name)

    await _score_unified(module, user_prompt="Score {caption} in one line. Caption: {caption}")
    await _score_unified(
        module,
        user_prompt="pkg://tests.utils.reward_score.test_vlm_as_a_judge_on_cpu.build_unified_prompt",
    )

    assert requests[0]["messages"][0]["content"][1]["text"] == ("Score a cat in one line. Caption: a cat")
    assert requests[1]["messages"][0]["content"][1]["text"] == "Judge the image against: a cat"


@pytest.mark.asyncio
@pytest.mark.parametrize("module, function_name, transport_name", OCR_SCORERS)
async def test_ocr_default_model_path_does_not_depend_on_real_home(
    monkeypatch, tmp_path, module, function_name, transport_name
):
    requests = _capture(monkeypatch, module, transport_name)
    monkeypatch.setenv("HOME", str(tmp_path))

    await _score_ocr(module, model_name=None)

    assert requests[0]["model"] == str(tmp_path / "models/tiny-random/qwen3-vl")


@pytest.mark.asyncio
async def test_unified_default_model_path_is_unchanged(monkeypatch):
    requests = _capture(monkeypatch, vlm_as_a_judge, "_chat_complete_with_session")

    await _score_unified(vlm_as_a_judge, model_name=None)

    assert requests[0]["model"] == "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b"


@pytest.mark.asyncio
async def test_ocr_sampling_environment_and_explicit_seed_override(monkeypatch):
    requests = _capture(monkeypatch, vlm_as_a_judge, "_chat_complete")
    monkeypatch.setenv("GENRM_OCR_SEED", "777")
    monkeypatch.setenv("GENRM_OCR_MAX_TOKENS", "123")

    await _score_ocr(vlm_as_a_judge)
    await _score_ocr(vlm_as_a_judge, sampling_params={"max_tokens": 17, "temperature": 0.2})

    assert requests[0]["seed"] == 777
    assert requests[0]["max_tokens"] == 123
    assert "seed" not in requests[1]
    assert requests[1]["max_tokens"] == 17
    assert requests[1]["temperature"] == 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("module, function_name, transport_name", OCR_SCORERS + UNIFIED_SCORERS)
async def test_scorers_do_not_emit_deprecation_warnings(monkeypatch, module, function_name, transport_name):
    _capture(monkeypatch, module, transport_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if function_name == "compute_score_ocr":
            await _score_ocr(module)
        else:
            await _score_unified(module)

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
@pytest.mark.parametrize("module_path, object_name, transport_name", EXTERNAL_SCORERS)
async def test_file_entrypoints_are_callable(monkeypatch, module_path, object_name, transport_name):
    monkeypatch.chdir(REPO_ROOT)
    scorer = load_extern_object("file://" + module_path, object_name)
    requests = []

    async def transport(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        content = UNIFIED_MODEL_OUTPUT if chat_complete_request["messages"][0]["role"] == "user" else "hello"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    assert inspect.iscoroutinefunction(scorer)
    # The file-path loader returns only the object; its real module globals still
    # expose the legacy transport that existing tests patch.
    monkeypatch.setitem(scorer.__globals__, transport_name, transport)
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


def test_qwen_judge_prompt_matches_baseline_oracle():
    prompt = vlm_as_a_judge.build_judge_prompt("A answer", "B answer")
    encoded = prompt.encode("utf-8")

    assert len(encoded) == QWEN_EXPECTED_PROMPT_LENGTH
    assert hashlib.sha256(encoded).hexdigest() == QWEN_EXPECTED_PROMPT_SHA256
    assert prompt.startswith("You are a strict multimodal answer evaluator.")
    assert '"winner": "A|B|TIE"' in prompt
    assert prompt.endswith('  "rationale": "short explanation"\n}')


def _judge_result_fields(result) -> dict:
    return {key: value for key, value in vars(result).items() if key != "raw_response"}


def test_parse_judge_response_matches_baseline():
    result = vlm_as_a_judge.parse_judge_response(QWEN_VALID_RESPONSE, {"A": "reference", "B": "trained"})

    assert _judge_result_fields(result) == {
        "reference_score": 7.0,
        "trained_score": 5.0,
        "winner": "reference",
        "rationale": "more specific",
        "reference_dimension_scores": {
            "fluency": 7.0,
            "relevance": 8.0,
            "accuracy": 7.0,
            "reasoning_quality": 6.0,
            "safety": 9.0,
        },
        "trained_dimension_scores": {
            "fluency": 5.0,
            "relevance": 5.0,
            "accuracy": 5.0,
            "reasoning_quality": 5.0,
            "safety": 5.0,
        },
    }

    swapped = vlm_as_a_judge.parse_judge_response(QWEN_VALID_RESPONSE, {"A": "trained", "B": "reference"})
    assert swapped.reference_score == 5.0
    assert swapped.trained_score == 7.0
    assert swapped.winner == "trained"


def test_parse_judge_response_handles_fenced_and_think_blocks():
    fenced = (
        "```json\n"
        '{"A": {"overall_score": 3, "fluency": 3, "relevance": 3, "accuracy": 3, '
        '"reasoning_quality": 3, "safety": 3}, "B": {"overall_score": 9, "fluency": 9, '
        '"relevance": 9, "accuracy": 9, "reasoning_quality": 9, "safety": 9}, '
        '"winner": "B", "rationale": "B"}\n'
        "```"
    )
    result = vlm_as_a_judge.parse_judge_response(fenced, {"A": "reference", "B": "trained"})
    assert result.reference_score == 3.0
    assert result.trained_score == 9.0
    assert result.winner == "trained"

    with_think = "<think>secret</think> " + QWEN_VALID_RESPONSE
    result = vlm_as_a_judge.parse_judge_response(with_think, {"A": "reference", "B": "trained"})
    assert result.reference_score == 7.0
    assert result.trained_score == 5.0


def test_parse_judge_response_recovers_textual_scores():
    text = (
        "Candidate A:\nOverall Score: 7\nFluency: 7\nRelevance: 8\nAccuracy: 7\n"
        "Reasoning Quality: 6\nSafety: 9\n\nCandidate B:\nOverall Score: 5\n"
        "Fluency: 5\nRelevance: 5\nAccuracy: 5\nReasoning Quality: 5\nSafety: 5\n\nWinner: A"
    )

    result = vlm_as_a_judge.parse_judge_response(text, {"A": "reference", "B": "trained"})

    assert result.reference_score == 7.0
    assert result.trained_score == 5.0
    assert result.winner == "reference"
    assert result.rationale == "Recovered from non-JSON judge response."


def test_parse_judge_response_raises_on_non_json():
    with pytest.raises(json.JSONDecodeError):
        vlm_as_a_judge.parse_judge_response("not a judge response", {"A": "reference", "B": "trained"})


def _load_example_module():
    path = REPO_ROOT / "examples/dpo_trainer/qwen3_omni/vlm_as_judge.py"
    module_name = "test_vlm_as_a_judge_example_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


qwen_example = _load_example_module()


def _judge_sample():
    return qwen_example.EvalSample(
        data_file="sample.parquet",
        index=0,
        uid="sample-0",
        modality="text",
        prompt_text="Describe the scene.",
        media={"images": [], "videos": [], "audios": []},
        raw_prompt=[],
    )


def _judge_with_capture(monkeypatch, responses, *, seed=1, max_tokens=256, retry_max_tokens=512):
    calls = []
    response_iterator = iter(responses)

    def post_chat_completion(router_address, payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": next(response_iterator)}}]}

    monkeypatch.setattr(qwen_example, "_post_chat_completion", post_chat_completion)
    result = qwen_example.judge_pair(
        sample=_judge_sample(),
        reference_text="A reference answer.",
        trained_text="A trained answer.",
        router_address="judge:8000",
        model_name="judge-model",
        max_tokens=max_tokens,
        retry_max_tokens=retry_max_tokens,
        temperature=0.0,
        rng=random.Random(seed),
    )
    return result, calls


def test_judge_pair_request_matches_baseline_and_retries(monkeypatch):
    result, calls = _judge_with_capture(monkeypatch, ["not json", QWEN_VALID_RESPONSE])

    encoded = json.dumps(calls[0]).encode("utf-8")
    assert len(encoded) == QWEN_EXPECTED_REQUEST_LENGTH
    assert hashlib.sha256(encoded).hexdigest() == QWEN_EXPECTED_REQUEST_SHA256
    assert [call["max_tokens"] for call in calls] == [256, 512]
    assert result.reference_score == 7.0
    assert result.trained_score == 5.0
    assert result.winner == "reference"


@pytest.mark.parametrize(
    "seed, candidate_a",
    [
        (1, "Candidate A:\nA reference answer."),
        (0, "Candidate A:\nA trained answer."),
    ],
)
def test_judge_pair_randomizes_ab_order(monkeypatch, seed, candidate_a):
    _, calls = _judge_with_capture(monkeypatch, [QWEN_VALID_RESPONSE], seed=seed, retry_max_tokens=256)

    prompt = calls[0]["messages"][0]["content"][-1]["text"]
    assert candidate_a in prompt
    assert calls[0]["model"] == "judge-model"
    assert calls[0]["top_p"] == 1.0
    assert calls[0]["temperature"] == 0.0


def build_ocr_prompt() -> str:
    return "Transcribe every visible word."


def build_unified_prompt(caption: str) -> str:
    return f"Judge the image against: {caption}"
