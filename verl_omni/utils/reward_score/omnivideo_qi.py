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
"""OmniVideo-R1 query-intensive grounding reward.

Implements Eq. (8) from the paper::

    R_QI = r_format + r_answer + 0.5 * (r_consistency + r_completeness)

The two intent rewards use an OpenAI-compatible Qwen3-VL judge. The policy
video remains on shared storage; only a bounded set of sampled segment frames
is sent to the judge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
_PAIR_SOURCE = (
    r"<time>(?P<start>\d+\.\d)-(?P<end>\d+\.\d)</time>"
    r"<caption>(?P<caption>.*?)</caption>"
)
_PAIR_RE = re.compile(_PAIR_SOURCE, re.DOTALL)
_RESPONSE_RE = re.compile(
    rf"\A\s*(?P<pairs>(?:{_PAIR_SOURCE}\s*)+)"
    r"<thinking>(?P<thinking>.*?)</thinking>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CHOICE_RE = re.compile(r"^\s*([A-H])(?:[.):\s]|$)", re.IGNORECASE)

CONSISTENCY_SYSTEM_PROMPT = """You are an expert evaluator for video content description. Verify the factual
correctness of statements in the caption against the video content.

Evaluation Criteria:
1. Factual Verification: Every visual claim, action, and object in the caption is supported by the video.
2. Descriptive Accuracy: Attributes such as colors, directions, counts, and sequences match the visuals.
3. Absence of Fabrications: The caption contains no hallucinations or descriptions of absent objects and events.
4. Logical Audio Consistency: Audio descriptions are accurate when contextually plausible given the visual scene,
unless contradicted by the visual context.

Scoring Guidelines:
- 1.0: All statements are completely accurate and visually verified.
- 0.7-0.9: Mostly accurate, with only very minor phrasing issues or negligible details.
- 0.4-0.6: The caption describes the correct general topic but contains specific claims absent from the video.
- 0.1-0.3: Major fabrications describe actions or objects that clearly do not exist in the video.
- 0.0: A completely different scenario or an irrelevant caption.

Respond with only a JSON object: {"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}."""

COMPLETENESS_SYSTEM_PROMPT = """You are an expert evaluator for video-based question answering. Evaluate the
selected video segments based on their ability to support a specific question-answer pair. The input consists
of stitched clips.

Standards:
1. Content Sufficiency and Inference: The segments provide comprehensive visual evidence supporting the answer.
For questions requiring audio information, visual context may imply and support the answer.
2. Temporal Precision: The segments are tightly trimmed to key moments, avoiding excessive padding or redundant scenes.
3. Completeness: The segments capture all necessary steps or details required to derive the solution.

Scoring Guidelines:
- 1.0: All necessary evidence is present and precisely trimmed with minimal redundancy.
- 0.7-0.9: Sufficient context with minor irrelevant footage or slightly loose trimming.
- 0.4-0.6: Core visual context is present but significant redundancy or key logical links are missing.
- 0.1-0.3: Only partial visual information is present or visual context poorly matches the answer.
- 0.0: Completely irrelevant segments or failure to visually show the source of the answer.

Respond with only a JSON object: {"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}."""

ANSWER_SYSTEM_PROMPT = """You are an impartial question-answer evaluator. Compare the candidate answer with the
reference answer for the given question. Reward semantic correctness, not wording overlap. Respond with only a
JSON object: {"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}."""


@dataclass(frozen=True)
class GroundingPair:
    start: float
    end: float
    caption: str


@dataclass(frozen=True)
class ParsedResponse:
    pairs: tuple[GroundingPair, ...]
    thinking: str
    answer: str
    format_valid: bool


@dataclass(frozen=True)
class JudgeConfig:
    url: str
    model: str
    timeout_seconds: float = 300.0
    max_frames_per_segment: int = 8
    max_completeness_frames: int = 32


def parse_response(solution_str: Any, max_segments: int = 8) -> ParsedResponse:
    """Parse the strict QI template and validate ordered, disjoint spans."""
    text = str(solution_str or "")
    match = _RESPONSE_RE.fullmatch(text)
    if match is None:
        answer_match = _ANSWER_RE.search(text)
        return ParsedResponse((), "", answer_match.group(1).strip() if answer_match else "", False)

    pairs = tuple(
        GroundingPair(float(item.group("start")), float(item.group("end")), item.group("caption").strip())
        for item in _PAIR_RE.finditer(match.group("pairs"))
    )
    thinking = match.group("thinking").strip()
    answer = match.group("answer").strip()
    valid = bool(pairs and thinking and answer and len(pairs) <= max_segments)
    previous_end = -1.0
    for pair in pairs:
        if not pair.caption or pair.start < 0 or pair.end <= pair.start or pair.start < previous_end:
            valid = False
            break
        previous_end = pair.end
    return ParsedResponse(pairs, thinking, answer, valid)


def _choice_letter(value: Any) -> str:
    match = _CHOICE_RE.match(str(value or ""))
    return match.group(1).upper() if match else ""


def _normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(value or "").casefold()).split())


def _clamp_score(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _judge_config(reward_router_address: str | None = None, reward_model_tokenizer: Any = None) -> JudgeConfig:
    url = os.getenv("OMNIVIDEO_QI_JUDGE_URL", "").rstrip("/")
    if not url and reward_router_address:
        url = f"http://{reward_router_address}/v1/chat/completions"
    elif url and not url.endswith("/chat/completions"):
        url = f"{url}/v1/chat/completions" if not url.endswith("/v1") else f"{url}/chat/completions"
    if not url:
        raise RuntimeError("QI intent rewards require OMNIVIDEO_QI_JUDGE_URL or an enabled trainer reward-model router")

    tokenizer_model = getattr(reward_model_tokenizer, "name_or_path", None)
    model = os.getenv("OMNIVIDEO_QI_JUDGE_MODEL", "") or tokenizer_model or DEFAULT_JUDGE_MODEL
    return JudgeConfig(
        url=url,
        model=model,
        timeout_seconds=float(os.getenv("OMNIVIDEO_QI_JUDGE_TIMEOUT", "300")),
        max_frames_per_segment=int(os.getenv("OMNIVIDEO_QI_MAX_FRAMES_PER_SEGMENT", "8")),
        max_completeness_frames=int(os.getenv("OMNIVIDEO_QI_MAX_COMPLETENESS_FRAMES", "32")),
    )


def _parse_judge_output(content: Any) -> tuple[float, str]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if object_match is None:
            return 0.0, "invalid judge response"
        try:
            payload = json.loads(object_match.group(0))
        except json.JSONDecodeError:
            return 0.0, "invalid judge response"
    if not isinstance(payload, dict):
        return 0.0, "invalid judge response"
    return _clamp_score(payload.get("score")), str(payload.get("reason") or "")


async def _chat_complete(
    session: aiohttp.ClientSession,
    config: JudgeConfig,
    content: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[float, str]:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }
    async with session.post(config.url, json=payload) as response:
        response.raise_for_status()
        result = await response.json()
    try:
        output = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected QI judge response: {result}") from exc
    return _parse_judge_output(output)


def _sample_evenly(items: list[str], limit: int) -> list[str]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return items[:1]
    return [items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)]


def _load_segment_frames(video_path: str, pair: GroundingPair, max_frames: int) -> list[str]:
    """Decode a bounded segment and return frame data URIs for the judge."""
    import torch
    from PIL import Image
    from qwen_vl_utils.vision_process import fetch_video

    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    max_frames = max(2, max_frames - max_frames % 2)
    video = fetch_video(
        {
            "video": video_path,
            "video_start": pair.start,
            "video_end": pair.end,
            "fps": 1.0,
            "min_frames": 2,
            "max_frames": max_frames,
        }
    )
    if isinstance(video, tuple):
        video = video[0]
    if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"Expected decoded video [T, 3, H, W], got {type(video).__name__}")

    frames = video.detach().to(device="cpu", dtype=torch.float32)
    if frames.numel() and frames.max().item() <= 1.0:
        frames = frames * 255
    frames = frames.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    return [pil_image_to_base64(Image.fromarray(frame)) for frame in frames]


def _frame_content(frames: list[str], text: str) -> list[dict[str, Any]]:
    return [
        *[{"type": "image_url", "image_url": {"url": frame}} for frame in frames],
        {"type": "text", "text": text},
    ]


async def _score_open_answer(
    session: aiohttp.ClientSession,
    config: JudgeConfig,
    question: str,
    prediction: str,
    ground_truth: str,
) -> tuple[float, str]:
    if _normalize_text(prediction) == _normalize_text(ground_truth):
        return 1.0, "normalized exact match"
    prompt = f"Question: {question}\nReference answer: {ground_truth}\nCandidate answer: {prediction}"
    return await _chat_complete(session, config, [{"type": "text", "text": prompt}], ANSWER_SYSTEM_PROMPT)


async def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute QI reward components and their Eq. (8) sum."""
    del kwargs
    extra_info = extra_info or {}
    max_segments = int(os.getenv("OMNIVIDEO_QI_MAX_SEGMENTS", "8"))
    parsed = parse_response(solution_str, max_segments=max_segments)
    format_reward = float(parsed.format_valid)

    is_multiple_choice = bool(extra_info.get("is_multiple_choice", False))
    if is_multiple_choice:
        prediction_letter = _choice_letter(parsed.answer)
        answer_reward = float(bool(prediction_letter) and prediction_letter == _choice_letter(ground_truth))
        answer_reason = "choice exact match" if answer_reward else "choice mismatch"
    else:
        answer_reward = None
        answer_reason = ""

    needs_judge = bool(parsed.pairs) or answer_reward is None
    if not needs_judge:
        return {
            "score": format_reward + float(answer_reward),
            "format": format_reward,
            "answer": float(answer_reward),
            "consistency": 0.0,
            "completeness": 0.0,
            "intent": 0.0,
            "num_groundings": 0,
            "decode_failures": 0,
            "answer_reason": answer_reason,
            "completeness_reason": "no valid grounding frames",
        }

    config = _judge_config(reward_router_address, reward_model_tokenizer)
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        answer_task = None
        if answer_reward is None:
            answer_task = asyncio.create_task(
                _score_open_answer(
                    session,
                    config,
                    str(extra_info.get("question") or ""),
                    parsed.answer,
                    ground_truth,
                )
            )

        frame_results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _load_segment_frames,
                    str(extra_info.get("video_path") or ""),
                    pair,
                    config.max_frames_per_segment,
                )
                for pair in parsed.pairs
            ],
            return_exceptions=True,
        )
        segment_frames: list[list[str]] = []
        decode_failures = 0
        for result in frame_results:
            if isinstance(result, BaseException):
                logger.warning("Failed to decode a predicted QI segment: %s", result)
                segment_frames.append([])
                decode_failures += 1
            else:
                segment_frames.append(result)

        consistency_tasks = []
        for pair, frames in zip(parsed.pairs, segment_frames, strict=True):
            if not frames:
                consistency_tasks.append(None)
                continue
            prompt = (
                "Video segment is provided above.\n\n"
                f"Caption to evaluate: {pair.caption}\n\n"
                "Evaluate whether this caption is accurate based on the video content."
            )
            content = _frame_content(frames, prompt)
            consistency_tasks.append(
                asyncio.create_task(_chat_complete(session, config, content, CONSISTENCY_SYSTEM_PROMPT))
            )

        consistency_results = await asyncio.gather(*[task for task in consistency_tasks if task is not None])
        consistency_iter = iter(consistency_results)
        consistency_scores = [0.0 if task is None else next(consistency_iter)[0] for task in consistency_tasks]
        consistency_reward = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

        completeness_reward = 0.0
        completeness_reason = "no valid grounding frames"
        if segment_frames and decode_failures == 0:
            stitched_frames = _sample_evenly(
                [frame for frames in segment_frames for frame in frames],
                config.max_completeness_frames,
            )
            completeness_prompt = (
                "The video segments shown above are the key segments selected by the model.\n\n"
                f"Question: {extra_info.get('question', '')}\n\n"
                f"Ground Truth Answer: {ground_truth}\n\n"
                "Evaluate whether these segments provide sufficient evidence to support the answer "
                "while remaining concise."
            )
            completeness_reward, completeness_reason = await _chat_complete(
                session,
                config,
                _frame_content(stitched_frames, completeness_prompt),
                COMPLETENESS_SYSTEM_PROMPT,
            )

        if answer_task is not None:
            answer_reward, answer_reason = await answer_task

    answer_reward = float(answer_reward)
    intent_reward = 0.5 * (consistency_reward + completeness_reward)
    score = format_reward + answer_reward + intent_reward
    return {
        "score": score,
        "format": format_reward,
        "answer": answer_reward,
        "consistency": consistency_reward,
        "completeness": completeness_reward,
        "intent": intent_reward,
        "num_groundings": len(parsed.pairs),
        "decode_failures": decode_failures,
        "answer_reason": answer_reason,
        "completeness_reason": completeness_reason,
    }
