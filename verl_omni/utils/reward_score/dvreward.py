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

"""AlphaGRPO DVReward adapted from https://github.com/huangrh99/AlphaGRPO; see the recipe README."""

import asyncio
import math
from collections.abc import Mapping, Sequence

import aiohttp
import torch

from verl_omni.pipelines.rollout_media import ArtifactContractError, select_artifact
from verl_omni.utils.reward_score.reward_utils import image_tensor_to_pil, pil_image_to_base64


def normalize_questions(questions: Sequence[str | Mapping]) -> list[str]:
    """Read a non-empty question group from AlphaGRPO JSONL or parquet metadata."""
    if isinstance(questions, str | Mapping):
        raise ValueError("DVReward requires a list of questions, not a string or mapping")
    normalized = []
    for item in questions:
        question = item["question"] if isinstance(item, Mapping) else item
        if not isinstance(question, str) or not question.strip():
            raise ValueError("DVReward questions must be non-empty strings")
        normalized.append(question)
    if not normalized:
        raise ValueError("DVReward requires both semantic and quality questions")
    return normalized


def _yes_probability(completion: dict) -> float:
    choice = completion["choices"][0]
    probabilities = {"yes": 0.0, "no": 0.0}
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    top_logprobs = (content[0].get("top_logprobs") or []) if content else []
    for item in top_logprobs:
        answer = item["token"].lower().strip().translate(str.maketrans("", "", ".,?!"))
        if answer in probabilities:
            logprob = item["logprob"]
            if not math.isfinite(logprob) or logprob > 0:
                raise ValueError("DVReward received an invalid answer log-probability")
            probabilities[answer] += math.exp(logprob)
    total = probabilities["yes"] + probabilities["no"]
    if total > 0:
        return probabilities["yes"] / total
    answer = choice["message"]["content"].lower().strip().translate(str.maketrans("", "", ".,?!"))
    if answer not in probabilities:
        raise ValueError("DVReward judge must answer yes or no")
    return float(answer == "yes")


async def compute_score_dvreward(
    data_source: str,
    solution_image: torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    model_name: str,
    reward_model_tokenizer=None,
    sampling_params: dict | None = None,
) -> dict[str, float]:
    """Score uint8 RGB [3, H, W] with semantic/quality geometric mean in [0, 1]."""
    semantic = normalize_questions(extra_info["semantic_questions"])
    quality = normalize_questions(extra_info["quality_questions"])
    if "media_artifacts" in extra_info:
        artifact = select_artifact(
            extra_info["media_artifacts"],
            name=extra_info.get("preview_artifact"),
            modality="image",
            representation="decoded",
        )
        if artifact.spec.layout != "CHW":
            raise ArtifactContractError("DVReward requires a canonical CHW image preview")
        solution_image = artifact.data
    elif extra_info.get("media_kind") != "image":
        raise ArtifactContractError("DVReward requires a named decoded image preview or explicit media_kind='image'")
    if (
        solution_image.dtype != torch.uint8
        or solution_image.ndim != 3
        or solution_image.shape[0] != 3
        or solution_image.numel() == 0
    ):
        raise ArtifactContractError("DVReward expects a uint8 RGB image with shape [3, H, W]")
    image = image_tensor_to_pil(solution_image)
    image_url = await asyncio.to_thread(pil_image_to_base64, image)
    params = {"temperature": 0.0, "top_p": 0.9, "max_tokens": 10, **(sampling_params or {})}
    semaphore = asyncio.Semaphore(8)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def score_question(question: str) -> float:
            request = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert visual auditor. Your task is to strictly evaluate an AI-generated image "
                            "against a specific question. Answer only with 'yes' or 'no'. Do not give other outputs or "
                            "punctuation marks. If the subjects in the question don't exist, answer 'no'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": question},
                        ],
                    },
                ],
                **params,
                "logprobs": True,
                "top_logprobs": 5,
            }
            async with semaphore:
                async with session.post(
                    f"http://{reward_router_address}/v1/chat/completions", json=request
                ) as response:
                    response.raise_for_status()
                    return _yes_probability(await response.json())

        tasks = [asyncio.create_task(score_question(question)) for question in semantic + quality]
        try:
            scores = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    semantic_score = sum(scores[: len(semantic)]) / len(semantic)
    quality_score = sum(scores[len(semantic) :]) / len(quality)
    return {
        "score": math.sqrt(semantic_score * quality_score),
        "semantic_score": semantic_score,
        "quality_score": quality_score,
    }


async def compute_score_alphagrpo(*, extra_info: dict, **kwargs) -> dict[str, float]:
    """Add the official single-turn thinking-tag reward to DVReward."""
    thinking = extra_info.get("thinking_text")
    if not isinstance(thinking, str):
        raise ValueError("AlphaGRPO reward requires thinking_text from the rollout.")
    result = await compute_score_dvreward(extra_info=extra_info, **kwargs)
    format_score = float(thinking.startswith("<think>") and thinking.endswith("</think>"))
    result["dvreward"] = result["score"]
    result["thinking_format_score"] = format_score
    result["score"] += format_score
    return result
