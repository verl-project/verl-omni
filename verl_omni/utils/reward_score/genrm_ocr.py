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

"""Compatibility wrapper for the OCR generative reward-model scorer.

New code should import :func:`verl_omni.utils.reward_score.vlm_as_a_judge.compute_score_ocr`.
This module stays importable for existing ``pkg://`` configs and file-path loads.
"""

from typing import Optional

import numpy as np
import torch
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

from verl_omni.utils.reward_score.vlm_as_a_judge import (
    DEFAULT_GRM_MODEL_PATH,
    DEFAULT_GRM_PROMPT,
    DEFAULT_SAMPLING_PARAMS,
    _levenshtein_score,
    _sampling_params,
    _to_pil,
)

__all__ = [
    "DEFAULT_GRM_MODEL_PATH",
    "DEFAULT_GRM_PROMPT",
    "DEFAULT_SAMPLING_PARAMS",
    "compute_score_ocr",
    # Re-exported for callers that imported these helpers from this module.
    "_levenshtein_score",
    "_sampling_params",
    "_to_pil",
]


async def _chat_complete(router_address: str, chat_complete_request: dict) -> ChatCompletion:
    """POST a chat completion request through the shared implementation."""
    from verl_omni.utils.reward_score import vlm_as_a_judge

    return await vlm_as_a_judge._chat_complete(router_address, chat_complete_request)


async def compute_score_ocr(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
    sampling_params: dict = None,
    *,
    user_prompt: Optional[str] = None,
):
    """Compute an image OCR score via a generative reward model (GRM).

    The image is sent to a GRM via an OpenAI-compatible router; the returned
    transcription is compared to ``ground_truth`` using Levenshtein distance to
    yield a score in ``[0, 1]`` (1 = perfect match).

    Args:
        data_source: Source dataset identifier. Unused, kept for interface
            consistency.
        solution_image: The solution image or video to be evaluated.
        ground_truth: The ground truth text for comparison.
        extra_info: Additional information; ``frame_interval`` controls video
            frame subsampling.
        reward_router_address: ``host:port`` of the GRM router.
        reward_model_tokenizer: Tokenizer for the reward model. Unused, kept
            for interface consistency.
        model_name: Name or path of the GRM. Defaults to
            ``DEFAULT_GRM_MODEL_PATH``.
        sampling_params: Override the defaults from :func:`_sampling_params`.
            An explicit ``seed`` opts in to reproducible sampling; without one
            any env-inherited seed is dropped.
        user_prompt: User prompt for the GRM. Defaults to
            ``DEFAULT_GRM_PROMPT``. A ``pkg://`` spec names a zero-argument
            prompt builder.

    Returns:
        dict: ``{"score": float, "genrm_response": str}``.
    """
    from verl_omni.utils.reward_score import vlm_as_a_judge

    return await vlm_as_a_judge._compute_score_ocr_impl(
        data_source,
        solution_image,
        ground_truth,
        extra_info,
        reward_router_address,
        reward_model_tokenizer,
        model_name,
        sampling_params,
        user_prompt=user_prompt,
        transport=_chat_complete,
    )
