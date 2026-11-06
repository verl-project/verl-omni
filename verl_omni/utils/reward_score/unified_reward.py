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

"""Compatibility wrapper for the UnifiedReward 2.0 scorer.

New code should import
:func:`verl_omni.utils.reward_score.vlm_as_a_judge.compute_score_unified_reward`.
This module stays importable for existing ``pkg://`` configs and file-path loads.
"""

from typing import Optional

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

from verl_omni.utils.reward_score.vlm_as_a_judge import (
    DEFAULT_UNIFIED_REWARD_MODEL_PATH,
    DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS,
    UNIFIED_REWARD_SCORE_PATTERN,
    _aggregate_unified_reward_scores,
    _build_unified_reward_prompt,
    _parse_unified_reward_scores,
    _prepare_solution_frames,
)

__all__ = [
    "DEFAULT_UNIFIED_REWARD_MODEL_PATH",
    "DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS",
    "UNIFIED_REWARD_SCORE_PATTERN",
    "compute_score_unified_reward",
    # Re-exported for callers that imported these helpers from this module.
    "_aggregate_unified_reward_scores",
    "_build_unified_reward_prompt",
    "_parse_unified_reward_scores",
    "_prepare_solution_frames",
]


async def _chat_complete(
    router_address: str,
    chat_complete_request: dict,
    session: aiohttp.ClientSession | None = None,
) -> ChatCompletion:
    """POST a chat completion request through the shared implementation."""
    from verl_omni.utils.reward_score import vlm_as_a_judge

    return await vlm_as_a_judge._chat_complete_with_session(
        router_address,
        chat_complete_request,
        session=session,
    )


async def _score_single_image(
    image,
    prompt: str,
    router_address: str,
    model_name: str,
    loop,
    session: aiohttp.ClientSession,
) -> tuple[float, float, str]:
    """Score one image frame against ``prompt`` via UnifiedReward."""
    from verl_omni.utils.reward_score import vlm_as_a_judge

    return await vlm_as_a_judge._score_single_image(
        image,
        prompt,
        router_address,
        model_name,
        loop,
        session,
        transport=_chat_complete,
    )


async def compute_score_unified_reward(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
    *,
    user_prompt: Optional[str] = None,
):
    """Compute a human-preference score via UnifiedReward 2.0.

    The reward model scores the generated image against its text caption on
    Alignment, Coherence, and Style axes. The returned ``score`` is the mean of
    those axes normalized from the model's 1-5 scale to ``[0, 1]``.

    ``user_prompt`` overrides the default prompt. A ``pkg://`` spec names a
    builder called with ``caption``; literal text replaces ``{caption}``.
    """
    from verl_omni.utils.reward_score import vlm_as_a_judge

    return await vlm_as_a_judge._compute_score_unified_reward_impl(
        data_source,
        solution_image,
        ground_truth,
        extra_info,
        reward_router_address,
        reward_model_tokenizer,
        model_name,
        user_prompt=user_prompt,
        transport=_chat_complete,
    )
