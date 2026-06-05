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

"""Image-edit quality scoring backed by a generative reward model (GRM)."""

import asyncio
import io
import json
import logging
import os
import re
from typing import Optional

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from PIL import Image
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_GRM_MODEL_PATH = "~/models/tiny-random/qwen3-vl"
DEFAULT_SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64}
DEFAULT_GRM_PROMPT = """Rate how well the edited image follows the edit instruction while preserving unrelated content
from the source image and looking natural.

If a reference target image is provided, also consider whether the edited image matches it.

Output only one number from 0 to 1. 1 means perfect, 0 means failed."""
DEBUG_LOG_ENV = "GENRM_IMAGE_EDIT_DEBUG_PATH"

# Reward-server resilience knobs.
# The GRM is a remote vLLM server that occasionally drops requests (e.g. when
# the multimodal cache asserts on its receiver side, or when the server is
# briefly overloaded). These are transient and should not crash a multi-hour
# training run; soft-fail to a 0.0 score instead and keep going.
_REQUEST_TIMEOUT_S = 600.0
_MAX_ATTEMPTS = 3
_RETRY_BASE_BACKOFF_S = 1.0
_SOFT_FAIL_SCORE = 0.0


class _RewardServerError(RuntimeError):
    """Raised when the reward server cannot produce a usable response.

    Caught at the public ``compute_score_image_edit`` boundary and turned
    into a soft-fail score; never propagates up to the trainer.
    """


async def _chat_complete(router_address: str, chat_complete_request: dict) -> ChatCompletion:
    """POST a chat completion request to the GRM router and parse the response.

    Retries transient HTTP / network / decode failures up to ``_MAX_ATTEMPTS``
    times with exponential backoff. Raises ``_RewardServerError`` on terminal
    failure so the caller can soft-fail to a default score.
    """
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    last_err: Optional[str] = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=chat_complete_request) as resp:
                    body = await resp.text()
                    status = resp.status
            if status != 200:
                # Truncate to keep log lines bounded; vLLM error pages can be huge.
                last_err = f"HTTP {status}: {body[:500]!r}"
                raise _RewardServerError(last_err)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as e:
                last_err = f"non-JSON body (HTTP {status}): {body[:500]!r} ({e})"
                raise _RewardServerError(last_err) from e
            try:
                return ChatCompletion(**payload)
            except (TypeError, ValueError) as e:
                last_err = f"malformed ChatCompletion payload: {payload!r} ({e})"
                raise _RewardServerError(last_err) from e
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = f"transport error: {type(e).__name__}: {e}"
        except _RewardServerError:
            # Already has a useful ``last_err`` set above; let retry logic decide.
            pass

        if attempt < _MAX_ATTEMPTS:
            backoff = _RETRY_BASE_BACKOFF_S * (2 ** (attempt - 1))
            logger.warning(
                "[genrm_image_edit] reward-server attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                _MAX_ATTEMPTS,
                last_err,
                backoff,
            )
            await asyncio.sleep(backoff)

    raise _RewardServerError(f"reward server failed after {_MAX_ATTEMPTS} attempts: {last_err}")


def _to_pil(image) -> Image.Image:
    """Normalize a tensor / array / PIL image / parquet image dict to an RGB PIL image."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(io.BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])
    if isinstance(image, torch.Tensor):
        image = image.float().permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "must be in HWC format"
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return image.convert("RGB")


def _parse_score(text: str) -> float:
    """Parse the first numeric score from a GRM response."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return 0.0
    score = float(match.group(0))
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _write_debug_record(record: dict) -> None:
    """Append a reward debugging record when ``GENRM_IMAGE_EDIT_DEBUG_PATH`` is set."""
    debug_path = os.environ.get(DEBUG_LOG_ENV)
    if debug_path is None:
        return
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    with open(debug_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clamp_score(value, default: float = 0.0) -> float:
    """Convert a model-emitted value to a score in ``[0, 1]``."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


async def compute_score_image_edit(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
):
    """Compute image-edit quality with a vision-language GRM judge.

    The judge is instructed to output a single numeric score in ``[0, 1]``.
    Set ``GENRM_IMAGE_EDIT_DEBUG_PATH`` to append raw GRM responses as JSONL for
    debugging.
    """
    from verl.utils.ray_utils import get_event_loop

    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    instruction = extra_info.get("instruction") or ground_truth
    source_img = extra_info.get("source_img")
    target_img = extra_info.get("target_img")
    # Fail closed when ``source_img`` is missing instead of letting
    # ``_to_pil(None)`` raise an unhelpful ``AssertionError`` deep inside
    # the converter. ``target_img`` stays optional — only the source is
    # required for the GRM-style image-edit judge prompt.
    if source_img is None:
        raise ValueError(
            "compute_score_image_edit requires extra_info['source_img'] "
            "(the original image being edited); got None. Check the "
            "data preprocessor wired ``source_img`` into ``extra_info``."
        )
    has_target = target_img is not None

    if solution_image.ndim == 4:
        solution_image = solution_image[0]

    loop = get_event_loop()
    source_base64 = await loop.run_in_executor(None, pil_image_to_base64, _to_pil(source_img))
    solution_base64 = await loop.run_in_executor(None, pil_image_to_base64, _to_pil(solution_image))

    content = [
        {"type": "text", "text": f"Editing instruction: {instruction}"},
        {"type": "text", "text": "Source image:"},
        {"type": "image_url", "image_url": {"url": source_base64}},
        {"type": "text", "text": "Edited/generated image:"},
        {"type": "image_url", "image_url": {"url": solution_base64}},
    ]
    if has_target:
        target_base64 = await loop.run_in_executor(None, pil_image_to_base64, _to_pil(target_img))
        content.extend(
            [
                {"type": "text", "text": "Reference target image:"},
                {"type": "image_url", "image_url": {"url": target_base64}},
            ]
        )
    content.append({"type": "text", "text": DEFAULT_GRM_PROMPT})

    model_name = model_name or os.path.expanduser(DEFAULT_GRM_MODEL_PATH)
    chat_complete_request = {
        "messages": [
            {"role": "system", "content": "You are a strict image editing evaluation assistant."},
            {"role": "user", "content": content},
        ],
        "model": model_name,
        **DEFAULT_SAMPLING_PARAMS,
    }
    try:
        result = await _chat_complete(router_address=reward_router_address, chat_complete_request=chat_complete_request)
    except _RewardServerError as e:
        # Soft-fail: a single bad GRM call should not crash a multi-hour
        # training run. Returning the floor score lets GRPO advantage
        # normalization treat the sample as a no-signal example. The
        # ``genrm_response`` field doubles as a debug breadcrumb so the
        # failure is visible in rollout dumps.
        logger.warning("[genrm_image_edit] soft-failing to %.1f: %s", _SOFT_FAIL_SCORE, e)
        _write_debug_record(
            {
                "data_source": data_source,
                "instruction": instruction,
                "ground_truth": ground_truth,
                "img_id": extra_info.get("img_id"),
                "turn_index": extra_info.get("turn_index"),
                "score": _SOFT_FAIL_SCORE,
                "genrm_response": None,
                "error": str(e),
            }
        )
        return {"score": _SOFT_FAIL_SCORE, "genrm_response": f"[reward-server-error] {e}"}

    try:
        grm_response = result.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning(
            "[genrm_image_edit] soft-failing to %.1f: missing choices/message in response: %s",
            _SOFT_FAIL_SCORE,
            e,
        )
        return {"score": _SOFT_FAIL_SCORE, "genrm_response": f"[malformed-response] {e}"}

    score = _parse_score(grm_response)
    _write_debug_record(
        {
            "data_source": data_source,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "img_id": extra_info.get("img_id"),
            "turn_index": extra_info.get("turn_index"),
            "score": score,
            "genrm_response": grm_response,
        }
    )
    return {"score": score, "genrm_response": grm_response}
