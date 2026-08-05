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
import logging
import os
from typing import Any, Optional

from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Conservative default so shutdown / persistent failures cannot loop forever.
DEFAULT_MAX_RETRIES = 50
DEFAULT_RETRY_WAIT_S = 1.0


class DiffusionWholeSampleRetryLLMServerClient(LLMServerClient):
    """LLM server client that retries whole diffusion samples on abort.

    Behavior:
    - Call the base ``LLMServerClient.generate`` with the original (unmodified)
      ``prompt_ids`` and ``sampling_params`` every attempt.
    - If the returned ``DiffusionOutput.stop_reason`` is ``"aborted"`` /
      ``"abort"``, sleep briefly and retry the whole sample.
    - Track ``min_global_steps`` (earliest version seen) and ``max_global_steps``
      (latest version seen) across attempts and write them onto the final
      output's ``extra_fields`` so off-policy staleness metrics stay correct.
    - Give up after ``max_retries`` attempts and return the last (aborted) output
      so the agent loop can mark the prompt as a failure.
    """

    def __init__(
        self,
        *args,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_retries = max(1, int(max_retries))
        self.retry_wait_s = float(retry_wait_s)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Generate a diffusion sample, retrying the whole sample on abort."""
        min_global_steps: Optional[int] = None
        max_global_steps: Optional[int] = None
        output = None
        for attempt in range(self.max_retries):
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            if global_steps is not None:
                if min_global_steps is None:
                    min_global_steps = global_steps
                max_global_steps = global_steps

            if output.stop_reason not in ("aborted", "abort"):
                break

            logger.debug(
                "diffusion rollout aborted (attempt %d/%d) for request_id=%s; retrying whole sample",
                attempt + 1,
                self.max_retries,
                request_id,
            )
            await asyncio.sleep(self.retry_wait_s)

        if output is not None:
            output.extra_fields["min_global_steps"] = min_global_steps
            output.extra_fields["max_global_steps"] = max_global_steps
        return output
