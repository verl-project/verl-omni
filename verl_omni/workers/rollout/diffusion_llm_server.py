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
import secrets
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
    - Copy ``prompt_ids`` and ``sampling_params`` for every base
      ``LLMServerClient.generate`` call so it cannot modify retry inputs.
    - Choose one seed when none is supplied and reuse it on every attempt.
    - If the returned ``DiffusionOutput.stop_reason`` is ``"aborted"`` /
      ``"abort"``, sleep briefly and retry the whole sample.
    - Keep the successful attempt's version metadata: aborted samples are
      discarded, so their versions do not belong to the returned trajectory.
    - With ``only_hybrid``, wait for routing to reopen during actor training.
      This wait does not consume the sample retry budget.
    - Raise after ``max_retries`` aborted attempts so the agent loop marks the
      prompt group as failed instead of writing an empty successful sample.
    """

    def __init__(
        self,
        *args,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
        only_hybrid: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_retries = max(1, int(max_retries))
        self.retry_wait_s = float(retry_wait_s)
        self.only_hybrid = only_hybrid

    async def _acquire_server(self, request_id: str):
        while True:
            try:
                return await super()._acquire_server(request_id)
            except RuntimeError as error:
                if not self.only_hybrid or "No available servers in load balancer" not in str(error):
                    raise
                await asyncio.sleep(self.retry_wait_s)

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
        sampling_params = dict(sampling_params)
        if sampling_params.get("seed") is None:
            sampling_params["seed"] = secrets.randbits(32)
        for attempt in range(self.max_retries):
            output = await super().generate(
                request_id=request_id,
                prompt_ids=list(prompt_ids),
                sampling_params=dict(sampling_params),
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )
            if output.stop_reason not in ("aborted", "abort"):
                return output

            logger.debug(
                "diffusion rollout aborted (attempt %d/%d) for request_id=%s; retrying whole sample",
                attempt + 1,
                self.max_retries,
                request_id,
            )
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(self.retry_wait_s)
        raise RuntimeError(f"Diffusion sample {request_id} aborted after {self.max_retries} attempts")
