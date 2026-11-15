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
import math
import os
import secrets
import time
from typing import Any, Optional

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.router import GlobalRequestLoadBalancer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Conservative default so shutdown / persistent failures cannot loop forever.
DEFAULT_MAX_RETRIES = 50
DEFAULT_RETRY_WAIT_S = 1.0
DEFAULT_REQUEST_TIMEOUT_S = 1800.0


class RolloutPoolUnavailable(RuntimeError):
    """The rollout pool is temporarily empty."""


class PausableRequestLoadBalancer(GlobalRequestLoadBalancer):
    """Keep upstream routing and expose an empty pool as a typed error."""

    def acquire_server(self, request_id: str):
        """Acquire normally, or report the closed routing pool."""
        if not self.get_all_servers():
            raise RolloutPoolUnavailable("No available servers in load balancer")
        return super().acquire_server(request_id)


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
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries <= 0:
            raise ValueError("max_retries must be a positive integer")
        if not math.isfinite(retry_wait_s) or retry_wait_s < 0:
            raise ValueError("retry_wait_s must be finite and nonnegative")
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        self.max_retries = max_retries
        self.retry_wait_s = float(retry_wait_s)
        self.only_hybrid = only_hybrid
        self.request_timeout_s = request_timeout_s

    async def _acquire_server(self, request_id: str, **extra):
        """Wait through hybrid training pauses without consuming sample retries."""
        started = time.perf_counter()
        waited = False
        try:
            while True:
                try:
                    return await super()._acquire_server(request_id, **extra)
                except RolloutPoolUnavailable:
                    if not self.only_hybrid:
                        raise
                    waited = True
                    await asyncio.sleep(self.retry_wait_s)
        finally:
            if waited:
                logger.info("rollout_routing_wait request_id=%s wait_s=%.6f", request_id, time.perf_counter() - started)

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
        async with asyncio.timeout(self.request_timeout_s if self.only_hybrid else None):
            sampling_params = dict(sampling_params)
            if sampling_params.get("seed") is None:
                sampling_params["seed"] = secrets.randbits(32)
            retry_wait_s = 0.0
            for attempt in range(self.max_retries):
                logger.debug("diffusion_rollout_attempt request_id=%s attempt=%d", request_id, attempt + 1)
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
                    output.extra_fields = {
                        **output.extra_fields,
                        "rollout_retries": attempt,
                        "rollout_retry_wait_s": retry_wait_s,
                    }
                    return output

                logger.debug(
                    "diffusion rollout aborted (attempt %d/%d) for request_id=%s; retrying whole sample",
                    attempt + 1,
                    self.max_retries,
                    request_id,
                )
                if attempt + 1 < self.max_retries:
                    wait_started = time.perf_counter()
                    await asyncio.sleep(self.retry_wait_s)
                    retry_wait_s += time.perf_counter() - wait_started
            raise RuntimeError(f"Diffusion sample {request_id} aborted after {self.max_retries} attempts")
