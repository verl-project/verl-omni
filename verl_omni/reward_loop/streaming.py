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
"""Direct, bounded sample dispatch to existing named reward worker groups."""

import asyncio
import random
from contextlib import asynccontextmanager

from verl.protocol import DataProto

from verl_omni.workers.config.reward import StreamingRewardConfig

from .reward_model_executor import _await_owned


def merge_reward_outputs(outputs: list[dict]) -> dict:
    """Merge already weighted group results without weighting them a second time."""
    total = 0.0
    info = {}
    for output in outputs:
        total += float(output["reward_score"])
        for key, value in output.get("reward_extra_info", {}).items():
            if key == "reward/combined":
                continue
            if key in info:
                raise ValueError(f"Duplicate reward extra-info key {key!r} across worker groups")
            info[key] = value
    info["reward/combined"] = total
    return {"reward_score": total, "reward_extra_info": info}


class StreamingRewardClient:
    """Own submitted sample RPCs until all groups finish, even after cancellation.

    One client belongs to one diffusion agent worker. A deadline discards the
    result, but accepted RPCs drain before releasing their admission slot.
    """

    def __init__(self, worker_groups: dict, config: StreamingRewardConfig):
        self.worker_groups = worker_groups
        self.timeout = config.timeout
        self._slots = asyncio.Semaphore(config.max_inflight)
        self._pending = set()
        self._closed = False

    async def compute_score(self, data: DataProto) -> dict:
        """Score one sample on every group and return the combined result."""
        if len(data) != 1:
            raise ValueError(f"Streaming rewards require one sample, got {len(data)}")
        async with self.sample_slot():
            return await self.compute_admitted_score(data)

    @asynccontextmanager
    async def sample_slot(self):
        """Bound generation through scoring, so completed media cannot queue without a limit."""
        async with self._slots:
            if self._closed:
                raise RuntimeError("Streaming reward admission is closed")
            yield

    async def compute_admitted_score(self, data: DataProto) -> dict:
        """Score an admitted sample; a slot owner must hold sample_slot()."""
        if self._closed:
            raise RuntimeError("Streaming reward admission is closed")
        async with asyncio.timeout(self.timeout):
            task = asyncio.create_task(self._compute_score(data))
            self._pending.add(task)
            try:
                return await _await_owned(task)
            finally:
                self._pending.remove(task)

    async def close(self) -> None:
        """Fence new submissions and wait for accepted group RPCs to settle."""
        self._closed = True
        await _await_owned(asyncio.gather(*self._pending, return_exceptions=True))

    async def _compute_score(self, data):
        requests = []
        submission_error = None
        try:
            for workers in self.worker_groups.values():
                requests.append(random.choice(workers).compute_score.remote(data))
        except Exception as exc:
            submission_error = exc
        results = await asyncio.gather(*requests, return_exceptions=True)
        if submission_error is not None:
            raise submission_error
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return merge_reward_outputs(results)
