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
"""Adapt legacy and named reward handles to the agent loop's async scoring path."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import numpy as np
from verl.protocol import DataProto

from verl_omni.pipelines.rollout_media import (
    ARTIFACT_CONTEXT,
    ARTIFACT_PREFIX,
    ARTIFACT_SPECS,
    PREVIEW_ARTIFACT,
    PRIMARY_ARTIFACT,
)
from verl_omni.workers.config.reward import StreamingRewardConfig

from .replica_dispatch import dispatch_reward_groups
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


def create_async_reward_client(
    handles: list | dict | None, config: StreamingRewardConfig
) -> "AsyncRewardClient | None":
    """Adapt manager-owned handles without changing legacy dispatch semantics."""
    if handles is None:
        return None
    if isinstance(handles, list):

        async def score(data):
            return await random.choice(handles).compute_score.remote(data)

        return AsyncRewardClient(score)

    async def score(data):
        if len(data) != 1:
            raise ValueError(f"Streaming rewards require one sample, got {len(data)}")
        if config.extra_fields is not None and "tool_extra_fields" in data.non_tensor_batch:
            retained = {
                *config.extra_fields,
                ARTIFACT_SPECS,
                PRIMARY_ARTIFACT,
                PREVIEW_ARTIFACT,
                ARTIFACT_CONTEXT,
                "media_kind",
                "audio",
                "audio_sample_rate",
                "img_shapes",
            }
            projected = [
                {key: value for key, value in fields.items() if key in retained or key.startswith(ARTIFACT_PREFIX)}
                for fields in data.non_tensor_batch["tool_extra_fields"]
            ]
            data = DataProto(
                batch=data.batch,
                non_tensor_batch={**data.non_tensor_batch, "tool_extra_fields": np.array(projected, dtype=object)},
                meta_info=data.meta_info,
            )
        selected = {name: (random.choice(workers),) for name, workers in handles.items()}
        outputs = await dispatch_reward_groups(data, selected, {})
        return merge_reward_outputs([values[0] for values in outputs.values()])

    return AsyncRewardClient(score, config)


class AsyncRewardClient:
    """Score through existing reward workers; protect admitted named requests until drain."""

    def __init__(self, score: Callable[[DataProto], Awaitable[dict]], config: StreamingRewardConfig | None = None):
        self._score = score
        self._timeout = config.timeout if config is not None else None
        self._slots = asyncio.Semaphore(config.max_inflight) if config is not None else None
        self._pending = set()
        self._closed = False

    @property
    def requires_drain(self) -> bool:
        """Whether the caller must drain sibling sample tasks on failure."""
        return self._slots is not None

    @asynccontextmanager
    async def sample_slot(self):
        """Bound generation through scoring, so completed media cannot queue without a limit."""
        if self._slots is None:
            yield
        else:
            async with self._slots:
                if self._closed:
                    raise RuntimeError("Streaming reward admission is closed")
                yield

    async def compute_score(self, data: DataProto) -> dict:
        """Score a sample; the generation caller owns sample_slot()."""
        if self._slots is None:
            return await self._score(data)
        if self._closed:
            raise RuntimeError("Streaming reward admission is closed")
        async with asyncio.timeout(self._timeout):
            task = asyncio.create_task(self._score(data))
            self._pending.add(task)
            try:
                return await _await_owned(task)
            finally:
                self._pending.remove(task)

    async def close(self) -> None:
        """Fence new submissions and wait for accepted group RPCs to settle."""
        self._closed = True
        await _await_owned(asyncio.gather(*self._pending, return_exceptions=True))
