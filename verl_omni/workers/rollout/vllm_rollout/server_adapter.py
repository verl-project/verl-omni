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
"""vLLM-Omni server adapter: verl's vLLM ``ServerAdapter`` plus the delta wire format."""

import logging
import os
import time
from typing import Generator

import torch
from verl.workers.rollout.vllm_rollout import ServerAdapter
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VLLMOmniServerAdapter(ServerAdapter):
    """verl's vLLM ``ServerAdapter`` plus the ``delta_flush`` wire format.

    The ``delta_sharded`` checkpoint engine's ``receive_weights`` yields per-flush
    sparse payloads (``__delta_spec__`` / ``__positions__`` / ``__values__``) already
    resident on this worker's GPU. Each flush's sentinel tensors ride the existing
    bucketed ZMQ channel to the colocated vllm-omni worker, whose
    ``update_weights_from_ipc`` (``delta_flush=True``) reassembles and applies them
    in place -- no full-model mirror is staged on the rollout side.
    """

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        if wire_format != "delta_flush":
            return await super().update_weights(weights, global_steps=global_steps, wire_format=wire_format, **kwargs)

        start_time = time.time()
        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={"use_shm": self.use_shm, "delta_flush": True},
        )

        def _flatten(flushes):
            # Every rank must iterate the flush generator to the end: the receive
            # loop's collective broadcasts deadlock otherwise.
            for named, _is_last in flushes:
                yield from named

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(_flatten(weights))

        if future is not None:
            await future

        # reset caches after updating weights
        if self._has_server:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"delta update_weights done, time cost: {time.time() - start_time:.2f}s")
