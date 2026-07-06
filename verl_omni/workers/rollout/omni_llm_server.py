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

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig, OmegaConf
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput

from verl_omni.workers.rollout.request_routing import (
    DEFAULT_ROUTING_CACHE_SIZE,
    OmniRequestLoadBalancer,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OmniLLMServerClient(LLMServerClient):
    """LLM server client that routes requests via ``OmniRequestLoadBalancer``."""

    def __init__(self, config: DictConfig, load_balancer_handle: ray.actor.ActorHandle | None = None, **kwargs):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)

    async def _acquire_server(
        self,
        request_id: str,
        routing_key: str | None = None,
    ) -> tuple[str, ray.actor.ActorHandle]:
        return await self._load_balancer.acquire_server.remote(
            request_id=request_id,
            routing_key=routing_key,
        )

    @rollout_trace_op
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
        routing_key: str | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        server_id, server = await self._acquire_server(request_id, routing_key=routing_key)
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)


class OmniLLMServerManager(LLMServerManager):
    """Launch rollout replicas with ``OmniRequestLoadBalancer``."""

    async def _init_global_load_balancer(self) -> None:
        policy = OmegaConf.select(self.rollout_config, "server_routing.policy", default="least_inflight")
        routing_key_field = OmegaConf.select(self.rollout_config, "server_routing.routing_key_field", default="uid")
        max_imbalance = OmegaConf.select(self.rollout_config, "server_routing.max_imbalance", default=8)
        self.global_load_balancer = OmniRequestLoadBalancer.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            policy=policy,
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            max_imbalance=max_imbalance,
        )
        logger.info(
            "[OmniLLMServerManager] OmniRequestLoadBalancer policy=%s routing_key_field=%s max_imbalance=%s",
            policy,
            routing_key_field,
            max_imbalance,
        )

    def get_client(self, client_cls=OmniLLMServerClient, **kwargs) -> OmniLLMServerClient:
        return super().get_client(client_cls=client_cls, **kwargs)
