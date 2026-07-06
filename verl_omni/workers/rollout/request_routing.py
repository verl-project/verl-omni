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

import hashlib
import logging
import os

import ray
from cachetools import LRUCache

from verl_omni.workers.config.rollout_routing import (
    VALID_ROLLOUT_SERVER_ROUTING_POLICIES,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


def stable_shard_index(key: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


@ray.remote
class OmniRequestLoadBalancer:
    """Multi-replica request load balancer for verl-omni rollout servers.

    Replaces verl's ``GlobalRequestLoadBalancer`` with pluggable routing policies
    (for example ``prompt_uid_affinity`` and ``least_inflight``) for diffusion and
    omni async rollouts.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        policy: str = "least_inflight",
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        max_imbalance: int | None = None,
    ) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        if policy not in VALID_ROLLOUT_SERVER_ROUTING_POLICIES:
            raise ValueError(f"Unsupported routing policy: {policy!r}")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._policy = policy
        self._max_imbalance = max_imbalance
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._round_robin_idx = 0

    def acquire_server(
        self,
        request_id: str,
        routing_key: str | None = None,
    ) -> tuple[str, ray.actor.ActorHandle]:
        sticky_key = routing_key or request_id

        if self._policy == "prompt_hash_sharding":
            server_id = self._pick_sharded_server(sticky_key)
            self._inflight_requests[server_id] += 1
            return server_id, self._servers[server_id]

        if self._policy == "round_robin":
            server_id = self._pick_round_robin_server()
            self._inflight_requests[server_id] += 1
            return server_id, self._servers[server_id]

        if self._policy == "prompt_uid_affinity":
            return self._acquire_sticky(sticky_key)

        return self._acquire_sticky(request_id)

    def _pick_sharded_server(self, sticky_key: str) -> str:
        server_ids = sorted(self._servers.keys())
        return server_ids[stable_shard_index(sticky_key, len(server_ids))]

    def _pick_round_robin_server(self) -> str:
        server_ids = sorted(self._servers.keys())
        server_id = server_ids[self._round_robin_idx % len(server_ids)]
        self._round_robin_idx += 1
        return server_id

    def _acquire_sticky(self, sticky_key: str) -> tuple[str, ray.actor.ActorHandle]:
        if sticky_key in self._request_id_to_server:
            server_id = self._request_id_to_server[sticky_key]
            if server_id in self._inflight_requests:
                # Soft Affinity load check
                if (
                    self._policy == "prompt_uid_affinity"
                    and self._max_imbalance is not None
                    and self._max_imbalance > 0
                ):
                    min_server_id = min(self._inflight_requests, key=self._inflight_requests.get)
                    if (
                        self._inflight_requests[server_id] - self._inflight_requests[min_server_id]
                        > self._max_imbalance
                    ):
                        # Divert request to least loaded replica and update stickiness mapping
                        server_id = min_server_id
                        self._request_id_to_server[sticky_key] = server_id

                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            del self._request_id_to_server[sticky_key]

        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[sticky_key] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str) -> None:
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle

    def remove_servers(self, server_ids: list[str]) -> None:
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)

    def get_status(self) -> dict:
        return {
            "policy": self._policy,
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
        }
