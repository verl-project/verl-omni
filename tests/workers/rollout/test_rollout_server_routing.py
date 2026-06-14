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

from collections import Counter

import pytest
import ray
from omegaconf import OmegaConf

from verl_omni.workers.config.rollout_routing import RolloutServerRoutingConfig
from verl_omni.workers.rollout.request_routing import (
    ConfigurableRequestLoadBalancer,
    stable_shard_index,
)


@ray.remote
class _DummyServer:
    pass


def _make_lb(policy: str, num_servers: int = 4) -> ray.actor.ActorHandle:
    servers = {f"s{i}": _DummyServer.remote() for i in range(num_servers)}
    return ConfigurableRequestLoadBalancer.remote(servers=servers, policy=policy)


@pytest.fixture(scope="module", autouse=True)
def _ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    yield
    ray.shutdown()


def test_stable_shard_index_is_deterministic():
    assert stable_shard_index("uid-42", 4) == stable_shard_index("uid-42", 4)
    assert 0 <= stable_shard_index("uid-42", 4) < 4


def test_rollout_server_routing_config_from_rollout_yaml():
    import os

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        cfg = compose(config_name="diffusion_trainer")
    assert OmegaConf.select(cfg.actor_rollout_ref.rollout, "server_routing.policy") == "prompt_uid_affinity"
    assert OmegaConf.select(cfg.actor_rollout_ref.rollout, "server_routing.routing_key_field", default="uid") == "uid"


def test_rollout_server_routing_config_override():
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "server_routing": {
                        "_target_": "verl_omni.workers.config.RolloutServerRoutingConfig",
                        "policy": "prompt_uid_affinity",
                    }
                }
            }
        }
    )
    assert OmegaConf.select(cfg.actor_rollout_ref.rollout, "server_routing.policy") == "prompt_uid_affinity"


def test_rollout_server_routing_config_rejects_invalid_policy():
    with pytest.raises(ValueError, match="Invalid rollout server_routing.policy"):
        RolloutServerRoutingConfig(policy="random_shuffle")


@pytest.mark.parametrize("policy", ["prompt_hash_sharding", "round_robin"])
def test_hash_and_round_robin_cover_all_servers(policy: str):
    lb = _make_lb(policy)
    counts: Counter[str] = Counter()
    for i in range(64):
        server_id, _ = ray.get(lb.acquire_server.remote(request_id=f"req-{i}", routing_key=f"uid-{i % 8}"))
        counts[server_id] += 1
        ray.get(lb.release_server.remote(server_id))
    assert len(counts) == 4
    assert all(count > 0 for count in counts.values())


def test_prompt_uid_affinity_clusters_rollout_copies():
    """FlowGRPO pattern: rollout.n copies of the same uid must share one replica."""
    lb = _make_lb("prompt_uid_affinity")
    per_uid_server: dict[str, str] = {}

    num_prompts = 32
    rollout_n = 16
    for uid in range(num_prompts):
        uid_key = f"prompt-{uid}"
        acquired_servers: list[str] = []
        for _ in range(rollout_n):
            server_id, _ = ray.get(lb.acquire_server.remote(request_id=f"req-{uid}-{_}", routing_key=uid_key))
            per_uid_server.setdefault(uid_key, server_id)
            assert per_uid_server[uid_key] == server_id
            acquired_servers.append(server_id)

        for server_id in acquired_servers:
            ray.get(lb.release_server.remote(server_id))

    assert len(per_uid_server) == num_prompts


def test_least_inflight_spreads_unique_request_ids():
    lb = _make_lb("least_inflight")
    server_load: Counter[str] = Counter()
    acquired_servers: list[str] = []
    for i in range(64):
        server_id, _ = ray.get(lb.acquire_server.remote(request_id=f"unique-req-{i}"))
        server_load[server_id] += 1
        acquired_servers.append(server_id)

    for server_id in acquired_servers:
        ray.get(lb.release_server.remote(server_id))

    assert len(server_load) == 4
    assert max(server_load.values()) - min(server_load.values()) <= 8
