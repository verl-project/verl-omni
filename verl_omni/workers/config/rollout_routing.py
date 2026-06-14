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

from dataclasses import dataclass
from typing import Optional

from verl.base_config import BaseConfig

VALID_ROLLOUT_SERVER_ROUTING_POLICIES = frozenset(
    {
        "least_inflight",
        "prompt_uid_affinity",
        "prompt_hash_sharding",
        "round_robin",
    }
)


@dataclass
class RolloutServerRoutingConfig(BaseConfig):
    """How agent loops route HTTP rollout requests across omni server replicas.

    This applies to any async rollout using ``OmniLLMServerClient`` (diffusion
    and omni LLM). It controls **verl-side replica selection**, not the internal
    vllm-omni diffusion scheduler policy (``scheduling_policy``).
    """

    # Replica routing policy:
    # - least_inflight: spread requests for fairness (verl default)
    # - prompt_uid_affinity: sticky route by routing_key (e.g. batch uid)
    # - prompt_hash_sharding: deterministic shard by hash(routing_key)
    # - round_robin: rotate replicas regardless of load
    policy: str = "least_inflight"

    # Field on per-sample agent kwargs used as routing_key (e.g. ``uid`` for
    # FlowGRPO rollout.n copies). Ignored when null.
    routing_key_field: Optional[str] = "uid"

    def __post_init__(self) -> None:
        if self.policy not in VALID_ROLLOUT_SERVER_ROUTING_POLICIES:
            raise ValueError(
                f"Invalid rollout server_routing.policy={self.policy!r}. "
                f"Must be one of {sorted(VALID_ROLLOUT_SERVER_ROUTING_POLICIES)}."
            )
