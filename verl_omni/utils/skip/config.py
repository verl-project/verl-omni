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
"""Config schema of the ``omni_skip:`` block, mirroring verl's ``SkipManagerConfig``."""

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = ["ActorUpdateSkipConfig", "OmniSkipConfig", "StageSkipConfig", "WeightSyncSkipConfig"]


@dataclass
class StageSkipConfig(BaseConfig):
    """Skip settings of one pipeline stage.

    Args:
        enable: Whether the stage's skip role is active.
        dump_dir: Root directory of the stage caches.
        steps: Global steps to replay or skip; every other step runs for real.
        action: ``cache`` (exact step), ``repeat`` (closest dumped step), or ``empty`` where the role supports it.

    Raises:
        ValueError: If the role is enabled with no steps, which could never fire.
    """

    enable: bool = False
    dump_dir: str = "~/.verl/stage_dump"
    steps: list[int] = field(default_factory=list)
    action: str = "cache"

    def __post_init__(self) -> None:
        if self.enable and not self.steps:
            raise ValueError("An enabled omni_skip role needs a non-empty steps list: an empty list never fires.")


@dataclass
class ActorUpdateSkipConfig(StageSkipConfig):
    """Skip settings of the actor update, which also supports the ``empty`` action."""

    action: str = "empty"


@dataclass
class WeightSyncSkipConfig(StageSkipConfig):
    """Skip settings of the mid-cycle weight sync.

    Args:
        ack_stale_weights: Must be true when enabled: replicas keep pre-update weights until the next sync.

    Raises:
        ValueError: If the role is enabled without the acknowledgment.
    """

    action: str = "empty"
    ack_stale_weights: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.enable and not self.ack_stale_weights:
            raise ValueError(
                "omni_skip.weight_sync leaves rollout replicas on pre-update weights until the next lifecycle "
                "sync, silently degrading training. Set omni_skip.weight_sync.ack_stale_weights=true to "
                "acknowledge (debug runs only)."
            )


@dataclass
class OmniSkipConfig(BaseConfig):
    """Top-level ``omni_skip:`` config: one entry per stage role."""

    reward: StageSkipConfig = field(default_factory=StageSkipConfig)
    old_log_prob: StageSkipConfig = field(default_factory=StageSkipConfig)
    ref_log_prob: StageSkipConfig = field(default_factory=StageSkipConfig)
    advantage: StageSkipConfig = field(default_factory=StageSkipConfig)
    actor_update: ActorUpdateSkipConfig = field(default_factory=ActorUpdateSkipConfig)
    weight_sync: WeightSyncSkipConfig = field(default_factory=WeightSyncSkipConfig)
