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
"""Configuration for omni rollout."""

from dataclasses import dataclass
from typing import Optional

from verl.workers.config.rollout import RolloutConfig

__all__ = ["OmniRolloutConfig"]


@dataclass
class OmniRolloutConfig(RolloutConfig):
    """Rollout config with vLLM-Omni pipeline metadata."""

    _mutable_fields = RolloutConfig._mutable_fields | {
        "pipeline_mode",
        "stage_configs_path",
        "deploy_config",
        "model_type",
    }

    pipeline_mode: str = "thinker_only"
    model_type: Optional[str] = None
    stage_configs_path: Optional[str] = None
    deploy_config: Optional[str] = None
