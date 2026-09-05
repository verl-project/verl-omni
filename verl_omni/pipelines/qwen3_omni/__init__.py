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
"""Qwen3-Omni pipeline adapters (Thinker training + rollout pipeline topology)."""

import os

from .thinker_training_adapter import Qwen3OmniThinkerAdapter

__all__ = ["Qwen3OmniThinkerAdapter"]

# The explicit stage-config launcher does not use the rollout topology adapter.
# Keep that optional registration out of lightweight Ray actors because their
# vLLM-Omni checkout can legitimately predate the generated-pipeline API.
if os.environ.get("VERL_OMNI_SKIP_PIPELINES", "0").strip().lower() not in {"1", "true", "yes"}:
    from .omni_rollout_adapter import Qwen3OmniRolloutAdapter as Qwen3OmniRolloutAdapter

    __all__.append("Qwen3OmniRolloutAdapter")
