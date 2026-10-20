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
"""verl-omni SkipManager extensions: per-stage skip for the diffusion RL pipeline."""

from verl_omni.utils.skip.manager import STAGE_SKIP_ROLES, init_skip_manager
from verl_omni.utils.skip.stage_skip import skip_stage_metrics

__all__ = ["STAGE_SKIP_ROLES", "init_skip_manager", "skip_stage_metrics"]
