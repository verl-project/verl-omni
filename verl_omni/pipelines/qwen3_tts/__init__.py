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

from transformers.models.mimi.configuration_mimi import MimiConfig

from .omni_rollout_adapter import Qwen3TTSRolloutAdapter
from .talker_training_adapter import Qwen3TTSTalkerAdapter

# The pinned qwen-tts installs a process-global default-rope initializer that
# reads the bare rope_theta attribute; transformers 5.19 moved Mimi's theta
# into rope_parameters, so restore the attribute the hook reads.
MimiConfig.rope_theta = 10_000.0

__all__ = ["Qwen3TTSTalkerAdapter", "Qwen3TTSRolloutAdapter"]
