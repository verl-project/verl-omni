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

from verl_omni.utils.vllm_omni.patch import patch_qwen3_tts_default_rope_read

from .omni_rollout_adapter import Qwen3TTSRolloutAdapter
from .talker_training_adapter import Qwen3TTSTalkerAdapter

# qwen_tts installs a process-global default-rope initializer; the wrapper makes
# its bare rope_theta read tolerate configs that moved the value (see patch).
patch_qwen3_tts_default_rope_read()

__all__ = ["Qwen3TTSTalkerAdapter", "Qwen3TTSRolloutAdapter"]
