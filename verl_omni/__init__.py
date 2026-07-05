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
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


# Fallback for CPU-only environments where vLLM-Omni current_omni_platform.device_type is empty.
# This prevents RuntimeError: Device string must not be empty when importing modules with torch.amp.autocast.
try:
    import vllm_omni.platforms

    if not vllm_omni.platforms.current_omni_platform.device_type:
        vllm_omni.platforms.current_omni_platform.device_type = "cpu"
except Exception:
    pass


# Apply model patches and auto-register pipelines / rollout / reward loop / engines
from verl_omni.utils.hf_processor_patch import install_auto_patch  # noqa: E402, F401

install_auto_patch()
import verl_omni.experimental  # noqa: E402, F401
import verl_omni.models  # noqa: E402, F401
import verl_omni.pipelines  # noqa: E402, F401
import verl_omni.reward_loop  # noqa: E402, F401
import verl_omni.workers.engine  # noqa: E402, F401
import verl_omni.workers.rollout  # noqa: E402, F401
