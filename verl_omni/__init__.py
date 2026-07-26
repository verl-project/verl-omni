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
import importlib
import logging
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


logger = logging.getLogger(__name__)


# Fallback for CPU-only environments where vLLM-Omni current_omni_platform.device_type is empty.
# This prevents RuntimeError: Device string must not be empty when importing modules with torch.amp.autocast.
try:
    import vllm_omni.platforms

    if not vllm_omni.platforms.current_omni_platform.device_type:
        vllm_omni.platforms.current_omni_platform.device_type = "cpu"
except Exception:
    pass


def _safe_import(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        logger.warning("Skipping optional module import %s due to missing dependency: %s", module_name, exc)


# Import pipelines / rollout / reward loop / engines to auto-register them.
# Some of these modules depend on optional rollout stacks such as vLLM/vLLM-Omni.
# Keep basic CPU-side imports working when those extras are not installed.
for _module_name in (
    "verl_omni.experimental",
    "verl_omni.models",
    "verl_omni.pipelines",
    "verl_omni.reward_loop",
    "verl_omni.workers.engine",
    "verl_omni.workers.rollout",
):
    _safe_import(_module_name)
