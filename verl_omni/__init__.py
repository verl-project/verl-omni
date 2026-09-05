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
import logging
import os

logger = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


# Apply model patches and auto-register the components requested by the caller.
# Lightweight or compatibility-sensitive launchers may opt out of unrelated
# registrations before importing verl_omni as an external module.
def _registration_enabled(skip_env: str) -> bool:
    return os.environ.get(skip_env, "0").strip().lower() not in {"1", "true", "yes"}


if _registration_enabled("VERL_OMNI_SKIP_AGENT_LOOP"):
    import verl_omni.agent_loop  # noqa: E402, F401
if _registration_enabled("VERL_OMNI_SKIP_MODELS"):
    import verl_omni.models  # noqa: E402, F401

if _registration_enabled("VERL_OMNI_SKIP_PIPELINES"):
    import verl_omni.pipelines  # noqa: E402, F401
if _registration_enabled("VERL_OMNI_SKIP_REWARD_LOOP"):
    import verl_omni.reward_loop  # noqa: E402, F401
if _registration_enabled("VERL_OMNI_SKIP_TRAINER"):
    import verl_omni.trainer  # noqa: E402, F401
if _registration_enabled("VERL_OMNI_SKIP_ENGINES"):
    import verl_omni.workers.engine  # noqa: E402, F401
import verl_omni.workers.rollout  # noqa: E402, F401
