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


def _load_verl_plugins() -> None:
    """Load verl's ``verl.plugins`` entry points before the imports below.

    verl imports the modules in ``VERL_USE_EXTERNAL_MODULES`` before it loads its
    plugin entry points, and the engine modules imported below read the device at
    import time. With ``VERL_USE_EXTERNAL_MODULES=verl_omni`` a hardware plugin
    would therefore register its platform too late. Loading the entry points here,
    with verl's ``VERL_USE_EXTERNAL_PLUGINS`` policy, keeps the order right; verl's
    own pass afterwards finds the plugins already imported.
    """
    import verl  # noqa: F401 -- when verl is imported first, its own pass has already run

    policy = os.getenv("VERL_USE_EXTERNAL_PLUGINS", "auto").strip().lower()
    if policy == "none":
        return
    allowed = None if policy == "auto" else {name.strip() for name in policy.split(",") if name.strip()}

    from importlib.metadata import entry_points

    for ep in entry_points(group="verl.plugins"):
        if allowed is not None and ep.name not in allowed:
            continue
        try:
            ep.load()
        except Exception as e:
            logger.debug("Failed to load plugin '%s': %s", ep.name, e)


_load_verl_plugins()


# Apply model patches and auto-register agent loops / pipelines / rollout / reward loop / engines
import verl_omni.agent_loop  # noqa: E402, F401
import verl_omni.models  # noqa: E402, F401
import verl_omni.pipelines  # noqa: E402, F401
import verl_omni.reward_loop  # noqa: E402, F401
import verl_omni.trainer  # noqa: E402, F401
import verl_omni.workers.checkpoint_engine  # noqa: E402, F401 -- omni_delta_sharded backend registration
import verl_omni.workers.engine  # noqa: E402, F401
import verl_omni.workers.rollout  # noqa: E402, F401
