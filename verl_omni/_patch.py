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
from __future__ import annotations

import sys

_PATCHED = False


def _patch_diffusion_agent_loop() -> None:
    """Alias ``verl.experimental.agent_loop.diffusion_agent_loop`` to the
    ``verl_omni`` implementation so that upstream lazy imports
    (e.g. inside ``AgentLoopManager.__init__``) resolve to it.

    Must be invoked *before* ``verl.experimental.agent_loop`` is imported
    for the first time, since that package's ``__init__`` eagerly imports
    ``diffusion_agent_loop``.
    """
    import verl_omni.experimental.agent_loop.diffusion_agent_loop as _omni_dal

    sys.modules["verl.experimental.agent_loop.diffusion_agent_loop"] = _omni_dal

    # Also set the attribute on the parent package if it has already been
    # imported, so ``from verl.experimental.agent_loop import
    # diffusion_agent_loop`` works correctly.
    parent = sys.modules.get("verl.experimental.agent_loop")
    if parent is not None:
        parent.diffusion_agent_loop = _omni_dal


def _patch_vllm_omni_replica() -> None:
    """Replace the upstream ``vllm_omni`` rollout replica registration so
    that verl-omni's ``vLLMOmniReplica`` (which uses verl-omni's
    ``DiffusionModelConfig`` / ``DiffusionRolloutConfig``) is used."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    def _load_vllm_omni():
        try:
            from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica
        except ImportError as err:
            raise ImportError("vllm-omni rollout requires vllm-omni to be installed.") from err

        return vLLMOmniReplica

    RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)


def _patch_diffusers_model() -> None:
    """Alias ``verl.models.diffusers_model`` (and its ``base`` / ``utils``
    submodules) to verl-omni's ``verl_omni.models.diffusion_model`` so
    that upstream code using the verl-side ``DiffusionModelBase`` registry
    (e.g. ``verl/workers/engine/fsdp/diffusers_impl.py``) resolves to the
    verl-omni registry where pipelines are actually registered."""
    import verl_omni.models.diffusion_model as _omni_pkg
    import verl_omni.models.diffusion_model.base as _omni_base
    import verl_omni.models.diffusion_model.utils as _omni_utils

    sys.modules["verl.models.diffusers_model"] = _omni_pkg
    sys.modules["verl.models.diffusers_model.base"] = _omni_base
    sys.modules["verl.models.diffusers_model.utils"] = _omni_utils

    parent = sys.modules.get("verl.models")
    if parent is not None:
        parent.diffusers_model = _omni_pkg


def apply_patches() -> None:
    """Apply all verl-omni compatibility patches.  Safe to call multiple
    times."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_diffusion_agent_loop()
    _patch_vllm_omni_replica()
    _patch_diffusers_model()
    _PATCHED = True
