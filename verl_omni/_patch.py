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

import logging
import sys

logger = logging.getLogger(__name__)

_PATCHED = False


def _patch_diffusion_agent_loop() -> None:
    """Alias ``verl.experimental.agent_loop.diffusion_agent_loop`` and
    ``verl.experimental.agent_loop.single_turn_agent_loop`` to the
    ``verl_omni`` implementations so that upstream lazy imports
    (e.g. inside ``AgentLoopManager.__init__``) resolve to them.
    TODO (mike): to be dropped
    """
    import verl_omni.agent_loop.diffusion_agent_loop as _omni_dal
    import verl_omni.agent_loop.single_turn_agent_loop  # noqa: F401

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
    ``DiffusionModelConfig`` / ``DiffusionRolloutConfig``) is used.
    TODO (mike): to be dropped
    """
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    def _load_vllm_omni():
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

        return vLLMOmniReplica

    RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)


def _patch_diffusers_model() -> None:
    """Alias ``verl.models.diffusers_model`` (and its ``base`` / ``utils``
    submodules) to verl-omni's ``verl_omni.pipelines`` so
    that upstream code using the verl-side ``DiffusionModelBase`` registry
    (e.g. ``verl/workers/engine/fsdp/diffusers_impl.py``) resolves to the
    verl-omni registry where pipelines are actually registered.
    TODO (mike): to be dropped
    """
    import verl_omni.pipelines as _omni_pkg
    import verl_omni.pipelines.model_base as _omni_base
    import verl_omni.pipelines.utils as _omni_utils

    sys.modules["verl.models.diffusers_model"] = _omni_pkg
    sys.modules["verl.models.diffusers_model.base"] = _omni_base
    sys.modules["verl.models.diffusers_model.utils"] = _omni_utils

    parent = sys.modules.get("verl.models")
    if parent is not None:
        parent.diffusers_model = _omni_pkg


def _patch_fsdp_diffusers_engine() -> None:
    """Alias ``verl.workers.engine.fsdp.diffusers_impl`` to verl-omni's
    implementation so that ``EngineRegistry`` resolves the
    ``diffusion_model`` engine to verl-omni's ``DiffusersFSDPEngine``.
    TODO (mike): to be dropped
    """
    # Force-trigger verl's eager import chain so that upstream's
    # ``DiffusersFSDPEngine`` registers first.
    import verl.workers.engine  # noqa: F401
    from verl.workers.engine.base import EngineRegistry

    # Drop upstream's ``diffusion_model`` registration so that verl-omni's
    # decorator (which would otherwise be a no-op) wins.
    EngineRegistry._engines.pop("diffusion_model", None)

    # Import verl-omni's implementation to (re-)register its engine.
    import verl_omni.workers.engine.fsdp.diffusers_impl as _omni_impl

    # Alias the module path so any direct references resolve to ours.
    sys.modules["verl.workers.engine.fsdp.diffusers_impl"] = _omni_impl
    parent = sys.modules.get("verl.workers.engine.fsdp")
    if parent is not None:
        parent.diffusers_impl = _omni_impl


def _patch_visual_reward_manager() -> None:
    """Replace the ``"visual"`` entry in verl's ``REWARD_MANAGER`` registry
    with verl-omni's ``VisualRewardManager``, which uses verl-omni's
    ``default_compute_score_image`` dispatcher (supporting reward score
    functions defined in ``verl_omni.utils.reward_score``).
    TODO (mike): to be dropped
    """
    import verl.experimental.reward_loop.reward_manager  # noqa: F401 — triggers verl's @register("visual")
    from verl.experimental.reward_loop.reward_manager.registry import REWARD_MANAGER

    from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager as _OmniVisual

    REWARD_MANAGER["visual"] = _OmniVisual


def apply_patches() -> None:
    """Apply all verl-omni compatibility patches.  Safe to call multiple
    times."""
    global _PATCHED
    if _PATCHED:
        return
    logger.warning(
        "Applying verl-omni monkey-patches to override upstream verl's legacy "
        "diffusion implementations. These patches are temporary and will be "
        "dropped once upstream verl no longer ships its legacy diffusion code."
    )
    _patch_diffusion_agent_loop()
    _patch_vllm_omni_replica()
    _patch_diffusers_model()
    _patch_fsdp_diffusers_engine()
    _patch_visual_reward_manager()
    _PATCHED = True
