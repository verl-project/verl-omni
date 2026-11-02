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
"""SkipManager initialization for verl-omni trainers.

verl's ``SkipManager.init`` instantiates only the roles it knows from the
``skip:`` config block. This module wraps it: the verl-omni stage roles are
instantiated from the separate ``omni_skip:`` block (kept out of ``skip:``
because verl parses that block against its fixed ``SkipManagerConfig`` schema).
Conflict checks here fail the run early for skip configurations that would be
silently inert or unsafe.
"""

from omegaconf import DictConfig, OmegaConf
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip import SkipManager
from verl.utils.skip.base_skip import SKIP_REGISTRY

from verl_omni.utils.skip.config import OmniSkipConfig
from verl_omni.utils.skip.stage_skip import STAGE_SKIP_ROLES
from verl_omni.workers.config.reward import streaming_reward_enabled

__all__ = ["init_skip_manager"]


def init_skip_manager(config: DictConfig) -> None:
    """Initialize SkipManager with verl's rollout roles plus verl-omni's stage roles.

    Args:
        config: The whole trainer config; a missing ``omni_skip`` block enables no stage role.

    Raises:
        ValueError: If an enabled role is invalid, conflicting, or could never fire with this config.
    """
    SkipManager.init(config)
    if OmegaConf.select(config, "omni_skip") is None:
        return
    stage_configs = omega_conf_to_dataclass(config.omni_skip, dataclass_type=OmniSkipConfig)
    for role in STAGE_SKIP_ROLES:
        stage_config = getattr(stage_configs, role)
        if stage_config.enable:
            SkipManager.skip_instances[role] = SKIP_REGISTRY[role](stage_config, config)
    _check_skip_conflicts(config)


def _check_skip_conflicts(config: DictConfig) -> None:
    enabled = {role for role in STAGE_SKIP_ROLES if role in SkipManager.skip_instances}
    if not enabled:
        return

    use_v1 = bool(OmegaConf.select(config, "trainer.use_v1", default=False))
    if "weight_sync" in enabled and (
        not use_v1 or OmegaConf.select(config, "trainer.v1.trainer_mode", default="sync") == "sync"
    ):
        raise ValueError(
            "omni_skip.weight_sync is inert here: only the V1 async modes have a mid-cycle sync. "
            "Set trainer.use_v1=true with a non-sync trainer.v1.trainer_mode, or disable the role."
        )
    if enabled & {"reward", "weight_sync"} and streaming_reward_enabled(config):
        raise ValueError(
            "omni_skip.reward / omni_skip.weight_sync are inert without a colocated reward model: streaming "
            "rewards are scored inside rollout, so freeze them with skip.rollout_tq / skip.rollout instead."
        )
    if "ref_log_prob" in enabled:
        loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.diffusion_loss.loss_mode")
        if not need_reference_policy(config) and loss_mode != "dpo":
            raise ValueError("omni_skip.ref_log_prob is inert: this configuration runs no reference policy.")

    trainer_type = OmegaConf.select(config, "algorithm.trainer_type", default="policy_gradient")
    if trainer_type != "policy_gradient" and enabled & {"old_log_prob", "advantage"}:
        raise ValueError(
            f"omni_skip.old_log_prob / omni_skip.advantage are inert with algorithm.trainer_type={trainer_type}: "
            "direct-preference trainers never recompute old log probs or advantages."
        )
    if "advantage" in enabled and not use_v1:
        raise ValueError(
            "omni_skip.advantage is inert on the V0 policy-gradient trainer: it computes advantages inline "
            "with no annotated stage hook. Enable it only with trainer.use_v1=true."
        )

    if "old_log_prob" in enabled:
        if bool(OmegaConf.select(config, "algorithm.rollout_correction.bypass_mode", default=False)):
            raise ValueError(
                "omni_skip.old_log_prob conflicts with algorithm.rollout_correction.bypass_mode: both "
                "substitute old log probs, so the values actually used would be ambiguous. Disable one."
            )
        from verl_omni.trainer.diffusion.diffusion_trainer_utils import skip_old_log_prob

        if skip_old_log_prob(config):
            raise ValueError(
                "omni_skip.old_log_prob is inert: the configured diffusion loss reads no recomputed old "
                "log probs, so the stage call never fires."
            )
