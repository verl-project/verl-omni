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
registered by importing :mod:`verl_omni.utils.skip.stage_skip` and instantiated
from the separate ``omni_skip:`` block (kept out of ``skip:`` because verl
parses that block against its fixed ``SkipManagerConfig`` schema). Conflict
checks here fail the run early for skip configurations that would be silently
inert or unsafe.
"""

from omegaconf import DictConfig, OmegaConf
from verl.utils.skip import SkipManager
from verl.utils.skip.base_skip import SKIP_REGISTRY

from verl_omni.utils.skip.stage_skip import STAGE_SKIP_ROLES

__all__ = ["init_skip_manager"]


def init_skip_manager(config: DictConfig) -> None:
    """Initialize SkipManager with verl's rollout roles plus verl-omni's stage roles."""
    import verl_omni.utils.skip.stage_skip  # noqa: F401 — registers the skip classes

    SkipManager.init(config)
    for role in STAGE_SKIP_ROLES:
        stage_config = OmegaConf.select(config, f"omni_skip.{role}")
        if stage_config is None:
            continue
        _validate_stage_config(role, stage_config)
        if stage_config.enable:
            SkipManager.skip_instances[role] = SKIP_REGISTRY[role](stage_config, config)
    _check_skip_conflicts(config)


def _validate_stage_config(role: str, stage_config: DictConfig) -> None:
    if not isinstance(stage_config.enable, bool):
        raise ValueError(f"omni_skip.{role}.enable must be bool, got {type(stage_config.enable).__name__}")
    if not isinstance(stage_config.dump_dir, str):
        raise ValueError(f"omni_skip.{role}.dump_dir must be str, got {type(stage_config.dump_dir).__name__}")
    steps = OmegaConf.to_container(stage_config.steps, resolve=True)
    if not isinstance(steps, list) or not all(
        isinstance(step, int) and not isinstance(step, bool) for step in steps
    ):
        raise ValueError(f"omni_skip.{role}.steps must be a list of ints, got {steps}")


def _check_skip_conflicts(config: DictConfig) -> None:
    enabled = [role for role in STAGE_SKIP_ROLES if role in SkipManager.skip_instances]
    if not enabled:
        return

    if "weight_sync" in enabled and not bool(
        OmegaConf.select(config, "omni_skip.weight_sync.ack_stale_weights", default=False)
    ):
        raise ValueError(
            "omni_skip.weight_sync leaves rollout replicas on pre-update weights until the next lifecycle "
            "sync, silently degrading training. Set omni_skip.weight_sync.ack_stale_weights=true to "
            "acknowledge (debug runs only)."
        )

    trainer_type = OmegaConf.select(config, "algorithm.trainer_type", default="policy_gradient")
    if trainer_type != "policy_gradient" and ("old_log_prob" in enabled or "advantage" in enabled):
        raise ValueError(
            f"omni_skip.old_log_prob / omni_skip.advantage are inert with algorithm.trainer_type={trainer_type}: "
            "direct-preference trainers never recompute old log probs or advantages."
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
