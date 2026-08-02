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
"""Static preflight for the diffusion teacher runtime.

Everything decidable from the config alone, checked before ``ray.init`` so a
misconfigured run fails at the command line instead of drifting until a batch key
is missing. Checks that need a constructed config, a built scheduler, or a live
worker belong to the later tiers, not here.

Bound at the top of ``run_diffusion()`` and ``run_diffusion_v1()`` rather than in
``main()`` or the TaskRunner: two Hydra entrypoints exist and only one dispatches
on ``use_v1``, so the run functions are the single place that covers the
canonical v0 CLI, the v1 dispatcher, and programmatic callers alike.
"""

from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.workers.config.diffusion import DiffusionTeacherConfig

__all__ = ["validate_teacher_preflight", "DISTILL_MODES"]

# The distillation loss modes that consume the teacher_* batch keys.
DISTILL_MODES = ("distill_kl", "distill_fm_mse")

FSDP_STRATEGIES = ("fsdp", "fsdp2")


def _selected_distill_modes(actor) -> set[str]:
    """Distillation losses this run would compute, standalone or auxiliary."""
    modes = set()
    if actor.diffusion_loss.loss_mode in DISTILL_MODES:
        modes.add(actor.diffusion_loss.loss_mode)
    if actor.use_distill_loss and actor.distill_loss_mode in DISTILL_MODES:
        modes.add(actor.distill_loss_mode)
    return modes


def validate_teacher_preflight(config, stack: str) -> None:
    """Reject teacher configurations this runtime cannot serve.

    Args:
        config: the composed trainer config.
        stack: which trainer stack is actually executing, ``"v0"`` or ``"v1"``.
            Passed explicitly rather than read from ``trainer.use_v1``: a caller
            can invoke ``run_diffusion_v1(cfg)`` with the flag still false, so the
            flag states intent while the argument states fact.
    """
    # verl's top-level namespace drives the *token* distillation loss, and the
    # actor worker dispatches on it before it dispatches on modality -- arming it
    # on a diffusion run silently swaps diffusion_loss for distillation_ppo_loss.
    if OmegaConf.select(config, "distillation.enabled", default=False):
        raise ValueError(
            "Top-level `distillation.enabled` is verl's token distillation switch and must stay "
            "false on a diffusion run: the actor worker checks it before the model-type dispatch, "
            "so it would replace diffusion_loss with the token loss. The diffusion teacher runtime "
            "lives at `actor_rollout_ref.teacher.*`."
        )

    actor = config.actor_rollout_ref.actor
    selected_modes = _selected_distill_modes(actor)
    teacher_enabled = bool(OmegaConf.select(config, "actor_rollout_ref.teacher.enabled", default=False))

    if selected_modes and not teacher_enabled:
        raise ValueError(
            f"Distillation loss {sorted(selected_modes)} is selected but "
            "`actor_rollout_ref.teacher.enabled` is false, so nothing produces the teacher_* keys "
            "it reads. Enable the teacher or select a non-distillation loss."
        )
    if teacher_enabled and not selected_modes:
        raise ValueError(
            "`actor_rollout_ref.teacher.enabled` is set but no distillation loss consumes its "
            f"output: set actor.diffusion_loss.loss_mode to one of {list(DISTILL_MODES)}, or set "
            "actor.use_distill_loss with actor.distill_loss_mode."
        )

    if not teacher_enabled:
        return

    # Constructing the dataclass is what validates `models` and `placement`; the
    # `_target_` route would leave the entries as bare dicts and skip both.
    teacher = omega_conf_to_dataclass(config.actor_rollout_ref.teacher, DiffusionTeacherConfig)

    if stack == "v1":
        raise ValueError(
            "The v1 diffusion trainer does not wire the teacher runtime. Run the v0 trainer "
            "(`trainer.use_v1=false`) or disable `actor_rollout_ref.teacher.enabled`."
        )
    if config.trainer.get("use_v1", False):
        raise ValueError(
            "`trainer.use_v1` selects the v1 diffusion trainer, which does not wire the teacher "
            "runtime. Unset it or disable `actor_rollout_ref.teacher.enabled`."
        )

    if config.algorithm.trainer_type != "policy_gradient":
        raise ValueError(
            f"The diffusion teacher runtime requires algorithm.trainer_type='policy_gradient', got "
            f"{config.algorithm.trainer_type!r}. Only the policy-gradient trainer has a teacher hook."
        )
    if config.algorithm.sample_source != "online":
        raise ValueError(
            f"The diffusion teacher runtime requires algorithm.sample_source='online', got "
            f"{config.algorithm.sample_source!r}. Offline sampling skips the rollout stack, so there "
            "is no student trajectory for the teacher to replay."
        )

    if "distill_fm_mse" in selected_modes:
        raise ValueError(
            "`distill_fm_mse` needs `teacher_noise_pred`, which no producer emits on the "
            "policy-gradient path -- that path's prepare_model_outputs computes no noise_pred at "
            "all. Its producer arrives with the direct-preference follow-up; use `distill_kl`."
        )

    if actor.strategy not in FSDP_STRATEGIES:
        raise ValueError(
            f"The diffusion teacher runtime requires actor_rollout_ref.actor.strategy in "
            f"{list(FSDP_STRATEGIES)}, got {actor.strategy!r}. The teacher backend is pinned to "
            "FSDP, and the two engine configs disagree on the sequence-parallel field name, so a "
            "mixed-backend run has no validated path."
        )
    for key, entry in teacher.models.items():
        if entry.engine.strategy not in FSDP_STRATEGIES:
            raise ValueError(
                f"Teacher {key!r}: engine.strategy must be one of {list(FSDP_STRATEGIES)}, got "
                f"{entry.engine.strategy!r}. PR A implements the FSDP teacher engine only."
            )
