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
"""Bind diffusion V1 GEN hooks onto the Co-RL (Joint-Training) ``TaskRunnerV1`` owner.

RFC: one job. ``OmniBagelCoRLTrainerSync`` is the PPO V1 owner.
``PolicyGradientDiffusionTrainerV1`` methods run as a **lane** on the same
``actor_rollout_wg`` / config. Never ``fit()``, never a second ``optimizer.step``.

Import of ``trainer_base`` is deferred: some verl pins lack ``ReplayBufferAsync``.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["DiffusionV1GenLane", "diffusion_v1_gen_hooks", "gen_norm_adv_by_std"]


def _lane_config_with_gen_estimator(config: Any, gen_adv_estimator: str) -> Any:
    """Copy ``config`` with ``algorithm.adv_estimator`` set to the GEN estimator."""
    view = copy.deepcopy(config)
    try:
        from omegaconf import OmegaConf, open_dict

        if OmegaConf.is_config(view):
            with open_dict(view):
                if view.get("algorithm") is None:
                    view.algorithm = OmegaConf.create({})
                view.algorithm.adv_estimator = gen_adv_estimator
                gen_norm = gen_norm_adv_by_std(view.algorithm)
                if gen_norm is not None:
                    view.algorithm.norm_adv_by_std_in_grpo = gen_norm
            return view
    except ImportError:  # pragma: no cover - omegaconf is a hard dep in practice
        pass

    # Plain-namespace fallback (unit tests / non-Hydra owners).
    algorithm = copy.copy(getattr(view, "algorithm", None))
    if algorithm is None:
        algorithm = copy.copy(view)
    algorithm.adv_estimator = gen_adv_estimator
    gen_norm = gen_norm_adv_by_std(algorithm)
    if gen_norm is not None:
        algorithm.norm_adv_by_std_in_grpo = gen_norm
    view.algorithm = algorithm
    return view


def gen_norm_adv_by_std(algorithm: Any) -> bool | None:
    """The GEN-lane ``gen_norm_adv_by_std_in_grpo`` override, or ``None`` to inherit."""
    getter = getattr(algorithm, "get", None)
    value = getter("gen_norm_adv_by_std_in_grpo", None) if callable(getter) else None
    if value is None:
        value = getattr(algorithm, "gen_norm_adv_by_std_in_grpo", None)
    return None if value is None else bool(value)


def diffusion_v1_gen_hooks():
    """Return the three GEN methods from ``PolicyGradientDiffusionTrainerV1``."""
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    return (
        PolicyGradientDiffusionTrainerV1._compute_old_log_prob,
        PolicyGradientDiffusionTrainerV1._compute_ref_log_prob,
        PolicyGradientDiffusionTrainerV1._compute_advantage,
    )


class DiffusionV1GenLane:
    """GEN-only view of ``PolicyGradientDiffusionTrainerV1`` (no second trainer process)."""

    def __init__(self, owner: Any, *, gen_adv_estimator: str | None = None):
        self.config = owner.config
        self.gen_adv_estimator = None if gen_adv_estimator is None else str(gen_adv_estimator)
        if self.gen_adv_estimator is not None:
            self.config = _lane_config_with_gen_estimator(owner.config, self.gen_adv_estimator)
        self.actor_rollout_wg = owner.actor_rollout_wg
        self.ref_in_actor = bool(getattr(owner, "ref_in_actor", True))
        self.ref_policy_wg = getattr(owner, "ref_policy_wg", None)
        old, ref, adv = diffusion_v1_gen_hooks()
        self._compute_old_log_prob = old.__get__(self, DiffusionV1GenLane)
        self._compute_ref_log_prob = ref.__get__(self, DiffusionV1GenLane)
        self._compute_advantage = adv.__get__(self, DiffusionV1GenLane)
