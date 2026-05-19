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
"""Diffusion-specific loss functions and KL penalties."""

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl_omni.workers.config import DiffusionActorConfig

DiffusionLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        Optional[DictConfig | DiffusionActorConfig],  # config
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

DIFFUSION_LOSS_REGISTRY: dict[str, DiffusionLossFn] = {}


def register_diffusion_loss(name: str) -> Callable[[DiffusionLossFn], DiffusionLossFn]:
    """Register a diffusion loss function with the given name.

    Args:
        name (str): The name to register the diffusion loss function under.

    Returns:
        function: Decorator function that registers the diffusion loss function.
    """

    def decorator(func: DiffusionLossFn) -> DiffusionLossFn:
        DIFFUSION_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_diffusion_loss_fn(name):
    """Get the diffusion loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    if name not in DIFFUSION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported diffusion loss mode: {name}. Supported modes are: {list(DIFFUSION_LOSS_REGISTRY.keys())}"
        )
    return DIFFUSION_LOSS_REGISTRY[name]


class DiffusionAdvantageEstimator(str, Enum):
    """Advantage estimators specific to diffusion-based training."""

    FLOW_GRPO = "flow_grpo"


DIFFUSION_ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_diffusion_adv_est(name_or_enum: str | DiffusionAdvantageEstimator) -> Any:
    """Register a diffusion advantage estimator function with the given name.

    Args:
        name_or_enum: `(str)` or `(DiffusionAdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in DIFFUSION_ADV_ESTIMATOR_REGISTRY and DIFFUSION_ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Diffusion adv estimator {name} has already been registered: "
                f"{DIFFUSION_ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        DIFFUSION_ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_diffusion_adv_estimator_fn(name_or_enum):
    """Get the diffusion advantage estimator function with a given name."""
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in DIFFUSION_ADV_ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown diffusion advantage estimator: {name}. Supported: {list(DIFFUSION_ADV_ESTIMATOR_REGISTRY.keys())}"
        )
    return DIFFUSION_ADV_ESTIMATOR_REGISTRY[name]


@register_diffusion_adv_est(DiffusionAdvantageEstimator.FLOW_GRPO)
def compute_flow_grpo_outcome_advantage(
    sample_level_rewards: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-4,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DictConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        sample_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        global_std: `(bool)`
            whether to use global std for advantage normalization
        config: `(Optional[DictConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = sample_level_rewards.clone()
    assert scores.ndim == 2
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        if global_std:
            batch_std = torch.std(scores)
        else:
            batch_std = None

        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = id2score[idx][0]
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    return scores, scores


@register_diffusion_loss("flow_grpo")
def compute_diffusion_loss_flow_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    config: Optional[DictConfig | DiffusionActorConfig] = None,
    *,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective and related metrics for FlowGRPO.

    Adapted from
    https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_fast.py#L885

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size,).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size,).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size,).
        config (verl_omni.workers.config.DiffusionActorConfig):
            Config for the actor.
        rollout_is_weights (Optional[torch.Tensor]):
            Optional Rollout Correction multiplier (same shape as ``log_prob``) combining
            IS weights and RS rejection (rejected samples have weight 0). When provided,
            the per-element policy loss is multiplied by these (detached) weights before
            the mean reduction.
    """
    assert config is not None
    assert isinstance(config, DiffusionActorConfig)
    loss_cfg = config.diffusion_loss
    advantages = torch.clamp(
        advantages,
        -loss_cfg.adv_clip_max,
        loss_cfg.adv_clip_max,
    )
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - loss_cfg.clip_ratio,
        1.0 + loss_cfg.clip_ratio,
    )
    per_elem_loss = torch.maximum(unclipped_loss, clipped_loss)
    if rollout_is_weights is not None:
        per_elem_loss = per_elem_loss * rollout_is_weights.detach()
    pg_loss = torch.mean(per_elem_loss)

    with torch.no_grad():
        ppo_kl = torch.mean(-log_ratio)
        pg_clipfrac = torch.mean((torch.abs(ratio - 1.0) > loss_cfg.clip_ratio).float())
        pg_clipfrac_higher = torch.mean((ratio - 1.0 > loss_cfg.clip_ratio).float())
        pg_clipfrac_lower = torch.mean((1.0 - ratio > loss_cfg.clip_ratio).float())
        ratio_mean = ratio.mean()
        ratio_std = ratio.std()

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ratio_mean": ratio_mean.detach().item(),
        "actor/ratio_std": ratio_std.detach().item(),
    }
    return pg_loss, pg_metrics


@register_diffusion_loss("grpo_guard")
def compute_diffusion_loss_grpo_guard(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    config: Optional[DictConfig | DiffusionActorConfig] = None,
    *,
    old_prev_sample_mean: Optional[torch.Tensor] = None,
    prev_sample_mean: Optional[torch.Tensor] = None,
    std_dev_t: Optional[torch.Tensor] = None,
    sqrt_dt: Optional[torch.Tensor] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the GRPO-Guard policy objective.

    GRPO-Guard (https://arxiv.org/abs/2510.22319) augments the standard
    Flow-GRPO importance ratio with a "ratio-mean bias" term that explicitly
    penalises drift in the reverse-SDE proposal mean of the current policy
    relative to the rollout policy. The mean drift is then projected onto the
    same scale as ``log_prob - old_log_prob`` via the per-step diffusion
    coefficient ``sqrt_dt * sigma_t``, and the final policy loss is rescaled
    by ``1 / sqrt_dt**2`` so that gradients have a consistent magnitude across
    timesteps.

    Args:
        old_log_prob (torch.Tensor): Log-probabilities under the old policy,
            shape ``(B,)``.
        log_prob (torch.Tensor): Log-probabilities under the current policy,
            shape ``(B,)``.
        advantages (torch.Tensor): Advantage estimates, shape ``(B,)``.
        config: Actor configuration; ``diffusion_loss.clip_ratio`` and
            ``diffusion_loss.adv_clip_max`` are read from it.
        old_prev_sample_mean (torch.Tensor): Reverse-SDE mean from the rollout
            policy, shape ``(B, ...)``.
        prev_sample_mean (torch.Tensor): Reverse-SDE mean from the current
            policy, shape ``(B, ...)``.
        std_dev_t (torch.Tensor): Per-step SDE standard deviation, shape
            ``(B, 1, 1, ...)`` or scalar.
        sqrt_dt (torch.Tensor): ``sqrt(-dt)`` for the current denoising step,
            shape ``(B,)`` or scalar.
    """
    assert config is not None
    assert isinstance(config, DiffusionActorConfig)
    assert old_prev_sample_mean is not None, "GRPO-Guard requires `old_prev_sample_mean`"
    assert prev_sample_mean is not None, "GRPO-Guard requires `prev_sample_mean`"
    assert std_dev_t is not None, "GRPO-Guard requires `std_dev_t`"
    assert sqrt_dt is not None, "GRPO-Guard requires `sqrt_dt`"

    loss_cfg = config.diffusion_loss
    advantages = torch.clamp(
        advantages,
        -loss_cfg.adv_clip_max,
        loss_cfg.adv_clip_max,
    )

    sigma_t = std_dev_t.mean()
    sqrt_dt_mean = sqrt_dt.mean()
    scale = sqrt_dt_mean * sigma_t  # shared per-step scalar

    # mean over all non-batch dimensions: (B, ...) -> (B,)
    mean_diff_sq = (prev_sample_mean - old_prev_sample_mean).pow(2)
    if mean_diff_sq.ndim > 1:
        mean_diff_sq = mean_diff_sq.mean(dim=tuple(range(1, mean_diff_sq.ndim)))
    ratio_mean_bias = mean_diff_sq / (2 * scale**2)

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp((log_ratio + ratio_mean_bias) * scale)

    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - loss_cfg.clip_ratio,
        1.0 + loss_cfg.clip_ratio,
    )
    per_elem_loss = torch.maximum(unclipped_loss, clipped_loss)
    if rollout_is_weights is not None:
        per_elem_loss = per_elem_loss * rollout_is_weights.detach()
    pg_loss = torch.mean(per_elem_loss) / (sqrt_dt_mean**2)

    with torch.no_grad():
        ppo_kl = torch.mean(-log_ratio)
        pg_clipfrac = torch.mean((torch.abs(ratio - 1.0) > loss_cfg.clip_ratio).float())
        pg_clipfrac_higher = torch.mean((ratio - 1.0 > loss_cfg.clip_ratio).float())
        pg_clipfrac_lower = torch.mean((1.0 - ratio > loss_cfg.clip_ratio).float())
        ratio_mean = ratio.mean()
        ratio_std = ratio.std()

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ratio_mean": ratio_mean.detach().item(),
        "actor/ratio_std": ratio_std.detach().item(),
    }
    return pg_loss, pg_metrics


def kl_penalty_image(
    prev_sample_mean: torch.Tensor, ref_prev_sample_mean: torch.Tensor, std_dev_t: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence given previous sample mean and reference previous sample mean (for images or videos).
    Args:
        prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
        ref_prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
        std_dev_t: (torch.Tensor) shape is (bs, 1, 1)
    """
    kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1, 2), keepdim=True) / (2 * std_dev_t**2)
    return kl_loss.mean()
