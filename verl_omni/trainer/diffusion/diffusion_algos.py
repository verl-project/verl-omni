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

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl_omni.workers.config import DiffusionActorConfig


@dataclass
class DiffusionLossResult:
    """Output from a batch-aware diffusion loss function."""

    loss: torch.Tensor
    metrics: dict[str, Any]
    add_loss_metric: bool = True


def _format_available_keys(mapping: Any) -> str:
    try:
        keys = sorted(str(key) for key in mapping.keys())
    except AttributeError:
        return f"<{type(mapping).__name__} has no keys()>"
    return "[" + ", ".join(keys) + "]"


class DiffusionLossFn(ABC):
    """Abstract base for worker-side diffusion loss functions."""

    # Keys that must be present in ``model_output`` (tensors from the actor
    # forward pass, e.g. ``log_probs``). Subclasses override this so
    # ``validate_inputs`` can fail fast with a clear error when a pipeline
    # adapter does not populate everything the loss needs.
    required_model_output_keys: tuple[str, ...] = ()
    # Keys that must be present in ``data`` (batch tensors from rollout /
    # trainer, e.g. ``old_log_probs``, ``advantages``). Same early-check
    # contract as ``required_model_output_keys`` for batch-side inputs.
    required_data_keys: tuple[str, ...] = ()

    def validate_inputs(
        self,
        *,
        loss_name: str,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> None:
        """Validate that the worker batch contains inputs required by this loss."""

        missing_model_output = [key for key in self.required_model_output_keys if key not in model_output]
        missing_data = [key for key in self.required_data_keys if key not in data]
        if not missing_model_output and not missing_data:
            return

        details = [f"Diffusion loss `{loss_name}` is missing required inputs."]
        if missing_model_output:
            details.append(f"Missing model_output keys: {missing_model_output}.")
            details.append(f"Available model_output keys: {_format_available_keys(model_output)}.")
        if missing_data:
            details.append(f"Missing data keys: {missing_data}.")
            details.append(f"Available data keys: {_format_available_keys(data)}.")
        raise KeyError(" ".join(details))

    @classmethod
    @abstractmethod
    def compute_loss(cls, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the pure mathematical loss and related metrics.

        Subclasses define concrete tensor arguments (e.g. ``old_log_prob``,
        ``log_prob``, ``advantages``) in their implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        """Compute loss and metrics from the worker batch."""
        raise NotImplementedError


DIFFUSION_LOSS_REGISTRY: dict[str, DiffusionLossFn] = {}


def register_diffusion_loss(name: str) -> Callable[[type[DiffusionLossFn]], type[DiffusionLossFn]]:
    """Register a worker-side diffusion loss function class."""

    def decorator(cls: type[DiffusionLossFn]) -> type[DiffusionLossFn]:
        DIFFUSION_LOSS_REGISTRY[name] = cls()
        return cls

    return decorator


def get_diffusion_loss_fn(name: str) -> DiffusionLossFn:
    """Get a worker-side diffusion loss function by name."""
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
class FlowGRPOLoss(DiffusionLossFn):
    """Flow-GRPO clipped policy objective."""

    required_model_output_keys = ("log_probs",)
    required_data_keys = ("old_log_probs", "advantages")

    @classmethod
    def compute_loss(
        cls,
        *,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        config: DiffusionActorConfig,
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
        """
        assert config is not None, "config is required for FlowGRPOLoss!"
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
        pg_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

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

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        loss, metrics = self.compute_loss(
            old_log_prob=data["old_log_probs"],
            log_prob=model_output["log_probs"],
            advantages=data["advantages"],
            config=config,
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)


@register_diffusion_loss("grpo_guard")
class GRPOGuardLoss(DiffusionLossFn):
    """GRPO-Guard clipped policy objective with reverse-SDE mean drift."""

    required_model_output_keys = ("log_probs", "prev_sample_mean", "std_dev_t", "sqrt_dt")
    required_data_keys = ("old_log_probs", "advantages", "old_prev_sample_mean")

    @classmethod
    def compute_loss(
        cls,
        *,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        config: DiffusionActorConfig,
        old_prev_sample_mean: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        std_dev_t: torch.Tensor,
        sqrt_dt: torch.Tensor,
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
        pg_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (sqrt_dt_mean**2)

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

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        loss, metrics = self.compute_loss(
            old_log_prob=data["old_log_probs"],
            log_prob=model_output["log_probs"],
            advantages=data["advantages"],
            old_prev_sample_mean=data["old_prev_sample_mean"],
            prev_sample_mean=model_output["prev_sample_mean"],
            std_dev_t=model_output["std_dev_t"],
            sqrt_dt=model_output["sqrt_dt"],
            config=config,
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)


def diffusion_nft_old_policy_decay(step: int, decay_type: int) -> float:
    """Reference DiffusionNFT old-policy LoRA adapter decay schedules."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unsupported DiffusionNFT old_policy_decay_type: {decay_type}")
    return 0.0 if step < flat else min((step - flat) * uprate, uphold)


def compute_diffusion_nft_group_advantages(
    *,
    rewards: torch.Tensor,
    uid: np.ndarray,
    norm_by_std: bool,
    global_std: bool,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute group-relative scalar advantages for DiffusionNFT samples."""
    rewards = rewards.detach().float()
    advantages = rewards.clone()
    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    batch_std = torch.std(rewards) if global_std else None

    for idx, group_id in enumerate(uid):
        id2score[group_id].append(rewards[idx])

    id2mean = {}
    id2std = {}
    for group_id, group_scores in id2score.items():
        scores_tensor = torch.stack(group_scores)
        id2mean[group_id] = scores_tensor.mean()
        if global_std:
            id2std[group_id] = batch_std
        elif len(group_scores) > 1:
            id2std[group_id] = scores_tensor.std()
        else:
            id2std[group_id] = torch.tensor(1.0, device=rewards.device)

    for idx, group_id in enumerate(uid):
        advantages[idx] = rewards[idx] - id2mean[group_id]
        if norm_by_std:
            advantages[idx] = advantages[idx] / (id2std[group_id] + epsilon)
    return advantages


def diffusion_nft_advantage_to_reward_prob(
    advantages: torch.Tensor,
    *,
    adv_clip_max: float,
    adv_mode: str,
) -> torch.Tensor:
    """Map clipped DiffusionNFT advantages to implicit preference probabilities."""
    advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    if adv_mode == "positive_only":
        advantages = torch.clamp(advantages, 0, adv_clip_max)
    elif adv_mode == "negative_only":
        advantages = torch.clamp(advantages, -adv_clip_max, 0)
    elif adv_mode == "one_only":
        advantages = torch.where(advantages > 0, torch.ones_like(advantages), torch.zeros_like(advantages))
    elif adv_mode == "binary":
        advantages = torch.sign(advantages)

    reward_prob = (advantages / adv_clip_max) / 2.0 + 0.5
    return torch.clamp(reward_prob, 0, 1)


def select_diffusion_nft_train_timesteps(
    *,
    train_timesteps: torch.Tensor,
    timestep_fraction: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Select the forward-process timesteps used by DiffusionNFT actor updates."""
    if train_timesteps.ndim != 2:
        raise ValueError(f"DiffusionNFT `train_timesteps` must have shape [B, T], got {train_timesteps.shape}.")

    num_timesteps = train_timesteps.shape[1]
    num_train_timesteps = max(1, int(num_timesteps * timestep_fraction))
    generator = None
    if seed is not None:
        generator = torch.Generator(device=train_timesteps.device)
        generator.manual_seed(int(seed))

    permuted = []
    for row in train_timesteps:
        perm = torch.randperm(num_timesteps, device=train_timesteps.device, generator=generator)
        permuted.append(row[perm[:num_train_timesteps]])
    return torch.stack(permuted, dim=0).long()


def prepare_diffusion_nft_actor_batch(
    *,
    rollout_batch: dict[str, Any],
    rewards: torch.Tensor,
    config: Any,
    timestep_shuffle_seed: int | None = None,
) -> dict[str, Any]:
    """Prepare final-latent rollout data for DiffusionNFT direct-preference updates."""
    if "latents_clean" not in rollout_batch:
        raise ValueError("DiffusionNFT direct-preference training requires `latents_clean` from rollout.")
    if "train_timesteps" not in rollout_batch:
        raise ValueError("DiffusionNFT direct-preference training requires rollout `train_timesteps`.")
    if "uid" not in rollout_batch:
        raise ValueError("DiffusionNFT direct-preference training requires non-tensor `uid` groups.")

    nft_cfg = config.diffusion_nft
    advantages = compute_diffusion_nft_group_advantages(
        rewards=rewards,
        uid=rollout_batch["uid"],
        norm_by_std=config.norm_adv_by_std_in_grpo,
        global_std=config.global_std,
    )
    reward_prob = diffusion_nft_advantage_to_reward_prob(
        advantages,
        adv_clip_max=nft_cfg.adv_clip_max,
        adv_mode=nft_cfg.adv_mode,
    )
    train_timesteps = select_diffusion_nft_train_timesteps(
        train_timesteps=rollout_batch["train_timesteps"],
        timestep_fraction=nft_cfg.timestep_fraction,
        seed=timestep_shuffle_seed,
    )

    if reward_prob.ndim == 1 and train_timesteps.ndim == 2:
        reward_prob = reward_prob[:, None].expand(-1, train_timesteps.shape[1])

    actor_batch = dict(rollout_batch)
    actor_batch["train_timesteps"] = train_timesteps
    actor_batch["advantages"] = advantages[:, None].expand(-1, train_timesteps.shape[1])
    actor_batch["reward_prob"] = reward_prob
    actor_batch["returns"] = actor_batch["advantages"]
    actor_batch["sample_level_rewards"] = rewards[:, None].expand(-1, train_timesteps.shape[1])
    return actor_batch


@register_diffusion_loss("diffusion_nft")
class DiffusionNFTLoss(DiffusionLossFn):
    """DiffusionNFT forward-process direct-preference objective."""

    required_model_output_keys = (
        "forward_prediction",
        "old_prediction",
        "ref_forward_prediction",
        "x0",
        "xt",
        "t_expanded",
    )
    required_data_keys = ("reward_prob",)

    @classmethod
    def compute_loss(
        cls,
        *,
        forward_prediction: torch.Tensor,
        old_prediction: torch.Tensor,
        ref_forward_prediction: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t_expanded: torch.Tensor,
        reward_prob: torch.Tensor,
        config: DiffusionActorConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        loss_cfg = config.diffusion_loss.diffusion_nft
        beta = loss_cfg.mix_beta

        old_prediction = old_prediction.detach()
        ref_forward_prediction = ref_forward_prediction.detach()
        reward_weight = reward_prob
        if reward_weight.ndim > 1:
            reward_weight = reward_weight.flatten(1).mean(dim=1)
        reward_weight = reward_weight.to(device=x0.device, dtype=x0.dtype)

        reduce_dims = tuple(range(1, x0.ndim))
        positive_prediction = beta * forward_prediction + (1.0 - beta) * old_prediction
        implicit_negative_prediction = (1.0 + beta) * old_prediction - beta * forward_prediction

        x0_prediction = xt - t_expanded * positive_prediction
        negative_x0_prediction = xt - t_expanded * implicit_negative_prediction

        with torch.no_grad():
            positive_weight = (
                torch.abs(x0_prediction.double() - x0.double())
                .mean(dim=reduce_dims, keepdim=True)
                .clip(min=loss_cfg.adaptive_weight_min)
                .to(dtype=x0_prediction.dtype)
            )
            negative_weight = (
                torch.abs(negative_x0_prediction.double() - x0.double())
                .mean(dim=reduce_dims, keepdim=True)
                .clip(min=loss_cfg.adaptive_weight_min)
                .to(dtype=negative_x0_prediction.dtype)
            )

        positive_loss = ((x0_prediction - x0) ** 2 / positive_weight).mean(dim=reduce_dims)
        negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight).mean(dim=reduce_dims)
        policy_loss_per_sample = (reward_weight * positive_loss / beta) + (
            (1.0 - reward_weight) * negative_loss / beta
        )
        policy_loss = (policy_loss_per_sample * loss_cfg.adv_clip_max).mean()

        ref_kl_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=reduce_dims).mean()
        loss = policy_loss + loss_cfg.ref_kl_coef * ref_kl_loss

        with torch.no_grad():
            metrics = {
                "actor/policy_loss": policy_loss.detach().item(),
                "actor/positive_loss": positive_loss.mean().detach().item(),
                "actor/negative_loss": negative_loss.mean().detach().item(),
                "actor/ref_kl_loss": ref_kl_loss.detach().item(),
                "actor/old_deviate": ((forward_prediction - old_prediction) ** 2).mean().detach().item(),
                "actor/reward_prob_mean": reward_weight.mean().detach().item(),
                "actor/total_loss": loss.detach().item(),
            }
        return loss, metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        loss, metrics = self.compute_loss(
            forward_prediction=model_output["forward_prediction"],
            old_prediction=model_output["old_prediction"],
            ref_forward_prediction=model_output["ref_forward_prediction"],
            x0=model_output["x0"],
            xt=model_output["xt"],
            t_expanded=model_output["t_expanded"],
            reward_prob=data["reward_prob"],
            config=config,
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)


def compute_diffusion_loss_diffusion_nft(**kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """Backward-compatible pure loss entry used by existing DiffusionNFT tests."""
    return DiffusionNFTLoss.compute_loss(**kwargs)


@register_diffusion_loss("kl")
class KLLoss(DiffusionLossFn):
    """KL divergence between current and reference reverse-SDE means."""

    required_model_output_keys = ("prev_sample_mean", "std_dev_t")
    required_data_keys = ("ref_prev_sample_mean",)

    @classmethod
    def compute_loss(
        cls,
        *,
        prev_sample_mean: torch.Tensor,
        ref_prev_sample_mean: torch.Tensor,
        std_dev_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute KL divergence given previous sample mean and reference previous sample mean (for images or videos).

        Args:
            prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
            ref_prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
            std_dev_t: (torch.Tensor) shape is (bs, 1, 1)
        """
        kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1, 2), keepdim=True) / (2 * std_dev_t**2)
        metrics = {"actor/kl_loss": kl_loss.mean().detach().item()}
        return kl_loss.mean(), metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        kl_loss, metrics = self.compute_loss(
            prev_sample_mean=model_output["prev_sample_mean"],
            ref_prev_sample_mean=data["ref_prev_sample_mean"],
            std_dev_t=model_output["std_dev_t"],
        )
        return DiffusionLossResult(loss=kl_loss, metrics=metrics)
