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

import json
import logging
import os
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_art import art_guidance, art_loss, art_pairwise_labels, group_index_from_uid
from verl_omni.workers.config import DiffusionActorConfig

logger = logging.getLogger(__name__)


@dataclass
class DiffusionLossResult:
    """Output from a batch-aware diffusion loss function."""

    loss: torch.Tensor
    metrics: dict[str, Any]
    add_loss_metric: bool = False


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
    # Whether the loss trains against the LoRA ``old`` adapter that the rollout samples with.
    uses_old_adapter: bool = False
    # Whether the loss reduces over whole rollout groups inside a micro batch, so the
    # trainer must not shuffle samples across micro batches.
    requires_whole_groups: bool = False

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

    @staticmethod
    def prepare_actor_batch(
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
    ) -> DataProto:
        """Prepare rollout outputs for actor update when the trainer has not already done so.

        Reverse-process policy-gradient losses such as FlowGRPO can keep the batch
        unchanged because their trainer path has already added ``old_log_probs`` and
        ``advantages``. Offline DPO can also keep
        the batch unchanged because offline preference data plus reference
        predictions provide the loss inputs directly. Forward-process online
        losses such as DiffusionNFT, and online DPO override this hook to turn final-latent
        rollouts and rewards into loss-specific actor tensors.
        """
        return batch

    @staticmethod
    def validate_trainer_config(config: Any) -> None:
        """Fail at trainer start on settings this loss cannot honour.

        Called by the direct-preference trainers at start. Only DGPO consumes ``algorithm.train_timestep_range``
        and ``algorithm.train_timestep_count``, so the other direct-preference losses reject them instead of
        silently ignoring them.
        """
        algorithm_cfg = config.algorithm
        for key in ("train_timestep_range", "train_timestep_count"):
            if getattr(algorithm_cfg, key, None) is not None:
                raise ValueError(f"algorithm.{key} is only used by the dgpo loss.")


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
    DANCE_GRPO = "dance_grpo"


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
@register_diffusion_adv_est(DiffusionAdvantageEstimator.DANCE_GRPO)
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
@register_diffusion_loss("dance_grpo")
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
            rollout_is_weights=data.get("rollout_is_weights", None),
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)


@register_diffusion_loss("flow_dppo")
class FlowDPPOLoss(DiffusionLossFn):
    """Flow-DPPO policy objective with an exact divergence trust-region mask."""

    required_model_output_keys = ("log_probs", "prev_sample_mean", "std_dev_t", "sqrt_dt")
    required_data_keys = ("old_log_probs", "advantages", "old_prev_sample_mean")

    @staticmethod
    def _broadcast_sqrt_dt(sqrt_dt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if sqrt_dt.ndim == 0:
            return sqrt_dt.reshape(1, *([1] * (target.ndim - 1)))
        return sqrt_dt.reshape(sqrt_dt.shape[0], *([1] * (target.ndim - 1)))

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
        rollout_is_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Flow-DPPO's asymmetric KL-mask policy objective.

        Flow-DPPO replaces PPO ratio clipping with an exact Gaussian KL
        trust-region mask. Updates are masked only when they both exceed the
        divergence threshold and move farther from the old policy.
        """
        assert config is not None, "config is required for FlowDPPOLoss!"
        loss_cfg = config.diffusion_loss
        advantages = advantages.detach()

        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        unclipped_loss = -advantages * ratio

        mean_diff_sq = (prev_sample_mean - old_prev_sample_mean).pow(2)
        if getattr(loss_cfg, "add_kl_coefficient", True):
            sigma_t = std_dev_t * cls._broadcast_sqrt_dt(sqrt_dt, std_dev_t)
            kl_per_elem = mean_diff_sq / (2 * sigma_t.pow(2))
        else:
            kl_per_elem = mean_diff_sq / 2
        if kl_per_elem.ndim > 1:
            kl_per_sample = kl_per_elem.mean(dim=tuple(range(1, kl_per_elem.ndim)))
        else:
            kl_per_sample = kl_per_elem

        kl_mask_threshold = loss_cfg.kl_mask_threshold
        high_kl_mask = kl_per_sample >= kl_mask_threshold
        pos_rm_mask = high_kl_mask & (ratio > 1.0) & (advantages > 0)
        neg_rm_mask = high_kl_mask & (ratio < 1.0) & (advantages < 0)
        rm_mask = pos_rm_mask | neg_rm_mask
        keep_mask = (~rm_mask).detach()

        zero = torch.zeros((), dtype=unclipped_loss.dtype, device=unclipped_loss.device)
        per_elem_loss = torch.where(keep_mask, unclipped_loss, zero)
        if rollout_is_weights is not None:
            per_elem_loss = per_elem_loss * rollout_is_weights.detach()
        pg_loss = torch.mean(per_elem_loss)

        with torch.no_grad():
            ratio_std = ratio.std(unbiased=False)
            pg_metrics = {
                "actor/ppo_kl": torch.mean(-log_ratio).detach().item(),
                "actor/approx_kl": (0.5 * log_ratio.pow(2)).mean().detach().item(),
                "actor/ratio_mean": ratio.mean().detach().item(),
                "actor/ratio_std": ratio_std.detach().item(),
                "actor/ratio_min": ratio.min().detach().item(),
                "actor/ratio_max": ratio.max().detach().item(),
                "actor/kl_new_old_mean": kl_per_sample.mean().detach().item(),
                "actor/kl_new_old_max": kl_per_sample.max().detach().item(),
                "actor/kl_mask_fraction": high_kl_mask.float().mean().detach().item(),
                "actor/pos_rm_fraction": pos_rm_mask.float().mean().detach().item(),
                "actor/neg_rm_fraction": neg_rm_mask.float().mean().detach().item(),
                "actor/masked_fraction": rm_mask.float().mean().detach().item(),
                "actor/unmasked_fraction": keep_mask.float().mean().detach().item(),
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
            rollout_is_weights=data.get("rollout_is_weights", None),
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
            rollout_is_weights (Optional[torch.Tensor]):
                Optional Rollout Correction multiplier (same shape as ``log_prob``) combining
                IS weights and RS rejection (rejected samples have weight 0). When provided,
                the per-element policy loss is multiplied by these (detached) weights before
                the mean reduction.
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
            rollout_is_weights=data.get("rollout_is_weights", None),
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)


@register_diffusion_loss("dpo")
class DPOLoss(DiffusionLossFn):
    """DPO loss with win/reward optimization."""

    required_model_output_keys = ("noise", "latent", "noise_pred")
    required_data_keys = ("ref_noise_pred", "sample_level_rewards")

    @staticmethod
    def build_online_dpo_pair_indices(
        *,
        uids: np.ndarray,
        scores: torch.Tensor,
    ) -> list[int]:
        """Build one adjacent top-vs-bottom chosen/rejected pair per prompt."""

        flat_scores = scores.squeeze(-1) if scores.ndim > 1 else scores
        score_values = flat_scores.detach().cpu().float().tolist()
        uid_values = np.asarray(uids, dtype=object).reshape(-1)
        if len(uid_values) != len(score_values):
            raise ValueError(
                f"Online DPO pairing expects one uid per score, got {len(uid_values)} vs {len(score_values)}."
            )

        uid_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uid_values):
            uid_to_indices[uid].append(idx)

        selected_indices: list[int] = []
        for group_indices in uid_to_indices.values():
            if len(group_indices) < 2:
                continue

            sorted_indices = sorted(group_indices, key=lambda idx: score_values[idx], reverse=True)
            selected_indices.extend([sorted_indices[0], sorted_indices[-1]])

        return selected_indices

    @staticmethod
    def _select_dataproto_indices(batch: DataProto, selected_indices: list[int]) -> DataProto:
        selected_non_tensor = {}
        for key, value in batch.non_tensor_batch.items():
            try:
                if len(value) == len(batch):
                    values = np.empty(len(value), dtype=object)
                    values[:] = list(value)
                    selected_non_tensor[key] = values[selected_indices]
                else:
                    selected_non_tensor[key] = value
            except TypeError:
                selected_non_tensor[key] = value

        selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
        return DataProto(
            batch=batch.batch[selected_tensor],
            non_tensor_batch=selected_non_tensor,
            meta_info=batch.meta_info,
        )

    @staticmethod
    def prepare_actor_batch(
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
    ) -> DataProto:
        """Offline: no-op. Online: pick top/bottom reward pair per prompt uid."""
        rewards = reward_tensor.squeeze(-1).float() if reward_tensor.ndim > 1 else reward_tensor.float()
        if config.algorithm.sample_source == "offline":
            return batch

        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Online DPO pairing requires `uid` in non_tensor_batch.")

        selected_indices = DPOLoss.build_online_dpo_pair_indices(
            uids=batch.non_tensor_batch["uid"],
            scores=rewards,
        )
        if not selected_indices:
            raise RuntimeError("Online DPO could not build preference pairs; increase actor_rollout_ref.rollout.n.")

        batch = DPOLoss._select_dataproto_indices(batch, selected_indices)
        batch.meta_info["prepare_actor_batch_selected_indices"] = selected_indices
        return batch

    @classmethod
    def _dpo_adjacent_pairs_share_prompt_uid(cls, index: Any, n: int) -> bool:
        """Return True if each adjacent (chosen, rejected) pair shares the same prompt uid.

        Avoids slicing or ``np.asarray`` on TensorDict / TensorClass handles (batch_dims==0), which
        cannot be indexed like a plain tensor.
        """
        if isinstance(index, torch.Tensor):
            flat = index.reshape(-1)[:n]
            return bool(torch.all(flat[0::2] == flat[1::2]).item())
        if isinstance(index, np.ndarray):
            flat = np.asarray(index).reshape(-1)[:n]
            return bool(np.all(flat[0::2] == flat[1::2]))
        if isinstance(index, list | tuple):
            flat = np.asarray(index, dtype=object).reshape(-1)[:n]
            return bool(np.all(flat[0::2] == flat[1::2]))
        tolist = getattr(index, "tolist", None)
        if callable(tolist):
            try:
                raw = tolist()
            except (RuntimeError, TypeError, ValueError):
                raw = None
            if raw is not None and not isinstance(raw, str | bytes | bytearray):
                flat = np.asarray(raw, dtype=object).reshape(-1)[:n]
                return bool(np.all(flat[0::2] == flat[1::2]))
        raise TypeError(
            f"DPO `index` (prompt uid) has unsupported type {type(index)}. "
            "Use a torch.Tensor, numpy.ndarray, list, or tuple (or pass uids via non_tensor batch)."
        )

    @classmethod
    def compute_loss(
        cls,
        noise: torch.Tensor,
        latent: torch.Tensor,
        model_noise_pred: torch.Tensor,
        ref_noise_pred: torch.Tensor,
        sample_level_rewards: torch.Tensor,
        config: Optional[DictConfig | DiffusionActorConfig] = None,
        *,
        index: Optional[np.ndarray | torch.Tensor | list[Any]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute DPO loss from adjacent ``chosen, rejected`` sample pairs.
        Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/dpo/loss.py
        """
        assert config is not None
        assert isinstance(config, DiffusionActorConfig)

        scores = sample_level_rewards.squeeze(-1) if sample_level_rewards.ndim > 1 else sample_level_rewards
        if scores.shape[0] < 2 or scores.shape[0] % 2 != 0:
            raise ValueError("DPO loss expects an even batch of adjacent chosen/rejected pairs.")
        if index is not None:
            n = int(scores.shape[0])
            if not cls._dpo_adjacent_pairs_share_prompt_uid(index, n):
                raise ValueError("DPO loss expects each adjacent chosen/rejected pair to share the same prompt uid.")

        chosen_scores = scores[0::2]
        rejected_scores = scores[1::2]
        if torch.any(chosen_scores < rejected_scores).item():
            raise ValueError("DPO loss expects each chosen sample reward to be >= its rejected pair reward.")

        beta = config.diffusion_loss.dpo_beta
        target = noise.float() - latent.float()
        model_err = ((model_noise_pred.float() - target) ** 2).flatten(1).mean(dim=1)
        ref_err = ((ref_noise_pred.float() - target) ** 2).flatten(1).mean(dim=1)

        model_w_err = model_err[0::2]
        model_l_err = model_err[1::2]
        ref_w_err = ref_err[0::2]
        ref_l_err = ref_err[1::2]
        w_diff = model_w_err - ref_w_err
        l_diff = model_l_err - ref_l_err
        inside_term = -0.5 * beta * (w_diff - l_diff)
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        dpo_loss = -F.logsigmoid(inside_term).mean()

        with torch.no_grad():
            implicit_reward_chosen = -0.5 * beta * w_diff
            implicit_reward_rejected = -0.5 * beta * l_diff
            reward_margins = implicit_reward_chosen - implicit_reward_rejected
            metrics = {
                "actor/dpo_loss": dpo_loss.detach().item(),
                "actor/implicit_acc": implicit_acc.detach().item(),
                "rewards/chosen": implicit_reward_chosen.mean().detach().item(),
                "rewards/rejected": implicit_reward_rejected.mean().detach().item(),
                "rewards/margins": reward_margins.mean().detach().item(),
            }
        return dpo_loss, metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        loss, metrics = self.compute_loss(
            noise=model_output["noise"],
            latent=model_output["latent"],
            model_noise_pred=model_output["noise_pred"],
            ref_noise_pred=data["ref_noise_pred"],
            sample_level_rewards=data["sample_level_rewards"],
            config=config,
            index=tu.get_non_tensor_data(data, "uid", default=None),
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)


@register_diffusion_loss("diffusion_nft")
class DiffusionNFTLoss(DiffusionLossFn):
    """DiffusionNFT forward-process direct-preference objective."""

    uses_old_adapter = True
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
        """Compute the DiffusionNFT policy loss and auxiliary metrics.

        `dL/dv_theta` decomposes into exactly two parts, because `old` is detached and only
        `v_theta` moves both x0 estimates (with `dx0_pos/dv_theta = -t*beta` and
        `dx0_neg/dv_theta = +t*beta`, which cancels the `1/beta` in the loss):

            dL/dv_theta  ~  A*2/w * ( [ beta * t * (v_theta - v_old) ]  -  [ 2*(r-0.5) * (x0_old - x0) ] )
                                      \\____________ A * beta ____________/     \\_______ adv, no A _______/

        Only the first term grows with `A * beta`; the second is the whole learning signal, and
        `A` cancels out of it because `r - 0.5 = adv / (2*A)`. `log10_signal_ratio` reports the
        two scales separately, since the `1/beta` scaling of the loss values hides that balance.
        """
        loss_cfg = config.diffusion_loss
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
        policy_loss_per_sample = (reward_weight * positive_loss / beta) + ((1.0 - reward_weight) * negative_loss / beta)
        policy_loss = (policy_loss_per_sample * loss_cfg.adv_clip_max).mean()

        ref_kl_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=reduce_dims).mean()
        loss = policy_loss + loss_cfg.ref_kl_coef * ref_kl_loss

        with torch.no_grad():
            # The two gradient-scale terms of the decomposition in the docstring: the reward-free
            # contraction onto `v_old`, and the reward term, from which `adv_clip_max` cancels.
            x0_old = xt - t_expanded * old_prediction
            reward_weight_col = reward_weight.view(-1, *([1] * (x0.ndim - 1)))
            contraction = loss_cfg.adv_clip_max * beta * t_expanded * (forward_prediction - old_prediction)
            reward_term = loss_cfg.adv_clip_max * 2.0 * (reward_weight_col - 0.5) * (x0_old - x0)
            contraction_scale = contraction.abs().mean()
            reward_term_scale = reward_term.abs().mean()
            metrics = {
                "actor/policy_loss": policy_loss.detach().item(),
                "actor/positive_loss": positive_loss.mean().detach().item(),
                "actor/negative_loss": negative_loss.mean().detach().item(),
                "actor/contraction_scale": contraction_scale.detach().item(),
                "actor/reward_term_scale": reward_term_scale.detach().item(),
                "actor/log10_signal_ratio": (
                    (torch.log10(reward_term_scale.clamp(min=1e-30)) - torch.log10(contraction_scale.clamp(min=1e-30)))
                    .detach()
                    .item()
                ),
                "actor/ref_kl_loss": ref_kl_loss.detach().item(),
                "actor/ref_kl_contribution": (loss_cfg.ref_kl_coef * ref_kl_loss).detach().item(),
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

    # ------------------------------------------------------------------
    # Trainer-side helpers (batch preparation)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_group_advantages(
        rewards: torch.Tensor,
        uid: np.ndarray,
        norm_by_std: bool,
        global_std: bool,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """Group-normalize raw rewards for DiffusionNFT optimality probability.
        This is not the same as the policy gradient advantages for loss computation.

        Per prompt ``c`` (``uid``), DiffusionNFT Sec. 3.3 / Algo. 1 steps 4--5:

            r_norm = r^raw(x_0, c) - E_pi_old[r^raw | c]
            r_norm /= Z_c   (if ``norm_by_std``; ``Z_c`` = per-group or global reward std)

        Optimality reward: r = 1/2 + 1/2 * clip(r_norm / Z_c, -1, 1)  (clip/map in
        ``_advantage_to_reward_prob``). ``reward_prob`` weights the forward-process loss.
        """
        rewards = rewards.detach().float()
        advantages = rewards.clone()
        id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
        batch_std = torch.std(rewards) if global_std else None

        for idx, group_id in enumerate(uid):
            id2score[group_id].append(rewards[idx])

        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
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

    @staticmethod
    def _advantage_to_reward_prob(
        advantages: torch.Tensor,
        adv_clip_max: float,
        adv_mode: str,
    ) -> torch.Tensor:
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

    @staticmethod
    def _select_train_timesteps(
        train_timesteps: torch.Tensor,
        timestep_fraction: float,
        seed: int | None = None,
    ) -> torch.Tensor:
        if train_timesteps.ndim != 2:
            raise ValueError(f"`train_timesteps` must have shape [B, T], got {train_timesteps.shape}.")
        num_timesteps = train_timesteps.shape[1]
        num_train = max(1, int(num_timesteps * timestep_fraction))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=train_timesteps.device)
            generator.manual_seed(int(seed))
        permuted = []
        for row in train_timesteps:
            perm = torch.randperm(num_timesteps, device=train_timesteps.device, generator=generator)
            permuted.append(row[perm[:num_train]])
        return torch.stack(permuted, dim=0).long()

    @staticmethod
    def prepare_actor_batch(
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
    ) -> DataProto:
        """Prepare final-latent rollout data for DiffusionNFT actor updates."""

        algorithm_cfg = config.algorithm
        actor_cfg = config.actor_rollout_ref.actor
        adv_clip_max = actor_cfg.diffusion_loss.adv_clip_max
        timestep_shuffle_seed = actor_cfg.data_loader_seed

        rollout_batch = {key: batch.batch[key] for key in batch.batch.keys()}
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("DiffusionNFT actor batch requires `uid` in non_tensor_batch.")
        rollout_batch["uid"] = batch.non_tensor_batch["uid"]

        for key in ("latents_clean", "train_timesteps"):
            if key not in rollout_batch:
                raise ValueError(f"DiffusionNFT actor batch requires `{key}` from rollout.")

        advantages = DiffusionNFTLoss._compute_group_advantages(
            rewards=reward_tensor,
            uid=rollout_batch["uid"],
            norm_by_std=algorithm_cfg.norm_adv_by_std_in_grpo,
            global_std=algorithm_cfg.global_std,
        )
        reward_prob = DiffusionNFTLoss._advantage_to_reward_prob(
            advantages, adv_clip_max=adv_clip_max, adv_mode=algorithm_cfg.adv_mode
        )
        train_timesteps = DiffusionNFTLoss._select_train_timesteps(
            rollout_batch["train_timesteps"],
            timestep_fraction=algorithm_cfg.timestep_fraction,
            seed=timestep_shuffle_seed,
        )
        if reward_prob.ndim == 1 and train_timesteps.ndim == 2:
            reward_prob = reward_prob[:, None].expand(-1, train_timesteps.shape[1])

        batch.batch["train_timesteps"] = train_timesteps
        batch.batch["advantages"] = advantages[:, None].expand(-1, train_timesteps.shape[1])
        batch.batch["reward_prob"] = reward_prob
        batch.batch["returns"] = batch.batch["advantages"]
        batch.batch["sample_level_rewards"] = reward_tensor[:, None].expand(-1, train_timesteps.shape[1])
        return batch


@register_diffusion_loss("omni_nft")
class OmniNFTLoss(DiffusionNFTLoss):
    """Two-branch DiffusionNFT objective for joint audio-video generation."""

    @classmethod
    def validate_trainer_config(cls, config: Any) -> None:
        """Reject single-latent timestep staging before starting OmniNFT workers."""
        super().validate_trainer_config(config)
        if config.actor_rollout_ref.actor.get("enable_timestep_staging", False):
            raise ValueError(
                "OmniNFT does not support timestep staging for joint video/audio inputs; "
                "set actor_rollout_ref.actor.enable_timestep_staging=False."
            )

    required_model_output_keys = (
        "video_forward_prediction",
        "video_old_prediction",
        "video_ref_forward_prediction",
        "video_x0",
        "video_xt",
        "video_t_expanded",
        "audio_forward_prediction",
        "audio_old_prediction",
        "audio_ref_forward_prediction",
        "audio_x0",
        "audio_xt",
        "audio_t_expanded",
    )
    required_data_keys = ("reward_prob",)

    @staticmethod
    def _compute_modality_loss(
        *,
        forward_prediction: torch.Tensor,
        old_prediction: torch.Tensor,
        ref_forward_prediction: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t_expanded: torch.Tensor,
        reward_prob: torch.Tensor,
        config: DiffusionActorConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute NFT policy loss and reference velocity MSE for one modality.

        Adaptive weights use detached FP64 mean absolute errors. Only current
        predictions carry gradients; ``ref_kl_loss`` is the legacy MSE metric name.
        """
        loss_cfg = config.diffusion_loss
        beta = loss_cfg.mix_beta
        old_prediction = old_prediction.detach()
        ref_forward_prediction = ref_forward_prediction.detach()

        reward_weight = reward_prob
        if reward_weight.ndim > 1:
            reward_weight = reward_weight.flatten(1).mean(dim=1)
        reward_weight = reward_weight.to(device=x0.device, dtype=x0.dtype)

        reduce_dims = tuple(range(1, x0.ndim))
        positive_prediction = beta * forward_prediction + (1.0 - beta) * old_prediction
        negative_prediction = (1.0 + beta) * old_prediction - beta * forward_prediction
        positive_x0 = xt - t_expanded * positive_prediction
        negative_x0 = xt - t_expanded * negative_prediction

        with torch.no_grad():
            positive_weight = (
                torch.abs(positive_x0.double() - x0.double())
                .mean(dim=reduce_dims, keepdim=True)
                .clamp_min(loss_cfg.adaptive_weight_min)
                .to(dtype=positive_x0.dtype)
            )
            negative_weight = (
                torch.abs(negative_x0.double() - x0.double())
                .mean(dim=reduce_dims, keepdim=True)
                .clamp_min(loss_cfg.adaptive_weight_min)
                .to(dtype=negative_x0.dtype)
            )

        positive_loss = ((positive_x0 - x0).square() / positive_weight).mean(dim=reduce_dims)
        negative_loss = ((negative_x0 - x0).square() / negative_weight).mean(dim=reduce_dims)
        policy_per_sample = reward_weight * positive_loss / beta + (1.0 - reward_weight) * negative_loss / beta
        policy_loss = (policy_per_sample * loss_cfg.adv_clip_max).mean()
        ref_loss = (forward_prediction - ref_forward_prediction).square().mean()
        details = {
            "positive_loss": positive_loss.mean(),
            "negative_loss": negative_loss.mean(),
            "ref_kl_loss": ref_loss,
            "old_deviate": (forward_prediction - old_prediction).square().mean(),
            "reward_prob_mean": reward_weight.mean(),
        }
        return policy_loss, ref_loss, details

    @classmethod
    def compute_loss(
        cls,
        *,
        video_forward_prediction: torch.Tensor,
        video_old_prediction: torch.Tensor,
        video_ref_forward_prediction: torch.Tensor,
        video_x0: torch.Tensor,
        video_xt: torch.Tensor,
        video_t_expanded: torch.Tensor,
        video_reward_prob: torch.Tensor,
        audio_forward_prediction: torch.Tensor,
        audio_old_prediction: torch.Tensor,
        audio_ref_forward_prediction: torch.Tensor,
        audio_x0: torch.Tensor,
        audio_xt: torch.Tensor,
        audio_t_expanded: torch.Tensor,
        audio_reward_prob: torch.Tensor,
        config: DiffusionActorConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Combine modality policy losses and reference MSEs with separate coefficients.

        Modality loss weights are independent of reward-routing weights.
        """
        video_policy, video_ref, video_details = cls._compute_modality_loss(
            forward_prediction=video_forward_prediction,
            old_prediction=video_old_prediction,
            ref_forward_prediction=video_ref_forward_prediction,
            x0=video_x0,
            xt=video_xt,
            t_expanded=video_t_expanded,
            reward_prob=video_reward_prob,
            config=config,
        )
        audio_policy, audio_ref, audio_details = cls._compute_modality_loss(
            forward_prediction=audio_forward_prediction,
            old_prediction=audio_old_prediction,
            ref_forward_prediction=audio_ref_forward_prediction,
            x0=audio_x0,
            xt=audio_xt,
            t_expanded=audio_t_expanded,
            reward_prob=audio_reward_prob,
            config=config,
        )
        loss_cfg = config.diffusion_loss
        modality_weight_sum = max(loss_cfg.video_weight + loss_cfg.audio_weight, 1e-8)
        total_loss = (
            loss_cfg.video_weight * (video_policy + loss_cfg.video_ref_kl_coef * video_ref)
            + loss_cfg.audio_weight * (audio_policy + loss_cfg.audio_ref_kl_coef * audio_ref)
        ) / modality_weight_sum

        metrics = {
            "actor/video/policy_loss": video_policy.detach().item(),
            "actor/audio/policy_loss": audio_policy.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
        }
        for modality, details in (("video", video_details), ("audio", audio_details)):
            for name, value in details.items():
                metrics[f"actor/{modality}/{name}"] = value.detach().item()
        return total_loss, metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        """Validate single-step ``reward_prob[B, 2]`` and dispatch video/audio losses."""
        self.validate_inputs(loss_name="omni_nft", model_output=model_output, data=data)
        reward_prob = data["reward_prob"]
        if reward_prob.ndim != 2 or reward_prob.shape[1] != 2:
            raise ValueError(
                "OmniNFT reward_prob must have shape [B, 2] with columns ordered as video, audio; "
                f"got {tuple(reward_prob.shape)}."
            )
        loss, metrics = self.compute_loss(
            **{key: model_output[key] for key in self.required_model_output_keys},
            video_reward_prob=reward_prob[:, 0],
            audio_reward_prob=reward_prob[:, 1],
            config=config,
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)

    @staticmethod
    def _compute_component_advantages(
        scores: torch.Tensor,
        uid: Sequence[Any],
        *,
        norm_by_std: bool,
        global_std: bool,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """Center each reward component by prompt group and optionally normalize it.

        Use population standard deviation, either per group or across the full
        batch. Normalize components before routing them to video/audio.
        """
        if scores.ndim != 2:
            raise ValueError(f"OmniNFT reward scores must have shape [B, K], got {tuple(scores.shape)}.")
        if scores.shape[1] == 0:
            raise ValueError("OmniNFT reward scores must contain at least one component.")
        if len(uid) != scores.shape[0]:
            raise ValueError(f"OmniNFT uid count {len(uid)} does not match reward batch size {scores.shape[0]}.")
        if epsilon <= 0:
            raise ValueError(f"OmniNFT advantage epsilon must be positive, got {epsilon}.")

        scores = scores.detach().float()
        groups: dict[Any, list[int]] = defaultdict(list)
        for index, group_id in enumerate(uid):
            groups[group_id].append(index)

        advantages = torch.empty_like(scores)
        batch_std = scores.std(dim=0, correction=0) if global_std and norm_by_std else None
        for indices in groups.values():
            index_tensor = torch.tensor(indices, device=scores.device)
            group_scores = scores.index_select(0, index_tensor)
            centered = group_scores - group_scores.mean(dim=0, keepdim=True)
            if norm_by_std:
                std = batch_std if batch_std is not None else group_scores.std(dim=0, correction=0)
                centered = centered / (std.unsqueeze(0) + epsilon)
            advantages.index_copy_(0, index_tensor, centered)
        return advantages

    @staticmethod
    def _build_reward_routing_matrix(
        *,
        reward_names: Sequence[str],
        reward_functions: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build routing weights in reward_names order with video/audio columns.

        Return ``[K, 2]`` on the requested device/dtype. Reward names must be
        unique and exactly match configuration keys; each row requires
        nonnegative video/audio weights with at least one nonzero entry.
        Weights are not normalized.
        """
        reward_names = list(reward_names)
        if not reward_names or len(reward_names) != len(set(reward_names)):
            raise ValueError("OmniNFT reward_names must be non-empty and unique.")
        if set(reward_functions) != set(reward_names):
            raise ValueError("OmniNFT reward_names must match reward.reward_functions keys.")

        rows: list[list[float]] = []
        for name in reward_names:
            entry = reward_functions[name]
            routing_weights = entry.get("routing_weights")
            if routing_weights is None:
                raise ValueError(f"Reward '{name}' must define routing_weights.video and routing_weights.audio.")
            row = []
            for modality in ("video", "audio"):
                value = routing_weights.get(modality)
                if value is None:
                    raise ValueError(f"Reward '{name}' must define routing_weights.{modality}.")
                if value < 0:
                    raise ValueError(f"Reward '{name}' routing weight for {modality} must be non-negative.")
                row.append(float(value))
            if not any(row):
                raise ValueError(f"Reward '{name}' must route to at least one modality.")
            rows.append(row)
        return torch.tensor(rows, device=device, dtype=dtype)

    @classmethod
    def prepare_actor_batch(cls, batch: DataProto, reward_tensor: torch.Tensor, config: Any) -> DataProto:
        """Normalize reward components, route them to modalities, and select timesteps.

        Store per-step video/audio reward probabilities for the actor, plus
        reward summaries, advantages, and returns for existing metric consumers.
        """
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("OmniNFT actor batch requires `uid` in non_tensor_batch.")
        for key in ("video_latents_clean", "audio_latents_clean", "train_timesteps"):
            if key not in batch.batch:
                raise ValueError(f"OmniNFT actor batch requires `{key}` from rollout.")
        if "rm_scores" not in batch.batch:
            raise ValueError("OmniNFT actor batch requires `rm_scores` from the batch Reward Manager.")

        scores = batch.batch["rm_scores"].detach().float()
        if reward_tensor.ndim == 1 and scores.ndim == 2 and scores.shape[1] == 1:
            reward_tensor = reward_tensor.unsqueeze(-1)
        if reward_tensor.shape != scores.shape:
            raise ValueError(
                f"OmniNFT extracted reward shape {tuple(reward_tensor.shape)} does not match "
                f"rm_scores {tuple(scores.shape)}."
            )
        valid_mask = batch.batch.get("reward_valid_mask")
        if valid_mask is not None and (valid_mask.shape != scores.shape or not valid_mask.all()):
            raise ValueError("OmniNFT requires a fully valid reward_valid_mask matching rm_scores.")

        algorithm_cfg = config.algorithm
        actor_cfg = config.actor_rollout_ref.actor
        reward_cfg = config.reward
        reward_names = batch.meta_info.get("reward_names")
        if reward_names is None:
            raise ValueError("OmniNFT actor batch requires reward_names in meta_info.")
        reward_names = list(reward_names)

        reward_advantages = cls._compute_component_advantages(
            scores=scores,
            uid=batch.non_tensor_batch["uid"],
            norm_by_std=algorithm_cfg.norm_adv_by_std_in_grpo,
            global_std=algorithm_cfg.global_std,
        )
        routing_matrix = cls._build_reward_routing_matrix(
            reward_names=reward_names,
            reward_functions=reward_cfg.reward_functions,
            device=reward_advantages.device,
            dtype=reward_advantages.dtype,
        )
        if routing_matrix.shape != (reward_advantages.shape[1], 2):
            raise ValueError(
                f"OmniNFT routing matrix must have shape {(reward_advantages.shape[1], 2)}, "
                f"got {tuple(routing_matrix.shape)}."
            )
        modality_advantages = reward_advantages @ routing_matrix
        modality_reward_probs = cls._advantage_to_reward_prob(
            modality_advantages,
            adv_clip_max=actor_cfg.diffusion_loss.adv_clip_max,
            adv_mode=algorithm_cfg.adv_mode,
        )
        train_timesteps = DiffusionNFTLoss._select_train_timesteps(
            batch.batch["train_timesteps"],
            timestep_fraction=algorithm_cfg.timestep_fraction,
            seed=actor_cfg.data_loader_seed,
        )
        num_steps = train_timesteps.shape[1]
        timestep_modality_reward_probs = modality_reward_probs[:, None, :].expand(-1, num_steps, -1)
        summary_advantages = modality_advantages.mean(dim=1, keepdim=True).expand(-1, num_steps)

        batch.batch["train_timesteps"] = train_timesteps
        batch.batch["reward_prob"] = timestep_modality_reward_probs
        batch.batch["advantages"] = summary_advantages
        batch.batch["returns"] = summary_advantages
        batch.batch["sample_level_rewards"] = scores
        return batch


@register_diffusion_loss("dgpo")
class DGPOLoss(DiffusionLossFn):
    """DGPO group-level direct preference objective (Luo et al., arXiv:2510.08425).

    Every rollout group (one prompt, ``n`` images) shares its forward noise and timestep.
    Per sample ``i`` in group ``g`` with advantage ``A_i`` and flow-matching error
    ``dsm_i = ||v - v_theta(x_t)||^2`` (``ref_dsm_i`` under the reference model)::

        s_g  = sum_{i in g} A_i * beta * (dsm_i - ref_dsm_i) / |g|      (detached)
        loss = mean_i( sigmoid(s_g) * A_i * dsm_i )

    ``dsm_i`` is detached where the old-policy ratio ``exp(old_dsm_i - dsm_i)`` leaves
    ``[1 - clip_range, 1 + clip_range]`` in the direction of ``A_i``. The group sum is
    computed locally, so each micro batch must hold whole groups; the loss raises otherwise.
    """

    uses_old_adapter = True
    requires_whole_groups = True
    required_model_output_keys = (
        "forward_prediction",
        "old_prediction",
        "ref_forward_prediction",
        "x0",
        "xt",
        "t_expanded",
    )
    required_data_keys = ("advantages", "group_index", "group_size")

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
        advantages: torch.Tensor,
        group_index: torch.Tensor,
        group_size: torch.Tensor,
        config: DiffusionActorConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the DGPO loss for one micro batch of whole groups at one shared timestep."""
        loss_cfg = config.diffusion_loss
        reduce_dims = tuple(range(1, x0.ndim))
        # xt = (1 - t) * x0 + t * noise, so the flow-matching target noise - x0 is (xt - x0) / t.
        target = ((xt.float() - x0.float()) / t_expanded.float()).detach()
        advantages = advantages.to(device=x0.device, dtype=torch.float32)
        if advantages.ndim > 1:
            advantages = advantages.flatten(1).mean(dim=1)
        advantages = advantages.clamp(-loss_cfg.adv_clip_max, loss_cfg.adv_clip_max)

        dsm = ((target - forward_prediction.float()) ** 2).mean(dim=reduce_dims)
        with torch.no_grad():
            ref_dsm = ((target - ref_forward_prediction.detach().float()) ** 2).mean(dim=reduce_dims)
            old_dsm = ((target - old_prediction.detach().float()) ** 2).mean(dim=reduce_dims)

            group_index = group_index.to(device=x0.device).long().flatten()
            group_size = group_size.to(device=x0.device).long().flatten()
            groups, inverse = torch.unique(group_index, return_inverse=True)
            counts = torch.bincount(inverse, minlength=groups.numel())
            expected = torch.zeros_like(counts).scatter_(0, inverse, group_size)
            if not torch.equal(counts, expected):
                raise ValueError(
                    "DGPO needs every rollout group whole inside one micro batch; got group sizes "
                    f"{counts.tolist()} where {expected.tolist()} were expected. Use a micro batch size that is a "
                    "multiple of actor_rollout_ref.rollout.n and keep actor shuffling off."
                )

            per_sample = advantages * loss_cfg.dgpo_beta * (dsm.detach() - ref_dsm) / group_size.float()
            group_sums = torch.zeros(groups.numel(), device=x0.device, dtype=torch.float32)
            group_sums.index_add_(0, inverse, per_sample)
            group_weights = torch.sigmoid(group_sums)
            weights = group_weights[inverse]

            if loss_cfg.dgpo_clip_range > 0:
                ratio = torch.exp(old_dsm - dsm.detach())
                should_clip = torch.where(
                    advantages > 0, ratio > 1.0 + loss_cfg.dgpo_clip_range, ratio < 1.0 - loss_cfg.dgpo_clip_range
                )
            else:
                should_clip = torch.zeros_like(dsm, dtype=torch.bool)

        clipped_dsm = torch.where(should_clip, dsm.detach(), dsm)
        policy_loss = (weights * advantages * clipped_dsm).mean()
        ref_kl_loss = ((forward_prediction.float() - ref_forward_prediction.detach().float()) ** 2).mean()
        loss = policy_loss + loss_cfg.ref_kl_coef * ref_kl_loss

        metrics = {
            "actor/policy_loss": policy_loss.detach().item(),
            "actor/dsm_loss": dsm.detach().mean().item(),
            "actor/ref_dsm_loss": ref_dsm.mean().item(),
            "actor/old_dsm_loss": old_dsm.mean().item(),
            "actor/group_weight_mean": group_weights.mean().item(),
            # Mean distance from the neutral weight 0.5; stays meaningful when metrics are averaged across ranks.
            "actor/group_weight_dev": (group_weights - 0.5).abs().mean().item(),
            "actor/clip_frac": should_clip.float().mean().item(),
            "actor/ref_kl_loss": ref_kl_loss.detach().item(),
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
            advantages=data["advantages"],
            group_index=data["group_index"],
            group_size=data["group_size"],
            config=config,
        )
        return DiffusionLossResult(loss=loss, metrics=metrics)

    @staticmethod
    def validate_trainer_config(config: Any) -> None:
        """Fail at trainer start on configs DGPO cannot train, instead of after the first rollout.

        Each rank receives a contiguous ``batch * n / dp`` slice of the update batch and of every mini batch and
        cuts it into contiguous micro batches, so groups stay whole only when every division is exact in units
        of ``n``. The loss also needs the LoRA ``old`` adapter for rollout and the reference forward, complete
        groups from the v1 sampler, the unstaged NFT engine path, and training timesteps that fit the rollout
        schedule.
        """
        actor_cfg = config.actor_rollout_ref.actor
        model_cfg = config.actor_rollout_ref.model
        rollout_cfg = config.actor_rollout_ref.rollout
        n = int(rollout_cfg.n)
        micro = actor_cfg.get("ppo_micro_batch_size_per_gpu", None)
        sp_size = OmegaConf.select(actor_cfg, "fsdp_config.ulysses_sequence_parallel_size", default=1) or 1
        dp_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes) // int(sp_size)
        use_v1 = bool(OmegaConf.select(config, "trainer.use_v1", default=False))
        update_prompts = int(config.data.train_batch_size)
        if use_v1:
            mode = OmegaConf.select(config, "trainer.v1.trainer_mode", default="sync")
            update_prompts //= int(OmegaConf.select(config, f"trainer.v1.{mode}.parameter_sync_step", default=1) or 1)
        problems = []
        if micro is None or int(micro) % n != 0:
            problems.append(f"ppo_micro_batch_size_per_gpu={micro} must be a multiple of rollout.n={n}")
        if int(actor_cfg.ppo_mini_batch_size) % dp_size != 0:
            mini = actor_cfg.ppo_mini_batch_size
            problems.append(f"ppo_mini_batch_size={mini} must be divisible by dp size {dp_size}")
        if update_prompts % dp_size != 0:
            problems.append(f"the {update_prompts} prompts per actor update must be divisible by dp size {dp_size}")
        if actor_cfg.get("use_dynamic_bsz", False):
            problems.append("use_dynamic_bsz must be False")
        if not int(model_cfg.get("lora_rank", 0) or 0) > 0:
            problems.append("actor_rollout_ref.model.lora_rank must be > 0 (DGPO trains LoRA adapters)")
        if "old" not in tuple(model_cfg.get("policy_state_adapters", ("default",))):
            problems.append('actor_rollout_ref.model.policy_state_adapters must include "old"')
        if rollout_cfg.get("rollout_adapter", None) != "old":
            problems.append("actor_rollout_ref.rollout.rollout_adapter must be old")
        if actor_cfg.get("use_kl_loss", False):
            problems.append("actor_rollout_ref.actor.use_kl_loss must be False (use diffusion_loss.ref_kl_coef)")
        if actor_cfg.get("enable_timestep_staging", False):
            problems.append("actor_rollout_ref.actor.enable_timestep_staging must be False")
        if use_v1 and not OmegaConf.select(config, "trainer.v1.sampler.drop_incomplete_groups", default=False):
            problems.append("trainer.v1.sampler.drop_incomplete_groups must be True so groups arrive complete")
        num_steps = OmegaConf.select(rollout_cfg, "pipeline.num_inference_steps", default=None)
        timestep_range = OmegaConf.select(config, "algorithm.train_timestep_range", default=None)
        count = OmegaConf.select(config, "algorithm.train_timestep_count", default=None)
        if num_steps is not None:
            num_steps = int(num_steps)
            if timestep_range is None:
                start, end = 0, num_steps
            elif len(timestep_range) == 2:
                start, end = int(timestep_range[0]), int(timestep_range[1])
            else:
                start, end = 0, 0
            if not 0 <= start < end <= num_steps:
                problems.append(f"train_timestep_range {timestep_range} must satisfy 0 <= start < end <= {num_steps}")
            elif count is not None and not 1 <= int(count) <= end - start:
                problems.append(f"train_timestep_count={count} must be between 1 and {end - start}, the range width")
        if problems:
            raise ValueError("DGPO config is not supported: " + "; ".join(problems) + ".")

    @staticmethod
    def _first_appearance_index(uid: np.ndarray) -> np.ndarray:
        """Number groups 0, 1, ... in the order their uid first appears (shared with DiffusionART)."""
        return group_index_from_uid(uid)[0].numpy()

    @staticmethod
    def _select_shared_timesteps(
        schedule: torch.Tensor,
        timestep_range: Optional[list[int]],
        count: Optional[int],
        timestep_fraction: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw schedule indices once and apply them to every row, so groups share timesteps."""
        if schedule.ndim != 2:
            raise ValueError(f"DGPO rollout timesteps must have shape [B, T], got {tuple(schedule.shape)}.")
        if not torch.equal(schedule, schedule[:1].expand_as(schedule)):
            raise ValueError("DGPO shares timesteps across the batch, so every rollout must use the same schedule.")
        num_steps = schedule.shape[1]
        start, end = (0, num_steps) if timestep_range is None else (int(timestep_range[0]), int(timestep_range[1]))
        if not 0 <= start < end <= num_steps:
            raise ValueError(f"train_timestep_range {timestep_range} must satisfy 0 <= start < end <= {num_steps}.")
        width = end - start
        if count is not None and not 1 <= int(count) <= width:
            raise ValueError(f"train_timestep_count={count} must be between 1 and {width}, the range width.")
        count = max(1, int(width * timestep_fraction)) if count is None else int(count)
        index = torch.randperm(width, generator=generator)[:count] + start
        selected = schedule[:, index.to(schedule.device)].long()
        # The engine recovers the velocity target as (x_t - x_0) / t, so t must stay positive after flooring.
        if (selected < 1).any():
            raise ValueError("DGPO training timesteps must be >= 1; narrow train_timestep_range to skip the last step.")
        return selected

    @staticmethod
    def prepare_actor_batch(
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
    ) -> DataProto:
        """Turn ODE rollouts into DGPO actor tensors: advantages, groups, shared timesteps and noise."""
        algorithm_cfg = config.algorithm
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("DGPO actor batch requires `uid` in non_tensor_batch.")
        if "latents_clean" not in batch.batch.keys():
            raise ValueError("DGPO actor batch requires `latents_clean` from rollout.")
        timestep_key = next((key for key in ("train_timesteps", "all_timesteps") if key in batch.batch.keys()), None)
        if timestep_key is None:
            raise ValueError("DGPO actor batch requires `train_timesteps` or `all_timesteps` from rollout.")

        # Micro batches are contiguous slices, so groups must be contiguous. Rollouts usually are already
        # (prompts repeated n times in place); only then is the batch left in its original order.
        group_index = DGPOLoss._first_appearance_index(batch.non_tensor_batch["uid"])
        if np.any(np.diff(group_index) < 0):
            order = np.argsort(group_index, kind="stable")
            batch.reorder(torch.from_numpy(order))
            reward_tensor = reward_tensor[torch.from_numpy(order).to(reward_tensor.device)]
            group_index = group_index[order]

        uid = batch.non_tensor_batch["uid"]
        counts = np.bincount(group_index)
        group_size = torch.from_numpy(counts[group_index]).long()
        group_index = torch.from_numpy(group_index).long()

        advantages = DiffusionNFTLoss._compute_group_advantages(
            rewards=reward_tensor,
            uid=uid,
            norm_by_std=algorithm_cfg.norm_adv_by_std_in_grpo,
            global_std=algorithm_cfg.global_std,
        )
        train_timesteps = DGPOLoss._select_shared_timesteps(
            batch.batch[timestep_key],
            timestep_range=getattr(algorithm_cfg, "train_timestep_range", None),
            count=getattr(algorithm_cfg, "train_timestep_count", None),
            timestep_fraction=algorithm_cfg.timestep_fraction,
        )
        latents = batch.batch["latents_clean"]
        num_steps = train_timesteps.shape[1]
        group_noise = torch.randn(int(counts.size), num_steps, *latents.shape[1:], dtype=torch.float32)
        forward_noise = group_noise[group_index]

        batch.batch["train_timesteps"] = train_timesteps
        batch.batch["forward_noise"] = forward_noise
        batch.batch["group_index"] = group_index
        batch.batch["group_size"] = group_size
        batch.batch["advantages"] = advantages[:, None].expand(-1, num_steps)
        batch.batch["returns"] = batch.batch["advantages"]
        batch.batch["sample_level_rewards"] = reward_tensor[:, None].expand(-1, num_steps)
        return batch


@register_diffusion_loss("diffusion_art")
class DiffusionARTLoss(DiffusionLossFn):
    """DiffusionART: aggregation-first reward tilting (Wan et al., arXiv:2610.12572).

    Per prompt, the rewards and clean endpoints of its ``n`` ODE rollouts are aggregated into one
    unit-RMS guidance direction ``G(c)`` (the reward-endpoint covariance, ``art_guidance``). Every
    ODE state ``s_t`` the rollouts visited then regresses the current velocity toward the frozen
    target ``v_old(s_t) - G(c)`` (``v = noise - x0`` convention; the paper writes ``v_old + G``)
    with weight ``t``, and a trust-region mask drops items whose drift from ``v_old`` is both large
    and outward (``art_loss``). Splitting each rollout batch into two optimizer steps
    (``ppo_mini_batch_size = data.train_batch_size / 2``) gives the paper's second, drift-restoring
    update: the second mini batch sees ``v_theta`` after the first step while its target still comes
    from ``v_old``. There are no likelihoods, no differentiable reward, no EMA and no KL term.

    Audio-video pipelines keep one unified token sequence ``[video | audio]`` split at
    ``video_seq_len``. Guidance, mask and MSE are then computed per modality, each from its own
    reward route (``algorithm.art_modality_reward_weights`` or ``reward_functions.*.routing_weights``)
    and with its own unit RMS, and combined with ``art_video_weight`` / ``art_audio_weight``.
    """

    uses_old_adapter = True
    required_model_output_keys = ("forward_prediction", "old_prediction", "t_expanded")
    required_data_keys = ("art_guidance", "sigma_next")
    optional_data_keys = ("video_seq_len", "art_anchor", "rollout_velocity", "segment_sigma", "segment_sigma_next")
    #: Replicated per-sample tensors carrying whole-rollout scalars that the loss reports as ``actor/<name>``:
    #: OmniNFT's ``reward/<name>/{mean,std,min,max}`` and ``reward/sum/{mean,std}`` (same channel and
    #: semantics) plus ART's per-modality guidance statistics.
    batch_metric_prefix = "_art_metric::"

    @staticmethod
    def _segments(width: int, video_seq_len: Optional[torch.Tensor]) -> list[tuple[str, slice]]:
        """Token ranges along dim 1: the whole latent, or ``[video | audio]`` split at ``video_seq_len``."""
        if video_seq_len is None:
            return [("", slice(0, width))]
        values = video_seq_len.reshape(-1)
        split = int(values[0].item())
        if not torch.all(values == split) or not 0 < split < width:
            raise ValueError(f"video_seq_len must be one shared value in (0, {width}), got {values.tolist()}.")
        return [("video", slice(0, split)), ("audio", slice(split, width))]

    @classmethod
    def compute_loss(
        cls,
        *,
        forward_prediction: torch.Tensor,
        old_prediction: torch.Tensor,
        t_expanded: torch.Tensor,
        art_guidance: torch.Tensor,
        sigma_next: torch.Tensor,
        config: DiffusionActorConfig,
        video_seq_len: Optional[torch.Tensor] = None,
        art_anchor: Optional[torch.Tensor] = None,
        rollout_velocity: Optional[torch.Tensor] = None,
        segment_sigma: Optional[torch.Tensor] = None,
        segment_sigma_next: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Masked frozen-target MSE averaged over the micro batch (paper Algorithm 1, step 5).

        Masked and inactive items contribute zero but stay in the denominator, as in the paper's
        ``1/|B|`` average. With ``video_seq_len`` the loss is the ``art_video_weight`` /
        ``art_audio_weight`` average of the two modality losses. ``rollout_velocity`` (the Euler
        velocity implied by consecutive rollout states) is only used for the ``old_rollout_gap`` metric.
        ``segment_sigma`` / ``segment_sigma_next`` ``(B, n_segments)`` give each modality its own flow
        time when a pipeline denoises the modalities on different sigma grids (the pipeline adapter
        supplies them); without them every segment uses ``t_expanded`` / ``sigma_next``.
        """
        loss_cfg = config.diffusion_loss
        sigma = t_expanded.reshape(t_expanded.shape[0], -1)[:, 0]
        segments = cls._segments(forward_prediction.shape[1], video_seq_len)
        for name, value in (("segment_sigma", segment_sigma), ("segment_sigma_next", segment_sigma_next)):
            if value is not None and tuple(value.shape) != (sigma.shape[0], len(segments)):
                raise ValueError(
                    f"{name} must have shape (batch, {len(segments)}) for {len(segments)} segment(s), "
                    f"got {tuple(value.shape)}."
                )
        if (segment_sigma is None) != (segment_sigma_next is None):
            raise ValueError("segment_sigma and segment_sigma_next must be given together.")
        segment_weights = {
            "": 1.0,
            "video": float(loss_cfg.art_video_weight),
            "audio": float(loss_cfg.art_audio_weight),
        }
        total_weight = sum(segment_weights[name] for name, _ in segments)
        per_item_loss = torch.zeros_like(sigma, dtype=torch.float32)
        metrics: dict[str, Any] = {}
        for index, (name, rows) in enumerate(segments):
            anchor = None
            if art_anchor is not None:
                anchor = art_anchor if art_anchor.ndim == 1 else art_anchor[:, index]
            segment_t = sigma if segment_sigma is None else segment_sigma[:, index].float()
            segment_t_next = sigma_next if segment_sigma_next is None else segment_sigma_next[:, index].float()
            terms = art_loss(
                forward_prediction[:, rows],
                old_prediction[:, rows],
                art_guidance[:, rows],
                segment_t,
                segment_t_next,
                guidance_gain=loss_cfg.art_guidance_gain,
                loss_space=loss_cfg.art_loss_space,
                mask_metric=loss_cfg.art_mask_metric,
                mask_threshold=loss_cfg.art_mask_threshold,
                reference_noise_level=loss_cfg.art_reference_noise_level,
                corrective_mask_enabled=loss_cfg.art_corrective_mask_enabled,
                restore_anchor=anchor,
            )
            per_item_loss = per_item_loss + (segment_weights[name] / total_weight) * terms["per_item_loss"]
            prefix = f"actor/{name}/" if name else "actor/"
            gap_velocity = None if rollout_velocity is None else rollout_velocity[:, rows]
            metrics.update(
                cls._segment_metrics(prefix, terms, art_guidance[:, rows], old_prediction[:, rows], gap_velocity)
            )
            if name:
                metrics[f"{prefix}policy_loss"] = terms["per_item_loss"].detach().mean().item()
            metrics.update(cls._bucket_metrics(prefix, terms, segment_t))
        loss = per_item_loss.mean()
        metrics["actor/policy_loss"] = loss.detach().item()
        metrics["actor/total_loss"] = loss.detach().item()
        return loss, metrics

    @staticmethod
    def _segment_metrics(
        prefix: str,
        terms: dict[str, torch.Tensor],
        guidance: torch.Tensor,
        old_prediction: torch.Tensor,
        rollout_velocity: Optional[torch.Tensor],
    ) -> dict[str, float]:
        """Per-segment diagnostics; fractions are over active items so constant-reward batches stay readable."""
        with torch.no_grad():
            active = terms["active"]
            num_active = active.sum().clamp_min(1).float()
            metrics = {
                f"{prefix}unmasked_loss": ((terms["per_item_mse"] * active).sum() / num_active).item(),
                f"{prefix}active_frac": active.float().mean().item(),
                f"{prefix}anchor_frac": terms["anchor"].float().mean().item(),
                f"{prefix}mask_frac": (terms["masked"].float().sum() / num_active).item(),
                f"{prefix}outward_frac": (
                    ((terms["outward_alignment"] > 0) & active).float().sum() / num_active
                ).item(),
                f"{prefix}old_deviate": terms["movement_energy"].mean().item(),
                f"{prefix}movement_score": terms["movement_score"].mean().item(),
                f"{prefix}guidance_rms": guidance.float().flatten(1).square().mean(dim=1).sqrt().mean().item(),
            }
            if rollout_velocity is not None:
                gap = (old_prediction.float() - rollout_velocity.float()).square().flatten(1).mean(dim=1)
                metrics[f"{prefix}old_rollout_gap"] = gap.mean().item()
        return metrics

    @staticmethod
    def _bucket_metrics(prefix: str, terms: dict[str, torch.Tensor], sigma: torch.Tensor) -> dict[str, float]:
        """Trust-region drift and mask rate per flow-time bucket (``t_high`` >= 2/3, ``t_mid``, ``t_low`` < 1/3).

        Every key is emitted on every call (``0.0`` for an empty bucket) together with ``n_items_<bucket>``, the
        number of active items it averaged over: the dp aggregation requires an identical key set and value count
        on every micro batch and rank, so no key may be conditional.
        """
        metrics: dict[str, float] = {}
        with torch.no_grad():
            active = terms["active"]
            buckets = (
                ("t_high", sigma >= 2.0 / 3.0),
                ("t_mid", (sigma >= 1.0 / 3.0) & (sigma < 2.0 / 3.0)),
                ("t_low", sigma < 1.0 / 3.0),
            )
            for name, in_bucket in buckets:
                selected = active & in_bucket
                count = int(selected.sum().item())
                metrics[f"{prefix}n_items_{name}"] = float(count)
                if count:
                    metrics[f"{prefix}movement_score_{name}"] = terms["movement_score"][selected].mean().item()
                    metrics[f"{prefix}mask_frac_{name}"] = terms["masked"][selected].float().mean().item()
                else:
                    metrics[f"{prefix}movement_score_{name}"] = 0.0
                    metrics[f"{prefix}mask_frac_{name}"] = 0.0
        return metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        self.validate_inputs(loss_name="diffusion_art", model_output=model_output, data=data)
        optional = {key: data[key] for key in self.optional_data_keys if key in data.keys()}
        loss, metrics = self.compute_loss(
            forward_prediction=model_output["forward_prediction"],
            old_prediction=model_output["old_prediction"],
            t_expanded=model_output["t_expanded"],
            art_guidance=data["art_guidance"],
            sigma_next=data["sigma_next"],
            config=config,
            **optional,
        )
        metrics.update(self.collect_batch_metrics(data))
        return DiffusionLossResult(loss=loss, metrics=metrics)

    @classmethod
    def collect_batch_metrics(cls, data: TensorDict) -> dict[str, float]:
        """``actor/<name>`` from the replicated whole-rollout scalars attached by ``prepare_actor_batch``."""
        metrics = {}
        for key in data.keys():
            if isinstance(key, str) and key.startswith(cls.batch_metric_prefix):
                name = key[len(cls.batch_metric_prefix) :]
                metrics[f"actor/{name}"] = data[key].detach().float().mean().item()
        return metrics

    @classmethod
    def _attach_batch_metrics(cls, batch: DataProto, values: dict[str, Any], batch_size: int) -> None:
        """Replicate whole-rollout scalars per sample so every micro batch / rank reports the same value."""
        for name, value in values.items():
            scalar = torch.as_tensor(value, dtype=torch.float32).detach().reshape(())
            batch.batch[f"{cls.batch_metric_prefix}{name}"] = scalar.expand(batch_size).clone()

    @staticmethod
    def reward_summaries(scores: torch.Tensor, reward_names: list[str]) -> dict[str, torch.Tensor]:
        """OmniNFT's whole-rollout reward summaries on the raw (un-normalised) components.

        ``reward/sum/{mean,std}`` of the unweighted column sum and ``reward/<name>/{mean,std,min,max}`` per
        component, population std.
        """
        combined = scores.sum(dim=1)
        summaries = {"reward/sum/mean": combined.mean(), "reward/sum/std": combined.std(correction=0)}
        for index, name in enumerate(reward_names):
            column = scores[:, index]
            summaries.update(
                {
                    f"reward/{name}/mean": column.mean(),
                    f"reward/{name}/std": column.std(correction=0),
                    f"reward/{name}/min": column.min(),
                    f"reward/{name}/max": column.max(),
                }
            )
        return summaries

    @staticmethod
    def write_sample_records(directory: str, step: Any, rows: list[dict[str, Any]], seed: Optional[int] = None) -> str:
        """Write one JSON line per rollout sample to ``<directory>/step_<step>.jsonl`` and return the path.

        Also appends ``{"step", "seed", "file", "n_rows"}`` to ``<directory>/index.jsonl`` so files can be ordered
        post hoc even when ``step`` had to fall back to the uid-hash key.
        """
        os.makedirs(directory, exist_ok=True)
        filename = f"step_{step}.jsonl"
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        with open(os.path.join(directory, "index.jsonl"), "a", encoding="utf-8") as handle:
            entry = {
                "step": step if isinstance(step, int) else None,
                "step_key": str(step),
                "seed": seed,
                "file": filename,
                "n_rows": len(rows),
            }
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return path

    @staticmethod
    def _component_matrix(batch: DataProto) -> Optional[tuple[torch.Tensor, list[str]]]:
        """Raw per-component scores ``(B, K)`` and their names from the ``reward/<name>`` extra infos the
        weighted-sum reward manager attaches; ``None`` when the batch carries no component scores."""
        names = sorted(
            key[len("reward/") :]
            for key in batch.non_tensor_batch
            if isinstance(key, str) and key.startswith("reward/") and key.count("/") == 1 and key != "reward/combined"
        )
        if not names:
            return None
        columns = [torch.as_tensor(np.asarray(batch.non_tensor_batch[f"reward/{n}"], dtype=np.float32)) for n in names]
        return torch.stack(columns, dim=1), names

    @staticmethod
    def validate_trainer_config(config: Any) -> None:
        """Fail at trainer start on settings under which the run would no longer be DiffusionART.

        The rollout policy must be the exact old policy (hard-copied ``old`` adapter, no EMA), each
        visited state is used once per rollout (``ppo_epochs=1``), the guidance is a per-prompt
        covariance (``rollout.n >= 2``), the rollout is an ODE, ``v_old`` is recomputed by the
        training forward, so the training-side CFG must equal the rollout CFG, and the per-rank mini
        batch (``ppo_mini_batch_size * n / dp`` samples) must be divisible by
        ``ppo_micro_batch_size_per_gpu``, which the engine otherwise asserts only after the first rollout.
        The paper's two-update restoration follows from ``ppo_mini_batch_size`` splitting the rollout batch
        in two; any other split only logs a warning so ablations stay possible.
        """
        DiffusionLossFn.validate_trainer_config(config)
        algorithm_cfg = config.algorithm
        actor_cfg = config.actor_rollout_ref.actor
        model_cfg = config.actor_rollout_ref.model
        rollout_cfg = config.actor_rollout_ref.rollout
        problems = []
        if not int(model_cfg.get("lora_rank", 0) or 0) > 0:
            problems.append("actor_rollout_ref.model.lora_rank must be > 0 (DiffusionART trains LoRA adapters)")
        if "old" not in tuple(model_cfg.get("policy_state_adapters", ("default",))):
            problems.append('actor_rollout_ref.model.policy_state_adapters must include "old"')
        if rollout_cfg.get("rollout_adapter", None) != "old":
            problems.append("actor_rollout_ref.rollout.rollout_adapter must be old")
        if algorithm_cfg.get("old_policy_decay_schedule", "copy") != "copy":
            problems.append(
                "algorithm.old_policy_decay_schedule must be copy (the rollout policy is the exact old policy)"
            )
        if algorithm_cfg.get("old_policy_decay", None) not in (None, 0, 0.0):
            problems.append("algorithm.old_policy_decay must be null or 0.0 (no EMA)")
        if int(algorithm_cfg.get("old_policy_update_interval", 1)) != 1:
            problems.append("algorithm.old_policy_update_interval must be 1")
        if int(actor_cfg.get("ppo_epochs", 1)) != 1:
            problems.append(
                "actor_rollout_ref.actor.ppo_epochs must be 1 (each visited state is used once per rollout)"
            )
        if int(rollout_cfg.n) < 2:
            problems.append("actor_rollout_ref.rollout.n must be >= 2 (the guidance is a per-prompt covariance)")
        noise_level = OmegaConf.select(rollout_cfg, "algo.noise_level", default=None)
        if noise_level is not None and float(noise_level) != 0.0:
            problems.append("actor_rollout_ref.rollout.algo.noise_level must be 0.0 (DiffusionART rolls out an ODE)")
        if actor_cfg.get("use_kl_loss", False):
            problems.append("actor_rollout_ref.actor.use_kl_loss must be False (DiffusionART has no KL term)")
        if actor_cfg.get("enable_timestep_staging", False):
            problems.append("actor_rollout_ref.actor.enable_timestep_staging must be False")
        if bool(algorithm_cfg.get("paired_preference", False)):
            problems.append(
                "algorithm.paired_preference must be False (the trainers double the mini batch for it, which the "
                "per-rank micro-batching check below does not model; DiffusionART's pairwise variant is "
                "algorithm.art_pairwise_reward)"
            )
        for key in ("guidance_scale", "true_cfg_scale", "video_cfg_scale", "audio_cfg_scale"):
            model_value = OmegaConf.select(config, f"actor_rollout_ref.model.pipeline.{key}", default=None)
            rollout_value = OmegaConf.select(config, f"actor_rollout_ref.rollout.pipeline.{key}", default=None)
            if model_value is None and rollout_value is None:
                continue
            if float(model_value if model_value is not None else 1.0) != float(
                rollout_value if rollout_value is not None else 1.0
            ):
                problems.append(
                    f"actor_rollout_ref.model.pipeline.{key}={model_value} must equal "
                    f"actor_rollout_ref.rollout.pipeline.{key}={rollout_value} "
                    "(v_old is recomputed by the training forward)"
                )
        # The training rollout may only use CFG: the training adapter recomputes v_old with CFG alone, so STG,
        # modality and rescale guidance must stay neutral there (validation rollouts are free to use them).
        for key, neutral in (
            ("video_stg_scale", 0.0),
            ("audio_stg_scale", 0.0),
            ("video_modality_scale", 1.0),
            ("audio_modality_scale", 1.0),
            ("video_rescale_scale", 0.0),
            ("audio_rescale_scale", 0.0),
        ):
            value = OmegaConf.select(config, f"actor_rollout_ref.rollout.pipeline.{key}", default=None)
            if value is not None and float(value) != neutral:
                problems.append(
                    f"actor_rollout_ref.rollout.pipeline.{key}={value} must be {neutral} for the training rollout "
                    "(only CFG is reproduced when v_old is recomputed); set it in rollout.val_kwargs.pipeline instead"
                )
        # Micro-batching: each rank receives ppo_mini_batch_size * n / dp samples per mini batch and the engine
        # (verl prepare_micro_batches, force_group_size defaults to 1) asserts that this count is divisible by
        # ppo_micro_batch_size_per_gpu -- after the first rollout, so check it here. ART guidance is per sample, so
        # micro batches need not hold whole rollout groups.
        n = int(rollout_cfg.n)
        micro = actor_cfg.get("ppo_micro_batch_size_per_gpu", None)
        sp_size = int(OmegaConf.select(actor_cfg, "fsdp_config.ulysses_sequence_parallel_size", default=1) or 1)
        dp_size = max(int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes) // sp_size, 1)
        mini_samples = int(actor_cfg.ppo_mini_batch_size) * n
        if mini_samples % dp_size:
            problems.append(
                f"ppo_mini_batch_size * rollout.n = {mini_samples} samples must be divisible by the dp size {dp_size}"
            )
        elif micro is not None and (mini_samples // dp_size) % int(micro):
            per_gpu = mini_samples // dp_size
            problems.append(
                f"each rank gets {per_gpu} samples per mini batch (ppo_mini_batch_size "
                f"{actor_cfg.ppo_mini_batch_size} x rollout.n {n} / dp {dp_size}), which must be divisible by "
                f"ppo_micro_batch_size_per_gpu={micro}; set it to a divisor of {per_gpu} or raise ppo_mini_batch_size"
            )
        if problems:
            raise ValueError("DiffusionART config is not supported: " + "; ".join(problems) + ".")

        update_prompts = int(config.data.train_batch_size)
        if bool(OmegaConf.select(config, "trainer.use_v1", default=False)):
            mode = OmegaConf.select(config, "trainer.v1.trainer_mode", default="sync")
            update_prompts //= int(OmegaConf.select(config, f"trainer.v1.{mode}.parameter_sync_step", default=1) or 1)
        mini = int(actor_cfg.ppo_mini_batch_size)
        if update_prompts != 2 * mini:
            logger.warning(
                "DiffusionART (paper Sec. 4.2) applies two disjoint updates per rollout batch, i.e. "
                "ppo_mini_batch_size = prompts per update / 2; got ppo_mini_batch_size=%d for %d prompts.",
                mini,
                update_prompts,
            )

    @staticmethod
    def _sample_records(
        batch: DataProto,
        reward_tensor: torch.Tensor,
        group_index: torch.Tensor,
        components: Optional[tuple[torch.Tensor, list[str]]],
        standardized: Optional[tuple[torch.Tensor, list[str]]],
        records: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
    ) -> list[dict[str, Any]]:
        """One JSON-serialisable row per rollout sample: identity, raw and standardised reward components, routed
        per-modality rewards (and pairwise labels / anchors), group index and whether its group had guidance."""
        uid = batch.non_tensor_batch["uid"]
        extra_info = batch.non_tensor_batch.get("extra_info")
        rows = []
        for i in range(len(uid)):
            row: dict[str, Any] = {
                "sample": i,
                "uid": str(uid[i]),
                "group_index": int(group_index[i]),
                "reward": float(reward_tensor[i]),
            }
            if extra_info is not None and isinstance(extra_info[i], dict) and "index" in extra_info[i]:
                row["prompt_index"] = extra_info[i]["index"]
            if components is not None:
                scores, names = components
                row["reward_raw"] = {name: float(scores[i, k]) for k, name in enumerate(names)}
            if standardized is not None:
                scores, names = standardized
                row["reward_standardized"] = {name: float(scores[i, k]) for k, name in enumerate(names)}
            for key, values in records.items():
                row[key] = float(values[i]) if values.dtype.is_floating_point else bool(values[i])
            row["anchor"] = bool(anchors[0][i]) if len(anchors) == 1 else [bool(a[i]) for a in anchors]
            rows.append(row)
        return rows

    @staticmethod
    def _global_step(batch: DataProto) -> Optional[int]:
        """The trainer step of this rollout, if the batch carries it.

        The rollout strategy stamps ``global_steps`` per sample into the batch (non-tensor or tensor field);
        ``meta_info`` is honoured too but must never be *written* by trainers: ``DataProto.to_tensordict`` folds
        ``meta_info`` into the worker TensorDict and asserts that no key collides with a batch field.
        """
        value = batch.meta_info.get("global_steps")
        if value is None and "global_steps" in batch.non_tensor_batch:
            value = np.asarray(batch.non_tensor_batch["global_steps"]).reshape(-1)[0]
        if value is None and "global_steps" in batch.batch.keys():
            value = batch.batch["global_steps"].reshape(-1)[0].item()
        return None if value is None else int(value)

    @classmethod
    def _step_seed(cls, batch: DataProto, base_seed: int) -> int:
        """Deterministic per-rollout seed: ``data_loader_seed + global_steps``, or a hash of the ``uid``s when the
        batch carries no step (uids are fresh every rollout)."""
        global_steps = cls._global_step(batch)
        if global_steps is not None:
            return base_seed + global_steps
        return base_seed + zlib.crc32("".join(map(str, batch.non_tensor_batch["uid"])).encode())

    @staticmethod
    def _trajectory(batch: DataProto) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(states, next_states, timesteps)`` with ``T`` entries each, for both rollout layouts.

        Image rollouts emit ``all_latents`` ``(B, T + 1, ...)`` whose last entry is the clean endpoint;
        audio-video pipelines emit ``all_latents`` ``(B, T, ...)`` plus ``all_next_latents``.
        """
        all_latents = batch.batch["all_latents"]
        all_timesteps = batch.batch["all_timesteps"]
        batch_size, num_steps = all_timesteps.shape
        if "all_next_latents" in batch.batch.keys():
            next_states = batch.batch["all_next_latents"]
            if (
                all_latents.ndim < 3
                or all_latents.shape[:2] != (batch_size, num_steps)
                or next_states.shape != all_latents.shape
            ):
                raise ValueError(
                    "DiffusionART needs the full ODE trajectory: `all_latents` and `all_next_latents` must both have "
                    f"shape (B, T, ...) for T timesteps; got {tuple(all_latents.shape)} / {tuple(next_states.shape)}."
                )
            return all_latents, next_states, all_timesteps
        if all_latents.ndim < 3 or all_latents.shape[:2] != (batch_size, num_steps + 1):
            raise ValueError(
                "DiffusionART needs the full ODE trajectory: `all_latents` must have shape (B, T + 1, ...) for "
                f"`all_timesteps` of shape (B, T); got {tuple(all_latents.shape)} and {tuple(all_timesteps.shape)}."
            )
        return all_latents[:, :-1], all_latents[:, 1:], all_timesteps

    @staticmethod
    def _component_scores(batch: DataProto, name: str) -> torch.Tensor:
        """Per-sample score of one named reward from the reward-manager ``reward/<name>`` extra infos."""
        key = f"reward/{name}"
        if key in batch.non_tensor_batch:
            return torch.as_tensor(np.asarray(batch.non_tensor_batch[key], dtype=np.float32)).reshape(-1)
        raise ValueError(
            f"DiffusionART modality routing needs per-component scores for reward {name!r}: `reward/{name}` in the "
            "batch extra infos (the weighted-sum reward manager attaches them)."
        )

    @classmethod
    def _modality_rewards(
        cls, batch: DataProto, reward_tensor: torch.Tensor, config: Any, modalities: list[str]
    ) -> tuple[dict[str, torch.Tensor], Optional[tuple[torch.Tensor, list[str]]]]:
        """Route named reward components to the video/audio latents (OmniNFT-style), or share the scalar reward.

        Returns the routed reward per modality and, when components were routed, the standardised component
        matrix ``(B, K)`` with its names (for the per-sample records).

        Components are centred per prompt and scaled like DiffusionNFT advantages
        (``algorithm.norm_adv_by_std_in_grpo`` / ``global_std``) before the weighted sum, so routing
        weights act on comparable scales; the guidance itself is invariant to per-prompt affine reward
        changes, so only the relative component scales matter.
        """
        rewards = reward_tensor.detach().float().cpu()
        if modalities == [""]:
            return {"": rewards}, None
        algorithm_cfg = config.algorithm
        routing = algorithm_cfg.get("art_modality_reward_weights", None)
        if routing is None:
            routing = {}
            reward_cfg = getattr(config, "reward", None)
            reward_functions = reward_cfg.get("reward_functions", None) if hasattr(reward_cfg, "get") else None
            if reward_functions is not None:
                for name, entry in reward_functions.items():
                    weights = entry.get("routing_weights", None)
                    if weights is None:
                        continue
                    for modality in modalities:
                        value = float(weights.get(modality, 0.0) or 0.0)
                        if value > 0:
                            routing.setdefault(modality, {})[name] = value
        if not routing:
            return {modality: rewards for modality in modalities}, None
        names = sorted({name for weights in routing.values() for name in weights})
        uid = batch.non_tensor_batch["uid"]
        components = torch.stack(
            [
                DiffusionNFTLoss._compute_group_advantages(
                    rewards=cls._component_scores(batch, name),
                    uid=uid,
                    norm_by_std=algorithm_cfg.norm_adv_by_std_in_grpo,
                    global_std=algorithm_cfg.global_std,
                )
                for name in names
            ],
            dim=1,
        )
        routed = {}
        for modality in modalities:
            weights = dict(routing.get(modality) or {})
            if not weights:
                raise ValueError(
                    f"DiffusionART has no reward routed to the {modality} latents; set "
                    f"algorithm.art_modality_reward_weights.{modality} or "
                    f"reward_functions.*.routing_weights.{modality}."
                )
            vector = torch.tensor([float(weights.get(name, 0.0)) for name in names])
            routed[modality] = components @ vector
        return routed, (components, names)

    @classmethod
    def prepare_actor_batch(
        cls,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
    ) -> DataProto:
        """Turn ODE rollouts into DiffusionART actor tensors: cached states, schedule and per-prompt guidance.

        Requires the full trajectory from a ``diffusion_art`` rollout adapter (see ``_trajectory``).
        ``reward_tensor`` is the ``(B,)`` scalar reward of the weighted-sum reward manager; per-component scores
        for the modality routing and the ``reward/<name>`` summaries come from its ``reward/<name>`` extra infos.
        Writes ``train_latents`` ``(B, S, *latent)``, ``train_timesteps`` / ``train_timesteps_next``
        ``(B, S)`` (``S`` states per rollout drawn from the first ``T - 1`` ODE positions by default,
        ``algorithm.art_exclude_final_step``, times ``algorithm.timestep_fraction``, reseeded per rollout
        with ``data_loader_seed + global_steps``), ``art_guidance``
        ``(B, *latent)`` shared by all rows of a prompt (per modality for audio-video latents), the
        pairwise ``art_anchor`` flags, optionally ``train_latents_next`` for the rollout-velocity
        diagnostic, ``advantages`` / ``returns`` / ``sample_level_rewards`` for logging, and replicated
        whole-rollout scalars (``batch_metric_prefix``) that the loss reports as ``actor/<name>``: OmniNFT's
        ``reward/<name>/{mean,std,min,max}`` and ``reward/sum/{mean,std}`` when raw components are available,
        plus per-modality ``covariance_rms``, ``inactive_group_frac`` and ``routed_reward_{mean,std}``. With
        ``algorithm.art_sample_log_dir`` one JSON line per sample is written to ``<dir>/step_<N>.jsonl``. The raw
        trajectory tensors are dropped afterwards so the worker batch stays as small as DiffusionNFT's.
        """
        algorithm_cfg = config.algorithm
        actor_cfg = config.actor_rollout_ref.actor
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("DiffusionART actor batch requires `uid` in non_tensor_batch.")
        for key in ("all_latents", "all_timesteps"):
            if key not in batch.batch.keys():
                raise ValueError(f"DiffusionART actor batch requires `{key}` from the diffusion_art rollout adapter.")
        batch_metrics: dict[str, Any] = {}
        if reward_tensor.ndim != 1:
            raise ValueError(f"DiffusionART expects a (B,) reward tensor, got {tuple(reward_tensor.shape)}.")
        components = cls._component_matrix(batch)
        if components is not None:
            batch_metrics.update(cls.reward_summaries(*components))
        states, next_states, all_timesteps = cls._trajectory(batch)
        batch_size, num_steps = all_timesteps.shape
        endpoints = next_states[:, -1].float()
        segments = cls._segments(endpoints.shape[1], batch.batch.get("video_seq_len", None))

        uid = batch.non_tensor_batch["uid"]
        group_index, num_groups = group_index_from_uid(uid)
        modality_rewards, standardized = cls._modality_rewards(
            batch, reward_tensor, config, [name for name, _ in segments]
        )
        pairwise = bool(algorithm_cfg.get("art_pairwise_reward", False))
        step_seed = cls._step_seed(batch, int(actor_cfg.data_loader_seed))

        guidance = torch.zeros_like(endpoints)
        anchors = []
        any_active = False
        records: dict[str, torch.Tensor] = {}
        for name, rows in segments:
            rewards = modality_rewards[name]
            label = name or "reward"
            records[f"{label}_routed"] = rewards
            anchor = torch.zeros(batch_size, dtype=torch.bool)
            decided = torch.ones(batch_size, dtype=torch.bool)
            if pairwise:
                rewards = art_pairwise_labels(
                    rewards, group_index, num_groups, torch.Generator().manual_seed(step_seed)
                )
                decided = rewards != 0.5
                anchor = ~decided & bool(decided.any())  # paper App. C.3: ties restore toward v_old
                records[f"{label}_pairwise_label"] = rewards
            direction, active, covariance_rms = art_guidance(
                endpoints[:, rows], rewards, group_index, num_groups, return_covariance_rms=True
            )
            row_direction = direction[group_index] * decided.view(-1, *([1] * (direction.ndim - 1))).float()
            guidance[:, rows] = row_direction
            anchors.append(anchor)
            any_active |= bool(active.any())
            records[f"{label}_group_active"] = active[group_index]
            metric_prefix = f"{name}/" if name else ""
            batch_metrics[f"{metric_prefix}inactive_group_frac"] = 1.0 - active.float().mean()
            batch_metrics[f"{metric_prefix}covariance_rms"] = (
                covariance_rms[active].mean() if bool(active.any()) else torch.tensor(0.0)
            )
            batch_metrics[f"{metric_prefix}routed_reward_mean"] = modality_rewards[name].mean()
            batch_metrics[f"{metric_prefix}routed_reward_std"] = modality_rewards[name].std(correction=0)
        if not any_active:
            logger.warning(
                "DiffusionART: every prompt group in this batch has constant rewards; the update is a no-op."
            )
        cls._attach_batch_metrics(batch, batch_metrics, batch_size)
        sample_log_dir = algorithm_cfg.get("art_sample_log_dir", None)
        if sample_log_dir:
            global_steps = cls._global_step(batch)
            try:
                cls.write_sample_records(
                    str(sample_log_dir),
                    global_steps if global_steps is not None else f"uid{step_seed}",
                    cls._sample_records(batch, reward_tensor, group_index, components, standardized, records, anchors),
                    seed=step_seed,
                )
            except OSError as exc:  # diagnostics must never abort a training step
                logger.warning("DiffusionART: could not write per-sample records to %s: %s", sample_log_dir, exc)

        pool = num_steps - 1 if algorithm_cfg.get("art_exclude_final_step", True) and num_steps > 1 else num_steps
        # Fresh random subset of ODE positions per rollout batch (paper Algorithm 1); DiffusionNFT reseeds the same
        # helper with the fixed data_loader_seed every step, which would train the same positions each step.
        step_index = DiffusionNFTLoss._select_train_timesteps(
            torch.arange(pool).expand(batch_size, -1),
            timestep_fraction=algorithm_cfg.timestep_fraction,
            seed=step_seed,
        )
        num_train = step_index.shape[1]
        next_timesteps = torch.cat([all_timesteps[:, 1:], torch.zeros_like(all_timesteps[:, :1])], dim=1)
        latent_index = step_index.view(batch_size, num_train, *([1] * (states.ndim - 2))).expand(
            -1, -1, *states.shape[2:]
        )

        advantages = DiffusionNFTLoss._compute_group_advantages(
            rewards=reward_tensor,
            uid=uid,
            norm_by_std=algorithm_cfg.norm_adv_by_std_in_grpo,
            global_std=algorithm_cfg.global_std,
        )
        batch.batch["train_latents"] = states.gather(1, latent_index)
        if algorithm_cfg.get("art_log_rollout_velocity_gap", False):
            batch.batch["train_latents_next"] = next_states.gather(1, latent_index)
        batch.batch["train_timesteps"] = all_timesteps.gather(1, step_index)
        batch.batch["train_timesteps_next"] = next_timesteps.gather(1, step_index)
        segment_timesteps = batch.batch.get("all_segment_timesteps", None)
        if segment_timesteps is not None:
            # (B, T, K) per-segment flow times (modalities on different sigma grids); same gather as above.
            if segment_timesteps.shape[:2] != all_timesteps.shape or segment_timesteps.ndim != 3:
                raise ValueError(
                    "all_segment_timesteps must have shape (B, T, n_segments) matching all_timesteps, got "
                    f"{tuple(segment_timesteps.shape)} vs {tuple(all_timesteps.shape)}."
                )
            if segment_timesteps.shape[-1] != len(segments):
                raise ValueError(
                    f"all_segment_timesteps has {segment_timesteps.shape[-1]} columns for {len(segments)} segment(s)."
                )
            segment_next = torch.cat([segment_timesteps[:, 1:], torch.zeros_like(segment_timesteps[:, :1])], dim=1)
            segment_index = step_index.unsqueeze(-1).expand(-1, -1, segment_timesteps.shape[-1])
            batch.batch["train_segment_timesteps"] = segment_timesteps.gather(1, segment_index)
            batch.batch["train_segment_timesteps_next"] = segment_next.gather(1, segment_index)
        batch.batch["art_guidance"] = guidance
        if pairwise:
            batch.batch["art_anchor"] = anchors[0] if len(anchors) == 1 else torch.stack(anchors, dim=1)
        batch.batch["latents_clean"] = endpoints
        batch.batch["advantages"] = advantages[:, None].expand(-1, num_train)
        batch.batch["returns"] = batch.batch["advantages"]
        batch.batch["sample_level_rewards"] = reward_tensor[:, None].expand(-1, num_train)
        for key in ("all_latents", "all_next_latents", "all_log_probs", "all_segment_timesteps"):
            if key in batch.batch.keys():
                del batch.batch[key]
        return batch


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


@register_diffusion_loss("dmd2")
class DMDLoss(DiffusionLossFn):
    """DMD2 student surrogate and batch dispatch for fake-score denoising.

    Inputs share an explicit normalized latent layout ``(B, ...)``. The engine
    supplies detached score predictions and detached fake-stage model inputs;
    this loss also enforces the target/score stop-gradient boundary.
    """

    @classmethod
    def compute_loss(
        cls,
        *,
        generated_x0: torch.Tensor,
        fake_x0: torch.Tensor,
        teacher_x0: torch.Tensor,
        normalization_epsilon: float = 1e-6,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the fp32 distribution-matching surrogate, not paired regression."""
        from verl_omni.trainer.diffusion.distillation.utils import dmd_gradient, dmd_surrogate_loss

        gradient, normalizer, nonfinite = dmd_gradient(fake_x0, teacher_x0, generated_x0, normalization_epsilon)
        loss, active = dmd_surrogate_loss(generated_x0, gradient, gradient_mask)
        return loss, {
            "dmd/loss": loss.detach(),
            "dmd/normalizer": normalizer.mean(),
            "dmd/gradient_norm": gradient.norm(),
            "dmd/nonfinite": nonfinite,
            "dmd/active_elements": active,
        }

    def validate_inputs(self, *, loss_name: str, model_output: dict[str, Any], data: TensorDict) -> None:
        """Validate the stage-specific tensors without requiring PPO inputs."""
        stage = tu.get_non_tensor_data(data, "dmd_stage", default="student")
        if stage == "student":
            required = ("generated_x0", "fake_x0", "teacher_x0")
        elif stage == "fake_score":
            required = ("generated_x0", "noise_pred", "noise")
        else:
            raise ValueError(f"Invalid dmd_stage {stage!r}; expected 'student' or 'fake_score'.")
        missing = [key for key in required if key not in model_output]
        if missing:
            raise KeyError(f"Diffusion loss `{loss_name}` is missing model_output keys: {missing}")

    def __call__(self, *, config: DiffusionActorConfig, model_output: dict[str, Any], data: TensorDict):
        """Dispatch DMD2 loss with an engine-provided mask, falling back to the batch mask."""
        self.validate_inputs(loss_name="dmd2", model_output=model_output, data=data)
        stage = tu.get_non_tensor_data(data, "dmd_stage", default="student")
        gradient_mask = model_output.get("gradient_mask", data.get("gradient_mask"))
        if stage == "student":
            loss, metrics = self.compute_loss(
                generated_x0=model_output["generated_x0"],
                fake_x0=model_output["fake_x0"],
                teacher_x0=model_output["teacher_x0"],
                normalization_epsilon=tu.get_non_tensor_data(data, "dmd_normalization_epsilon", default=1e-6),
                gradient_mask=gradient_mask,
            )
        else:
            from verl_omni.trainer.diffusion.distillation.utils import fake_score_loss

            loss, active = fake_score_loss(
                model_output["noise_pred"],
                model_output["noise"],
                model_output["generated_x0"],
                gradient_mask=gradient_mask,
            )
            metrics = {"fake_score/loss": loss.detach(), "fake_score/active_elements": active}
        return DiffusionLossResult(loss=loss, metrics=metrics, add_loss_metric=True)


@register_diffusion_loss("distill_kl")
class DistillKLLoss(DiffusionLossFn):
    """KL divergence between student and teacher reverse-SDE means (online policy distillation)."""

    required_model_output_keys = ("prev_sample_mean", "std_dev_t")
    required_data_keys = ("teacher_prev_sample_mean",)

    @classmethod
    def compute_loss(
        cls,
        *,
        prev_sample_mean: torch.Tensor,
        teacher_prev_sample_mean: torch.Tensor,
        std_dev_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute KL divergence given student and teacher previous sample means.

        Args:
            prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
            teacher_prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
            std_dev_t: (torch.Tensor) shape is (bs, 1, 1)
        """
        kl_loss = ((prev_sample_mean - teacher_prev_sample_mean) ** 2).mean(dim=(1, 2), keepdim=True) / (
            2 * std_dev_t**2
        )
        metrics = {"actor/distill_kl_loss": kl_loss.mean().detach().item()}
        return kl_loss.mean(), metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        distill_loss, metrics = self.compute_loss(
            prev_sample_mean=model_output["prev_sample_mean"],
            teacher_prev_sample_mean=data["teacher_prev_sample_mean"],
            std_dev_t=model_output["std_dev_t"],
        )
        return DiffusionLossResult(loss=distill_loss, metrics=metrics)


@register_diffusion_loss("distill_fm_mse")
class DistillFlowMatchingMSELoss(DiffusionLossFn):
    """MSE between student and teacher flow/noise predictions (online policy distillation)."""

    required_model_output_keys = ("noise_pred",)
    required_data_keys = ("teacher_noise_pred",)

    @classmethod
    def compute_loss(
        cls,
        *,
        noise_pred: torch.Tensor,
        teacher_noise_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the elementwise MSE between student and teacher predictions.

        Args:
            noise_pred: (torch.Tensor) shape is (bs, ...), e.g. (bs, s, c) or (bs, c, h, w)
            teacher_noise_pred: (torch.Tensor) same shape as ``noise_pred``
        """
        mse_loss = ((noise_pred - teacher_noise_pred) ** 2).mean()
        metrics = {"actor/distill_fm_mse_loss": mse_loss.detach().item()}
        return mse_loss, metrics

    def __call__(
        self,
        *,
        config: DiffusionActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> DiffusionLossResult:
        distill_loss, metrics = self.compute_loss(
            noise_pred=model_output["noise_pred"],
            teacher_noise_pred=data["teacher_noise_pred"],
        )
        return DiffusionLossResult(loss=distill_loss, metrics=metrics)
