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
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.workers.config import DiffusionActorConfig


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
        """Compute the DiffusionNFT policy loss and auxiliary metrics."""
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
