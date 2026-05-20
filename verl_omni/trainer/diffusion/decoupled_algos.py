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
"""Thin handler registry for decoupled diffusion RL algorithms."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Protocol

import numpy as np
import torch

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig


class DecoupledDiffusionAlgorithm(Protocol):
    """Control-plane contract used by decoupled diffusion trainers."""

    loss_mode: str
    required_policy_states: tuple[str, ...]

    def rollout_options(self) -> dict[str, Any]: ...

    def prepare_actor_batch(self, rollout_batch: dict[str, Any], rewards: torch.Tensor) -> dict[str, Any]: ...

    def post_actor_update(self, actor_rollout_wg, global_steps: int, metrics: dict[str, Any] | None = None) -> None: ...


DECOUPLED_DIFFUSION_ALGORITHM_REGISTRY: dict[str, Callable[[DiffusionAlgoConfig], DecoupledDiffusionAlgorithm]] = {}


def register_decoupled_diffusion_algorithm(
    name: str,
) -> Callable[[Callable[[DiffusionAlgoConfig], DecoupledDiffusionAlgorithm]], Callable]:
    """Register a decoupled diffusion algorithm handler factory."""

    def decorator(factory: Callable[[DiffusionAlgoConfig], DecoupledDiffusionAlgorithm]) -> Callable:
        DECOUPLED_DIFFUSION_ALGORITHM_REGISTRY[name] = factory
        return factory

    return decorator


def get_decoupled_diffusion_algorithm(name: str, config: DiffusionAlgoConfig) -> DecoupledDiffusionAlgorithm:
    """Return a decoupled algorithm handler by name."""
    if name not in DECOUPLED_DIFFUSION_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unsupported decoupled diffusion algorithm: {name}. "
            f"Supported algorithms are: {list(DECOUPLED_DIFFUSION_ALGORITHM_REGISTRY.keys())}"
        )
    return DECOUPLED_DIFFUSION_ALGORITHM_REGISTRY[name](config)


def diffusion_nft_old_policy_decay(step: int, decay_type: int) -> float:
    """Reference DiffusionNFT old-policy decay schedules."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unsupported DiffusionNFT old_policy_decay_type: {decay_type}")
    return 0.0 if step < flat else min((step - flat) * uprate, uphold)


@register_decoupled_diffusion_algorithm("diffusion_nft")
class DiffusionNFTAlgorithm:
    """DiffusionNFT batch-preparation and policy-state control."""

    loss_mode = "diffusion_nft"
    required_policy_states = ("default", "old", "reference")

    def __init__(self, config: DiffusionAlgoConfig):
        self.config = config
        self.nft_config = config.diffusion_nft

    def rollout_options(self) -> dict[str, Any]:
        return {
            "rollout_adapter": self.nft_config.rollout_adapter,
            "collect_mode": self.nft_config.collect_mode,
        }

    def prepare_actor_batch(self, rollout_batch: dict[str, Any], rewards: torch.Tensor) -> dict[str, Any]:
        uid = rollout_batch["uid"]
        if "train_timesteps" not in rollout_batch:
            raise ValueError("DiffusionNFT requires final-latent rollout to return `train_timesteps`.")

        advantages = self._compute_group_advantages(
            rewards=rewards,
            uid=uid,
            norm_by_std=self.config.norm_adv_by_std_in_grpo,
            global_std=self.config.global_std,
        )
        reward_prob = self._advantage_to_reward_prob(advantages)
        train_timesteps = self._select_train_timesteps(
            train_timesteps=rollout_batch["train_timesteps"],
            seed=rollout_batch.get("timestep_shuffle_seed"),
        )
        if reward_prob.ndim == 1 and train_timesteps.ndim == 2:
            reward_prob = reward_prob[:, None].expand(-1, train_timesteps.shape[1])

        actor_batch = dict(rollout_batch)
        actor_batch["train_timesteps"] = train_timesteps
        actor_batch["advantages"] = advantages[:, None].expand(-1, train_timesteps.shape[1])
        actor_batch["reward_prob"] = reward_prob
        actor_batch["returns"] = actor_batch["advantages"]
        actor_batch["sample_level_rewards"] = rewards[:, None].expand(-1, train_timesteps.shape[1])
        actor_batch["loss_mode"] = self.loss_mode
        actor_batch["training_paradigm"] = "decoupled"
        return actor_batch

    def post_actor_update(self, actor_rollout_wg, global_steps: int, metrics: dict[str, Any] | None = None) -> None:
        if metrics is not None:
            metrics["old_policy/update_applied"] = 0.0
            metrics["old_policy/copy_update"] = 0.0
            metrics["old_policy/ema_update"] = 0.0
            metrics["old_policy/decay"] = 0.0
        if global_steps % self.nft_config.old_policy_update_interval != 0:
            return

        decay = self.nft_config.old_policy_decay
        if decay is None:
            decay = diffusion_nft_old_policy_decay(
                step=global_steps,
                decay_type=self.nft_config.old_policy_decay_type,
            )

        if metrics is not None:
            metrics["old_policy/update_applied"] = 1.0
            metrics["old_policy/decay"] = float(decay)
        if decay == 0:
            actor_rollout_wg.copy_adapter(source="default", target="old")
            if metrics is not None:
                metrics["old_policy/copy_update"] = 1.0
        else:
            actor_rollout_wg.ema_update_adapter(source="default", target="old", decay=decay)
            if metrics is not None:
                metrics["old_policy/ema_update"] = 1.0

    def _compute_group_advantages(
        self,
        *,
        rewards: torch.Tensor,
        uid: np.ndarray,
        norm_by_std: bool,
        global_std: bool,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
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

    def _advantage_to_reward_prob(self, advantages: torch.Tensor) -> torch.Tensor:
        adv_clip_max = self.nft_config.adv_clip_max
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
        if self.nft_config.adv_mode == "positive_only":
            advantages = torch.clamp(advantages, 0, adv_clip_max)
        elif self.nft_config.adv_mode == "negative_only":
            advantages = torch.clamp(advantages, -adv_clip_max, 0)
        elif self.nft_config.adv_mode == "one_only":
            advantages = torch.where(advantages > 0, torch.ones_like(advantages), torch.zeros_like(advantages))
        elif self.nft_config.adv_mode == "binary":
            advantages = torch.sign(advantages)

        reward_prob = (advantages / adv_clip_max) / 2.0 + 0.5
        return torch.clamp(reward_prob, 0, 1)

    def _select_train_timesteps(self, *, train_timesteps: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if train_timesteps.ndim != 2:
            raise ValueError(f"DiffusionNFT `train_timesteps` must have shape [B, T], got {train_timesteps.shape}.")

        num_timesteps = train_timesteps.shape[1]
        num_train_timesteps = max(1, int(num_timesteps * self.nft_config.timestep_fraction))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=train_timesteps.device)
            generator.manual_seed(int(seed))

        permuted = []
        for row in train_timesteps:
            perm = torch.randperm(num_timesteps, device=train_timesteps.device, generator=generator)
            permuted.append(row[perm[:num_train_timesteps]])
        return torch.stack(permuted, dim=0).long()
