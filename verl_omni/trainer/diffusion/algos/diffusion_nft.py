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
"""DiffusionNFT-specific trainer helpers."""

from collections import defaultdict
from typing import Any

import numpy as np
import torch


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
    adv_clip_max: float,
    timestep_shuffle_seed: int | None = None,
) -> dict[str, Any]:
    """Prepare final-latent rollout data for DiffusionNFT direct-preference updates."""
    if "latents_clean" not in rollout_batch:
        raise ValueError("DiffusionNFT direct-preference training requires `latents_clean` from rollout.")
    if "train_timesteps" not in rollout_batch:
        raise ValueError("DiffusionNFT direct-preference training requires rollout `train_timesteps`.")
    if "uid" not in rollout_batch:
        raise ValueError("DiffusionNFT direct-preference training requires non-tensor `uid` groups.")

    advantages = compute_diffusion_nft_group_advantages(
        rewards=rewards,
        uid=rollout_batch["uid"],
        norm_by_std=config.norm_adv_by_std_in_grpo,
        global_std=config.global_std,
    )
    reward_prob = diffusion_nft_advantage_to_reward_prob(
        advantages,
        adv_clip_max=adv_clip_max,
        adv_mode=config.adv_mode,
    )
    train_timesteps = select_diffusion_nft_train_timesteps(
        train_timesteps=rollout_batch["train_timesteps"],
        timestep_fraction=config.timestep_fraction,
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
