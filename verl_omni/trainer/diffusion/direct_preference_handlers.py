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
"""Direct-preference trainer hooks for diffusion algorithms."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from verl import DataProto

from verl_omni.trainer.diffusion.algos.diffusion_nft import (
    diffusion_nft_old_policy_decay,
    prepare_diffusion_nft_actor_batch,
)


DIRECT_PREFERENCE_HANDLER_REGISTRY: dict[str, "DirectPreferenceHandler"] = {}


def register_direct_preference_handler(name: str):
    """Register direct-preference trainer hooks for an algorithm name."""

    def decorator(cls: type["DirectPreferenceHandler"]) -> type["DirectPreferenceHandler"]:
        DIRECT_PREFERENCE_HANDLER_REGISTRY[name] = cls()
        return cls

    return decorator


def get_direct_preference_handler(name: str) -> "DirectPreferenceHandler":
    """Return registered direct-preference hooks for an algorithm name."""
    if name not in DIRECT_PREFERENCE_HANDLER_REGISTRY:
        raise ValueError(
            f"Unsupported direct-preference diffusion algorithm: {name}. "
            f"Supported algorithms are: {list(DIRECT_PREFERENCE_HANDLER_REGISTRY.keys())}"
        )
    return DIRECT_PREFERENCE_HANDLER_REGISTRY[name]


class DirectPreferenceHandler(ABC):
    """Algorithm-specific hooks used by the generic direct-preference trainer loop."""

    def validate_config(self, config: Any) -> None:
        """Validate algorithm-specific config invariants before training starts."""

    @abstractmethod
    def prepare_actor_batch(
        self,
        *,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
        global_steps: int,
    ) -> DataProto:
        """Convert rollout/reward output into the actor update batch."""
        raise NotImplementedError

    def post_actor_update(self, *, trainer: Any, metrics: dict[str, Any] | None = None) -> None:
        """Run algorithm-specific state updates after the actor optimizer step."""


@register_direct_preference_handler("diffusion_nft")
class DiffusionNFTDirectPreferenceHandler(DirectPreferenceHandler):
    """DiffusionNFT direct-preference hooks."""

    def validate_config(self, config: Any) -> None:
        model_cfg = config.actor_rollout_ref.model
        rollout_cfg = config.actor_rollout_ref.rollout
        actor_loss_cfg = config.actor_rollout_ref.actor.diffusion_loss

        policy_state_adapters = tuple(model_cfg.get("policy_state_adapters", ("default",)))
        if "old" not in policy_state_adapters:
            raise ValueError(
                "DiffusionNFT requires actor_rollout_ref.model.policy_state_adapters to include 'old'."
            )
        if rollout_cfg.collect_mode != "final_latent":
            raise ValueError("DiffusionNFT requires actor_rollout_ref.rollout.collect_mode=final_latent.")
        if rollout_cfg.rollout_adapter != "old":
            raise ValueError("DiffusionNFT requires actor_rollout_ref.rollout.rollout_adapter=old.")
        if actor_loss_cfg.loss_mode != "diffusion_nft":
            raise ValueError(
                "DiffusionNFT requires actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft."
            )

    def prepare_actor_batch(
        self,
        *,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        config: Any,
        global_steps: int,
    ) -> DataProto:
        rewards = reward_tensor.squeeze(-1).float() if reward_tensor.ndim > 1 else reward_tensor.float()
        rollout_batch = {key: batch.batch[key] for key in batch.batch.keys()}
        rollout_batch["uid"] = batch.non_tensor_batch["uid"]

        actor_cfg = config.actor_rollout_ref.actor
        actor_batch = prepare_diffusion_nft_actor_batch(
            rollout_batch=rollout_batch,
            rewards=rewards,
            config=config.algorithm,
            adv_clip_max=actor_cfg.diffusion_loss.adv_clip_max,
            timestep_shuffle_seed=int(actor_cfg.data_loader_seed + global_steps),
        )
        for key, value in actor_batch.items():
            if isinstance(value, torch.Tensor):
                batch.batch[key] = value
        return batch

    def post_actor_update(self, *, trainer: Any, metrics: dict[str, Any] | None = None) -> None:
        algo_cfg = trainer.config.algorithm
        if metrics is not None:
            # These are control-plane metrics for the old LoRA adapter refresh, not loss terms.
            metrics["old_policy/update_applied"] = 0.0
            metrics["old_policy/copy_update"] = 0.0
            metrics["old_policy/ema_update"] = 0.0
            metrics["old_policy/decay"] = 0.0
        if trainer.global_steps % algo_cfg.old_policy_update_interval != 0:
            return

        decay = algo_cfg.old_policy_decay
        if decay is None:
            decay = diffusion_nft_old_policy_decay(
                step=trainer.global_steps,
                decay_type=algo_cfg.old_policy_decay_type,
            )

        if metrics is not None:
            metrics["old_policy/update_applied"] = 1.0
            metrics["old_policy/decay"] = float(decay)
        if decay == 0:
            trainer.actor_rollout_wg.copy_adapter(source="default", target="old")
            if metrics is not None:
                metrics["old_policy/copy_update"] = 1.0
        else:
            trainer.actor_rollout_wg.ema_update_adapter(source="default", target="old", decay=decay)
            if metrics is not None:
                metrics["old_policy/ema_update"] = 1.0
