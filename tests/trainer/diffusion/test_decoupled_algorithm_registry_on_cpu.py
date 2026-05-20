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
"""CPU tests for decoupled diffusion algorithm handlers."""

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig, DiffusionNFTAlgoConfig
from verl_omni.trainer.diffusion.decoupled_algos import (
    DiffusionNFTAlgorithm,
    diffusion_nft_old_policy_decay,
    get_decoupled_diffusion_algorithm,
)
from verl_omni.trainer.diffusion.diffusion_metric_utils import compute_data_metrics_diffusion
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    RayCoupledDiffusionTrainer,
    RayDecoupledDiffusionTrainer,
    RayDiffusionRLTrainer,
    RayFlowGRPOTrainer,
)


def test_decoupled_registry_returns_diffusion_nft_handler() -> None:
    config = DiffusionAlgoConfig(paradigm="decoupled", name="diffusion_nft")
    handler = get_decoupled_diffusion_algorithm("diffusion_nft", config)

    assert isinstance(handler, DiffusionNFTAlgorithm)
    assert handler.loss_mode == "diffusion_nft"
    assert handler.required_policy_states == ("default", "old", "reference")
    assert handler.rollout_options() == {"rollout_adapter": "old", "collect_mode": "final_latent"}


def test_trainer_hierarchy_keeps_coupled_and_decoupled_paths_separate() -> None:
    assert RayFlowGRPOTrainer is RayCoupledDiffusionTrainer
    assert issubclass(RayCoupledDiffusionTrainer, RayDiffusionRLTrainer)
    assert issubclass(RayDecoupledDiffusionTrainer, RayDiffusionRLTrainer)
    assert not issubclass(RayDecoupledDiffusionTrainer, RayCoupledDiffusionTrainer)
    assert hasattr(RayCoupledDiffusionTrainer, "_compute_old_log_prob")
    assert "_compute_old_log_prob" not in RayDecoupledDiffusionTrainer.__dict__


@pytest.mark.parametrize(
    "decay_type,step,expected",
    [
        (0, 100, 0.0),
        (1, 250, 0.25),
        (1, 1000, 0.5),
        (2, 74, 0.0),
        (2, 100, 0.1875),
        (2, 1000, 0.999),
    ],
)
def test_diffusion_nft_old_policy_decay(decay_type: int, step: int, expected: float) -> None:
    assert diffusion_nft_old_policy_decay(step=step, decay_type=decay_type) == pytest.approx(expected)


def test_diffusion_nft_prepare_actor_batch_maps_group_rewards_to_reward_prob() -> None:
    config = DiffusionAlgoConfig(
        paradigm="decoupled",
        name="diffusion_nft",
        norm_adv_by_std_in_grpo=False,
        diffusion_nft=DiffusionNFTAlgoConfig(adv_clip_max=2.0),
    )
    handler = DiffusionNFTAlgorithm(config)
    rollout_batch = {
        "latents_clean": torch.randn(4, 3, 4, 4),
        "prompt_embeds": torch.randn(4, 5, 8),
        "prompt_embeds_mask": torch.ones(4, 5, dtype=torch.bool),
        "train_timesteps": torch.tensor([[100, 200], [100, 200], [100, 200], [100, 200]]),
        "uid": np.array(["a", "a", "b", "b"], dtype=object),
    }
    rewards = torch.tensor([0.0, 4.0, 1.0, 3.0], dtype=torch.float32)

    actor_batch = handler.prepare_actor_batch(rollout_batch, rewards)

    assert actor_batch["reward_prob"].shape == (4, 2)
    torch.testing.assert_close(actor_batch["reward_prob"][0], torch.zeros(2))
    torch.testing.assert_close(actor_batch["reward_prob"][1], torch.ones(2))
    torch.testing.assert_close(actor_batch["reward_prob"][2], torch.full((2,), 0.25))
    torch.testing.assert_close(actor_batch["reward_prob"][3], torch.full((2,), 0.75))
    torch.testing.assert_close(actor_batch["latents_clean"], rollout_batch["latents_clean"])


def test_diffusion_nft_prepare_actor_batch_applies_timestep_fraction_and_ignores_all_timesteps() -> None:
    config = DiffusionAlgoConfig(
        paradigm="decoupled",
        name="diffusion_nft",
        norm_adv_by_std_in_grpo=False,
        diffusion_nft=DiffusionNFTAlgoConfig(adv_clip_max=2.0, timestep_fraction=0.5),
    )
    handler = DiffusionNFTAlgorithm(config)
    rollout_batch = {
        "latents_clean": torch.randn(2, 3, 4, 4),
        "prompt_embeds": torch.randn(2, 5, 8),
        "prompt_embeds_mask": torch.ones(2, 5, dtype=torch.bool),
        "train_timesteps": torch.tensor([[100, 200, 300, 400], [100, 200, 300, 400]]),
        "all_timesteps": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
        "timestep_shuffle_seed": 123,
        "uid": np.array(["a", "a"], dtype=object),
    }
    rewards = torch.tensor([0.0, 4.0], dtype=torch.float32)

    actor_batch = handler.prepare_actor_batch(rollout_batch, rewards)

    assert actor_batch["train_timesteps"].shape == (2, 2)
    assert not torch.isin(actor_batch["train_timesteps"], rollout_batch["all_timesteps"]).any()
    assert actor_batch["reward_prob"].shape == (2, 2)
    assert actor_batch["advantages"].shape == (2, 2)


def test_diffusion_nft_prepare_actor_batch_requires_train_timesteps() -> None:
    handler = DiffusionNFTAlgorithm(DiffusionAlgoConfig(paradigm="decoupled", name="diffusion_nft"))
    rollout_batch = {
        "latents_clean": torch.randn(2, 3, 4, 4),
        "prompt_embeds": torch.randn(2, 5, 8),
        "prompt_embeds_mask": torch.ones(2, 5, dtype=torch.bool),
        "uid": np.array(["a", "a"], dtype=object),
    }

    with pytest.raises(ValueError, match="train_timesteps"):
        handler.prepare_actor_batch(rollout_batch, torch.tensor([0.0, 1.0]))


def test_diffusion_nft_post_actor_update_delegates_copy_and_ema() -> None:
    class FakeWorkerGroup:
        def __init__(self):
            self.calls = []

        def copy_adapter(self, source, target):
            self.calls.append(("copy", source, target))

        def ema_update_adapter(self, source, target, decay):
            self.calls.append(("ema", source, target, decay))

    copy_handler = DiffusionNFTAlgorithm(
        DiffusionAlgoConfig(
            paradigm="decoupled",
            name="diffusion_nft",
            diffusion_nft=DiffusionNFTAlgoConfig(old_policy_decay=0.0),
        )
    )
    worker_group = FakeWorkerGroup()
    metrics = {}
    copy_handler.post_actor_update(worker_group, global_steps=1, metrics=metrics)
    assert worker_group.calls == [("copy", "default", "old")]
    assert metrics == {
        "old_policy/update_applied": 1.0,
        "old_policy/copy_update": 1.0,
        "old_policy/ema_update": 0.0,
        "old_policy/decay": 0.0,
    }

    ema_handler = DiffusionNFTAlgorithm(
        DiffusionAlgoConfig(
            paradigm="decoupled",
            name="diffusion_nft",
            diffusion_nft=DiffusionNFTAlgoConfig(old_policy_decay=0.25),
        )
    )
    worker_group = FakeWorkerGroup()
    metrics = {}
    ema_handler.post_actor_update(worker_group, global_steps=1, metrics=metrics)
    assert worker_group.calls == [("ema", "default", "old", 0.25)]
    assert metrics == {
        "old_policy/update_applied": 1.0,
        "old_policy/copy_update": 0.0,
        "old_policy/ema_update": 1.0,
        "old_policy/decay": 0.25,
    }


def test_diffusion_nft_config_sync_copies_algorithm_values_to_actor_and_rollout() -> None:
    trainer = RayDiffusionRLTrainer.__new__(RayDiffusionRLTrainer)
    trainer.config = {
        "algorithm": {
            "paradigm": "decoupled",
            "name": "diffusion_nft",
            "diffusion_nft": {
                "mix_beta": 0.75,
                "ref_kl_coef": 0.2,
                "adv_clip_max": 3.0,
                "adaptive_weight_min": 1e-4,
                "collect_mode": "final_latent",
                "rollout_adapter": "old",
            },
        },
        "actor_rollout_ref": {
            "model": {"algorithm": "flow_grpo"},
            "actor": {
                "diffusion_loss": {
                    "loss_mode": "flow_grpo",
                    "diffusion_nft": {
                        "mix_beta": 0.5,
                        "ref_kl_coef": 0.0,
                        "adv_clip_max": 5.0,
                        "adaptive_weight_min": 1e-5,
                    },
                }
            },
            "rollout": {"collect_mode": "trajectory", "rollout_adapter": "default"},
        },
    }
    from omegaconf import OmegaConf

    trainer.config = OmegaConf.create(trainer.config)
    trainer._sync_and_validate_algorithm_config()

    assert trainer.config.actor_rollout_ref.model.algorithm == "diffusion_nft"
    assert trainer.config.actor_rollout_ref.actor.diffusion_loss.loss_mode == "diffusion_nft"
    assert trainer.config.actor_rollout_ref.actor.diffusion_loss.diffusion_nft.mix_beta == pytest.approx(0.75)
    assert trainer.config.actor_rollout_ref.actor.diffusion_loss.diffusion_nft.ref_kl_coef == pytest.approx(0.2)
    assert trainer.config.actor_rollout_ref.actor.diffusion_loss.diffusion_nft.adv_clip_max == pytest.approx(3.0)
    assert trainer.config.actor_rollout_ref.actor.diffusion_loss.diffusion_nft.adaptive_weight_min == pytest.approx(1e-4)
    assert trainer.config.actor_rollout_ref.rollout.collect_mode == "final_latent"
    assert trainer.config.actor_rollout_ref.rollout.rollout_adapter == "old"


def test_decoupled_trainer_prepares_metric_contract_fields() -> None:
    trainer = RayDecoupledDiffusionTrainer.__new__(RayDecoupledDiffusionTrainer)
    trainer.config = type(
        "Config",
        (),
        {
            "algorithm": DiffusionAlgoConfig(
                paradigm="decoupled",
                name="diffusion_nft",
                global_std=False,
                diffusion_nft=DiffusionNFTAlgoConfig(adv_clip_max=1.0),
            ),
            "actor_rollout_ref": type(
                "ActorRolloutRefConfig",
                (),
                {"actor": type("ActorConfig", (), {"data_loader_seed": 0})()},
            )(),
        },
    )()
    steps = 2
    batch = DataProto.from_dict(
        tensors={
            "latents_clean": torch.randn(4, 3, 4, 4),
            "train_timesteps": torch.tensor([[100, 200], [100, 200], [100, 200], [100, 200]]),
            "prompt_embeds": torch.randn(4, 5, 8),
            "prompt_embeds_mask": torch.ones(4, 5, dtype=torch.bool),
        },
        non_tensors={"uid": np.array(["a", "a", "b", "b"], dtype=object)},
    )
    rewards = torch.tensor([[1.0], [3.0], [2.0], [4.0]])

    prepared = trainer._prepare_decoupled_actor_batch(batch, rewards)

    assert prepared.batch["sample_level_rewards"].shape == (4, steps)
    assert prepared.batch["returns"].shape == (4, steps)
    assert prepared.batch["advantages"].shape == (4, steps)
    assert prepared.batch["reward_prob"].shape == (4, steps)
    metrics = compute_data_metrics_diffusion(prepared)
    assert "critic/rewards/mean" in metrics
    assert "critic/returns/mean" in metrics
