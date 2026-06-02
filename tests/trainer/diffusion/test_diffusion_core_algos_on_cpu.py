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

import os

import numpy as np
import pytest
import torch

from verl_omni.pipelines.wan22_dance_grpo.common import (
    apply_cfg,
    flatten,
    sd3_time_shift,
    seed_from_prompt_ids,
)
from verl_omni.trainer.diffusion import diffusion_algos


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    batch_size = 8
    steps = 10
    sample_level_rewards = torch.randn((batch_size, steps), dtype=torch.float32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=sample_level_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)


# ===========================================================================
# Wan2.2 DanceGRPO shared-utility tests
# ===========================================================================


class TestSD3TimeShift:

    def test_shape_preserved(self):
        t = torch.linspace(0, 1, 50)
        out = sd3_time_shift(5.0, t)
        assert out.shape == t.shape
        assert out.dtype == t.dtype

    def test_monotonic(self):
        t = torch.linspace(0, 1, 100)
        out = sd3_time_shift(5.0, t)
        diffs = out[1:] - out[:-1]
        assert torch.all(diffs >= 0), "output must be monotonically non-decreasing"

    def test_shift_one_is_identity(self):
        t = torch.linspace(0, 1, 50)
        out = sd3_time_shift(1.0, t)
        torch.testing.assert_close(out, t)

    def test_edge_cases(self):
        assert sd3_time_shift(3.0, torch.tensor(0.0)).item() == 0.0
        assert sd3_time_shift(3.0, torch.tensor(1.0)).item() == 1.0

    def test_larger_shift_increases_curvature(self):
        t = torch.tensor(0.5)
        low = sd3_time_shift(2.0, t)
        high = sd3_time_shift(10.0, t)
        assert high > low


class TestApplyCFG:

    def test_scale_one_is_positive_only(self):
        pos = torch.randn(2, 16, 8, 8)
        neg = torch.randn(2, 16, 8, 8)
        out = apply_cfg(pos, neg, 1.0)
        torch.testing.assert_close(out, pos)

    def test_scale_two_formula(self):
        pos = torch.ones(2, 4)
        neg = torch.zeros(2, 4)
        out = apply_cfg(pos, neg, 2.0)
        expected = neg + 2.0 * (pos - neg)
        torch.testing.assert_close(out, expected)

    def test_scale_zero_returns_negative_only(self):
        pos = torch.ones(2, 4)
        neg = torch.zeros(2, 4)
        out = apply_cfg(pos, neg, 0.0)
        torch.testing.assert_close(out, neg)


class TestFlatten:

    def test_flat_list(self):
        assert list(flatten([1, 2, 3])) == [1, 2, 3]

    def test_nested_list(self):
        assert list(flatten([[1, 2], [3, [4, 5]]])) == [1, 2, 3, 4, 5]

    def test_empty_nested(self):
        assert list(flatten([[], [[[]]], [1]])) == [1]

    def test_single_element(self):
        assert list(flatten([42])) == [42]


class TestSeedFromPromptIDs:

    def test_tensor_input(self):
        ids = torch.randint(0, 50000, (77,))
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**64

    def test_flat_list_input(self):
        ids = [49406, 320, 1000]
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)

    def test_nested_list_input(self):
        ids = [[49406, 320], [1000, 5001]]
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)

    def test_deterministic(self):
        ids = torch.randint(0, 50000, (77,))
        s1 = seed_from_prompt_ids(ids)
        s2 = seed_from_prompt_ids(ids.clone())
        assert s1 == s2

    def test_different_ids_different_seeds(self):
        ids_a = torch.zeros(10, dtype=torch.long)
        ids_b = torch.ones(10, dtype=torch.long)
        assert seed_from_prompt_ids(ids_a) != seed_from_prompt_ids(ids_b)

    def test_2d_tensor(self):
        ids = torch.randint(0, 50000, (1, 77))
        seed = seed_from_prompt_ids(ids)
        assert isinstance(seed, int)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            seed_from_prompt_ids(42)


def test_dance_grpo_loss_registered_and_callable():
    """``dance_grpo`` loss function is registered and can be invoked."""
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 8
    rollout_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    advantages = torch.randn((batch_size,), dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=dance_grpo",
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    dance_grpo_loss = diffusion_algos.get_diffusion_loss_fn("dance_grpo")
    pg_loss, pg_metrics = dance_grpo_loss.compute_loss(
        old_log_prob=rollout_log_probs,
        log_prob=current_log_probs,
        advantages=advantages,
        config=actor_config,
    )

    assert pg_loss.shape == ()
    assert isinstance(pg_loss.item(), float)
    for key in ("actor/ppo_kl", "actor/pg_clipfrac", "actor/pg_clipfrac_higher", "actor/pg_clipfrac_lower"):
        assert key in pg_metrics


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_grouped_uids(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    """Exercises the len > 1 branch: multiple samples sharing the same prompt UID."""
    steps = 5
    # 4 samples: uid-0 × 2, uid-1 × 2  →  2 groups of size 2
    group_rewards = torch.tensor(
        [[1.0] * steps, [3.0] * steps, [0.0] * steps, [2.0] * steps],
        dtype=torch.float32,
    )
    uid = np.array(["uid-0", "uid-0", "uid-1", "uid-1"], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=group_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (4, steps)

    if not norm_adv_by_std_in_grpo:
        # Without std scaling: advantage = reward - group_mean
        # group uid-0 mean = (1+3)/2 = 2.0  →  advantages: -1, +1
        # group uid-1 mean = (0+2)/2 = 1.0  →  advantages: -1, +1
        torch.testing.assert_close(advantages[0], torch.full((steps,), -1.0))
        torch.testing.assert_close(advantages[1], torch.full((steps,), 1.0))
        torch.testing.assert_close(advantages[2], torch.full((steps,), -1.0))
        torch.testing.assert_close(advantages[3], torch.full((steps,), 1.0))
    else:
        # With std scaling: mean should be 0 for each group
        torch.testing.assert_close(advantages[0:2].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(advantages[2:4].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)


def test_compute_policy_loss_flow_grpo() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    flow_grpo_loss = diffusion_algos.get_diffusion_loss_fn("flow_grpo")
    for step in range(steps):
        pg_loss, pg_metrics = flow_grpo_loss.compute_loss(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            config=actor_config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics
        assert "actor/pg_clipfrac" in pg_metrics
        assert "actor/pg_clipfrac_higher" in pg_metrics
        assert "actor/pg_clipfrac_lower" in pg_metrics


@pytest.mark.parametrize("norm_by_std", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_compute_diffusion_nft_group_advantages(norm_by_std: bool, global_std: bool) -> None:
    # 4 samples in 2 groups of 2
    rewards = torch.tensor([1.0, 3.0, 0.0, 2.0])
    uid = np.array(["uid-0", "uid-0", "uid-1", "uid-1"], dtype=object)

    advantages = diffusion_algos.DiffusionNFTLoss._compute_group_advantages(
        rewards=rewards, uid=uid, norm_by_std=norm_by_std, global_std=global_std
    )

    assert advantages.shape == (4,)
    if not norm_by_std:
        torch.testing.assert_close(advantages[0], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[1], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[2], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[3], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    else:
        # group means are zero after std normalization
        torch.testing.assert_close(advantages[0:2].mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[2:4].mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("adv_mode", ["continuous", "positive_only", "negative_only", "one_only", "binary"])
def test_diffusion_nft_advantage_to_reward_prob(adv_mode: str) -> None:
    adv_clip_max = 5.0
    advantages = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])

    reward_prob = diffusion_algos.DiffusionNFTLoss._advantage_to_reward_prob(
        advantages, adv_clip_max=adv_clip_max, adv_mode=adv_mode
    )

    assert reward_prob.shape == advantages.shape
    assert (reward_prob >= 0).all() and (reward_prob <= 1).all(), "reward_prob must be in [0, 1]"

    if adv_mode == "continuous":
        # clipped to [-5, 5] → mapped to [0, 1]
        torch.testing.assert_close(reward_prob[1], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[2], torch.tensor(0.5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[3], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    elif adv_mode == "positive_only":
        assert (reward_prob >= 0.5).all()
    elif adv_mode == "negative_only":
        assert (reward_prob <= 0.5).all()
    elif adv_mode == "one_only":
        # advantages binarized to {0, 1}, then mapped via (adv / adv_clip_max) / 2 + 0.5
        torch.testing.assert_close(reward_prob[0:3], torch.full((3,), 0.5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[3:5], torch.full((2,), 0.6), atol=1e-5, rtol=1e-5)
    elif adv_mode == "binary":
        # advantages signed to {-1, 0, 1}, then mapped via (adv / adv_clip_max) / 2 + 0.5
        torch.testing.assert_close(reward_prob[0:2], torch.full((2,), 0.4), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[2], torch.tensor(0.5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[3:5], torch.full((2,), 0.6), atol=1e-5, rtol=1e-5)


def test_prepare_diffusion_nft_actor_batch() -> None:
    from types import SimpleNamespace

    B, T, C, H, W = 4, 6, 4, 8, 8
    rewards = torch.randn(B)
    uid = np.array([f"uid-{i // 2}" for i in range(B)], dtype=object)
    rollout_dict = {
        "latents_clean": torch.randn(B, C, H, W),
        "train_timesteps": torch.randint(0, 1000, (B, T)),
        "uid": uid,
        "prompts": torch.zeros(B, 16, dtype=torch.long),
    }
    algorithm_config = SimpleNamespace(
        norm_adv_by_std_in_grpo=True,
        global_std=True,
        adv_mode="continuous",
        timestep_fraction=0.5,
    )
    config = SimpleNamespace(
        algorithm=algorithm_config,
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                diffusion_loss=SimpleNamespace(adv_clip_max=5.0),
                data_loader_seed=42,
            )
        ),
    )

    result = diffusion_algos.DiffusionNFTLoss.prepare_actor_batch(rollout_dict, rewards, config)

    num_train = max(1, int(T * algorithm_config.timestep_fraction))
    assert result["train_timesteps"].shape == (B, num_train)
    assert result["advantages"].shape == (B, num_train)
    assert result["reward_prob"].shape == (B, num_train)
    assert result["returns"].shape == (B, num_train)
    assert result["sample_level_rewards"].shape == (B, num_train)
    assert ((result["reward_prob"] >= 0) & (result["reward_prob"] <= 1)).all()


def test_compute_policy_loss_diffusion_nft() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    B, C, H, W = 4, 4, 8, 8
    x0 = torch.randn(B, C, H, W)
    xt = torch.randn(B, C, H, W)
    t_expanded = torch.full((B, C, H, W), 0.5)
    forward_prediction = torch.randn(B, C, H, W)
    old_prediction = torch.randn(B, C, H, W)
    ref_forward_prediction = torch.randn(B, C, H, W)
    reward_prob = torch.rand(B)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=diffusion_nft",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=4",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    nft_loss = diffusion_algos.get_diffusion_loss_fn("diffusion_nft")
    loss, metrics = nft_loss.compute_loss(
        forward_prediction=forward_prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_forward_prediction,
        x0=x0,
        xt=xt,
        t_expanded=t_expanded,
        reward_prob=reward_prob,
        config=actor_config,
    )

    assert loss.shape == ()
    assert isinstance(loss.item(), float)
    for key in (
        "actor/policy_loss",
        "actor/positive_loss",
        "actor/negative_loss",
        "actor/ref_kl_loss",
        "actor/old_deviate",
        "actor/reward_prob_mean",
        "actor/total_loss",
    ):
        assert key in metrics, key


def test_compute_policy_loss_grpo_guard() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 4
    rollout_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    advantages = torch.randn((batch_size,), dtype=torch.float32)
    old_prev_sample_mean = torch.randn((batch_size, 16, 8, 8), dtype=torch.float32)
    prev_sample_mean = old_prev_sample_mean + 0.01 * torch.randn_like(old_prev_sample_mean)
    std_dev_t = torch.full((batch_size, 1, 1, 1), 0.5, dtype=torch.float32)
    sqrt_dt = torch.full((batch_size,), 0.3, dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=grpo_guard",
                "diffusion_loss.clip_ratio=2e-6",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    grpo_guard_loss = diffusion_algos.get_diffusion_loss_fn("grpo_guard")
    pg_loss, pg_metrics = grpo_guard_loss.compute_loss(
        old_log_prob=rollout_log_probs,
        log_prob=current_log_probs,
        advantages=advantages,
        config=actor_config,
        old_prev_sample_mean=old_prev_sample_mean,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        sqrt_dt=sqrt_dt,
    )

    assert pg_loss.shape == ()
    assert isinstance(pg_loss.item(), float)
    for key in (
        "actor/ppo_kl",
        "actor/pg_clipfrac",
        "actor/pg_clipfrac_higher",
        "actor/pg_clipfrac_lower",
        "actor/ratio_mean",
        "actor/ratio_std",
    ):
        assert key in pg_metrics, key


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_dance_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    """``dance_grpo`` reuses the ``flow_grpo`` advantage estimator."""
    batch_size = 8
    steps = 10
    sample_level_rewards = torch.randn((batch_size, steps), dtype=torch.float32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=sample_level_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)
