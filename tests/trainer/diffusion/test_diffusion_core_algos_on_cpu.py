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

    for step in range(steps):
        pg_loss, pg_metrics = diffusion_algos.compute_diffusion_loss_flow_grpo(
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

    pg_loss, pg_metrics = diffusion_algos.compute_diffusion_loss_grpo_guard(
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
        "actor/grpo_guard/ratio_mean_bias",
        "actor/grpo_guard/sqrt_dt",
        "actor/grpo_guard/sigma_t",
    ):
        assert key in pg_metrics, key
