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
"""CPU tests for the DiffusionNFT prediction-space objective."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.trainer.diffusion.diffusion_algos import compute_diffusion_loss_diffusion_nft
from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig


def _actor_config() -> FSDPDiffusionActorConfig:
    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"),
        version_base=None,
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "ppo_micro_batch_size_per_gpu=4",
                "diffusion_loss.loss_mode=diffusion_nft",
                "diffusion_loss.diffusion_nft.mix_beta=0.5",
                "diffusion_loss.diffusion_nft.ref_kl_coef=0.25",
                "diffusion_loss.diffusion_nft.adaptive_weight_min=1e-5",
            ],
        )
    return omega_conf_to_dataclass(cfg)


def _inputs(batch_size: int = 4, *shape: int):
    if not shape:
        shape = (3, 8, 8)
    torch.manual_seed(0)
    x0 = torch.randn((batch_size, *shape), dtype=torch.float32)
    xt = torch.randn_like(x0)
    t_expanded = torch.rand((batch_size, *([1] * len(shape))), dtype=torch.float32)
    old_prediction = torch.randn_like(x0, requires_grad=True)
    forward_prediction = torch.randn_like(x0, requires_grad=True)
    ref_forward_prediction = torch.randn_like(x0, requires_grad=True)
    return x0, xt, t_expanded, old_prediction, forward_prediction, ref_forward_prediction


def test_diffusion_nft_loss_is_scalar_finite_and_registered_config_works() -> None:
    config = _actor_config()
    x0, xt, t_expanded, old_prediction, forward_prediction, ref_forward_prediction = _inputs()
    reward_prob = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float32)

    loss, metrics = compute_diffusion_loss_diffusion_nft(
        forward_prediction=forward_prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_forward_prediction,
        x0=x0,
        xt=xt,
        t_expanded=t_expanded,
        reward_prob=reward_prob,
        config=config,
    )

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert metrics["actor/policy_loss"] >= 0.0
    assert metrics["actor/ref_kl_loss"] >= 0.0


def test_diffusion_nft_loss_only_backprops_through_current_prediction() -> None:
    config = _actor_config()
    x0, xt, t_expanded, old_prediction, forward_prediction, ref_forward_prediction = _inputs()
    reward_prob = torch.full((x0.shape[0],), 0.5, dtype=torch.float32)

    loss, _ = compute_diffusion_loss_diffusion_nft(
        forward_prediction=forward_prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_forward_prediction,
        x0=x0,
        xt=xt,
        t_expanded=t_expanded,
        reward_prob=reward_prob,
        config=config,
    )
    loss.backward()

    assert forward_prediction.grad is not None
    assert forward_prediction.grad.abs().sum() > 0
    assert old_prediction.grad is None
    assert ref_forward_prediction.grad is None


@pytest.mark.parametrize("reward_value,expected_metric", [(1.0, "actor/positive_loss"), (0.0, "actor/negative_loss")])
def test_diffusion_nft_reward_extremes_select_expected_branch(reward_value: float, expected_metric: str) -> None:
    config = _actor_config()
    x0, xt, t_expanded, old_prediction, forward_prediction, ref_forward_prediction = _inputs()
    reward_prob = torch.full((x0.shape[0],), reward_value, dtype=torch.float32)

    _, metrics = compute_diffusion_loss_diffusion_nft(
        forward_prediction=forward_prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_forward_prediction,
        x0=x0,
        xt=xt,
        t_expanded=t_expanded,
        reward_prob=reward_prob,
        config=config,
    )

    assert metrics[expected_metric] > 0.0
    assert metrics["actor/reward_prob_mean"] == pytest.approx(reward_value)


def test_diffusion_nft_ref_penalty_is_prediction_space_mse() -> None:
    config = _actor_config()
    x0, xt, t_expanded, old_prediction, forward_prediction, _ = _inputs()
    reward_prob = torch.ones((x0.shape[0],), dtype=torch.float32)
    ref_forward_prediction = forward_prediction.detach().clone()

    _, metrics = compute_diffusion_loss_diffusion_nft(
        forward_prediction=forward_prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_forward_prediction,
        x0=x0,
        xt=xt,
        t_expanded=t_expanded,
        reward_prob=reward_prob,
        config=config,
    )

    assert metrics["actor/ref_kl_loss"] == pytest.approx(0.0, abs=1e-7)
