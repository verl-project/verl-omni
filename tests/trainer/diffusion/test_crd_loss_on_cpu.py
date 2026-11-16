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
"""CRD numerical, grouping and adapter contracts without model downloads."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from verl_omni.trainer.diffusion.diffusion_algos import CRDLoss, get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig


def inputs():
    torch.manual_seed(9)
    return {
        **{k: torch.randn(6, 2, 3, requires_grad=True) for k in CRDLoss.required_model_output_keys},
        "advantages": torch.tensor([-2.0, 0.0, 2.0, 1.0, -1.0, 0.0], requires_grad=True),
        "raw_rewards": torch.tensor([0.0, 0.5, 1.0, 0.3, 0.6, 0.7], requires_grad=True),
        "group_index": torch.tensor([2, 2, 2, 9, 9, 9]),
        "group_size": torch.full((6,), 3),
    }


def actor(**kwargs):
    return DiffusionActorConfig(
        strategy="fsdp2",
        rollout_n=3,
        diffusion_loss=DiffusionLossConfig(
            loss_mode="crd",
            ref_kl_coef=kwargs.pop("ref_kl_coef", 0.1),
            crd_kl_min=kwargs.pop("crd_kl_min", 1e-4),
            **kwargs,
        ),
    )


def reference(values, cfg):
    """Loop reference for the within-prompt CRD objective."""
    scores = []
    for current, old, target in zip(
        values["forward_prediction"], values["old_prediction"], values["velocity_target"], strict=False
    ):
        target = target.detach()
        current_norm = (current.double() - target.double()).abs().mean().detach().clamp_min(cfg.adaptive_weight_min)
        old_norm = (old.double() - target.double()).abs().mean().detach().clamp_min(cfg.adaptive_weight_min)
        scores.append(
            (old.detach() - target).square().mean() / old_norm - (current - target).square().mean() / current_norm
        )
    scores = torch.stack(scores)
    rewards = (values["advantages"].detach().clamp(-cfg.adv_clip_max, cfg.adv_clip_max) / cfg.adv_clip_max + 1) / 2
    terms = []
    for i, group in enumerate(values["group_index"]):
        mask = values["group_index"] == group
        residual = cfg.crd_beta * (scores[i] - scores[mask].mean().detach()) - (rewards[i] - rewards[mask].mean())
        anchor_error = (values["forward_prediction"][i] - values["ref_forward_prediction"][i].detach()).square().mean()
        coefficient = cfg.crd_kl_min + values["raw_rewards"][i].detach() * (cfg.ref_kl_coef - cfg.crd_kl_min)
        terms.append(residual.square() * cfg.adv_clip_max / cfg.crd_beta + coefficient * anchor_error)
    return torch.stack(terms).mean()


@pytest.mark.parametrize("beta", [0.1, 1.0, 3.0])
def test_loss_and_gradient_match_reference(beta):
    values = inputs()
    config = actor(crd_beta=beta)
    actual, _ = CRDLoss.compute_loss(**values, config=config)
    expected = reference(values, config.diffusion_loss)
    torch.testing.assert_close(actual, expected)
    (grad,) = torch.autograd.grad(actual, values["forward_prediction"], retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, values["forward_prediction"])
    torch.testing.assert_close(grad, expected_grad)


def test_hand_computed_loss_and_gradient():
    prediction = torch.tensor([[1.0], [3.0]], requires_grad=True)
    loss, metrics = CRDLoss.compute_loss(
        forward_prediction=prediction,
        old_prediction=torch.tensor([[2.0], [2.0]]),
        ref_forward_prediction=torch.zeros(2, 1),
        velocity_target=torch.zeros(2, 1),
        advantages=torch.tensor([-1.0, 1.0]),
        raw_rewards=torch.tensor([0.0, 1.0]),
        group_index=torch.tensor([0, 0]),
        group_size=torch.tensor([2, 2]),
        config=actor(crd_beta=1.0, adv_clip_max=1.0, crd_kl_min=0.1, ref_kl_coef=0.3),
    )
    # Implicit rewards [1, -1], centered rewards [-0.5, 0.5]: residuals [1.5, -1.5].
    assert metrics["actor/crd_policy_loss"] == pytest.approx(2.25)
    assert metrics["actor/crd_kl_loss"] == pytest.approx(1.4)  # (0.1 * 1 + 0.3 * 9) / 2
    assert loss.item() == pytest.approx(3.65)
    loss.backward()
    # Detached denominators and centering give policy gradients [-3, 3], plus KL [0.1, 0.9].
    torch.testing.assert_close(prediction.grad, torch.tensor([[-2.9], [3.9]]))


def test_registry_and_stop_gradients():
    values = inputs()
    result = get_diffusion_loss_fn("crd")(
        config=actor(),
        model_output={k: values[k] for k in CRDLoss.required_model_output_keys},
        data=TensorDict({k: values[k] for k in CRDLoss.required_data_keys}, batch_size=[6]),
    )
    result.loss.backward()
    assert values["forward_prediction"].grad.abs().sum() > 0
    for key in ("old_prediction", "ref_forward_prediction", "velocity_target", "raw_rewards", "advantages"):
        assert values[key].grad is None


def test_centering_is_per_prompt_and_permutation_invariant():
    values = inputs()
    config = actor()
    full, _ = CRDLoss.compute_loss(**values, config=config)
    pieces = [
        CRDLoss.compute_loss(**{k: v[sl] for k, v in values.items()}, config=config)[0]
        for sl in (slice(0, 3), slice(3, 6))
    ]
    torch.testing.assert_close(full, sum(pieces) / 2)
    order = torch.tensor([3, 1, 4, 2, 0, 5])
    shuffled, _ = CRDLoss.compute_loss(**{k: v[order] for k, v in values.items()}, config=config)
    torch.testing.assert_close(full, shuffled)


@pytest.mark.parametrize("reward", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_raw_rewards(reward):
    batch = DataProto.from_dict(
        tensors={"latents_clean": torch.zeros(2, 2, 3), "all_timesteps": torch.tensor([[900, 600]] * 2)},
        non_tensors={"uid": np.array(["a", "a"])},
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "algorithm": {"norm_adv_by_std_in_grpo": True, "global_std": True, "timestep_fraction": 1.0},
        }
    )
    with pytest.raises(ValueError, match="raw rewards"):
        CRDLoss.prepare_actor_batch(batch, torch.full((2,), reward), config)


def test_incomplete_group_rejected():
    values = {k: v[:2] for k, v in inputs().items()}
    with pytest.raises(ValueError, match="complete prompt groups"):
        CRDLoss.compute_loss(**values, config=actor())


def test_perfect_predictions_have_finite_loss_and_gradient():
    values = inputs()
    values["old_prediction"] = values["velocity_target"].detach().clone()
    values["forward_prediction"] = values["velocity_target"].detach().clone().requires_grad_(True)
    loss, _ = CRDLoss.compute_loss(**values, config=actor())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(values["forward_prediction"].grad).all()


def test_reward_adaptive_kl_endpoints():
    for reward, coefficient in [(0.0, 0.0001), (1.0, 0.1)]:
        values = inputs()
        values["raw_rewards"] = torch.full((6,), reward)
        _, metrics = CRDLoss.compute_loss(**values, config=actor())
        assert metrics["actor/crd_kl_coefficient"] == pytest.approx(coefficient)


@pytest.mark.parametrize("schedule", [[900, 0], [900.5, 0.5]])
def test_batch_preparation_reorders_rewards_and_uses_independent_noise(schedule):
    batch = DataProto.from_dict(
        tensors={"latents_clean": torch.zeros(4, 2, 3), "all_timesteps": torch.tensor([schedule] * 4)},
        non_tensors={"uid": np.array(["a", "b", "a", "b"])},
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "algorithm": {
                "norm_adv_by_std_in_grpo": True,
                "global_std": True,
                "timestep_fraction": 1.0,
                "train_timestep_range": None,
                "train_timestep_count": 2,
            },
        }
    )
    result = CRDLoss.prepare_actor_batch(batch, torch.tensor([0.1, 0.2, 0.9, 0.8]), config)
    assert result.non_tensor_batch["uid"].tolist() == ["a", "a", "b", "b"]
    torch.testing.assert_close(result.batch["raw_rewards"], torch.tensor([0.1, 0.9, 0.2, 0.8]))
    torch.testing.assert_close(result.batch["train_timesteps"].sort(dim=1).values, torch.tensor([sorted(schedule)] * 4))
    assert result.batch["forward_noise"].shape == (4, 2, 2, 3)
    assert not torch.equal(result.batch["forward_noise"][0], result.batch["forward_noise"][1])
    assert result.batch["group_size"].tolist() == [2] * 4


@pytest.mark.parametrize(
    "kwargs", [{"crd_beta": 0}, {"crd_beta": float("nan")}, {"crd_kl_min": -0.1}, {"ref_kl_coef": 0}]
)
def test_loss_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        actor(**kwargs)
