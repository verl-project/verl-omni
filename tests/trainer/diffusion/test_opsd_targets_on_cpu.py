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

"""L1 CPU tests for the torch-only DiffusionOPSD math core.

OPSD target construction (paper Eq. 6-7) and the two-branch policy loss
(Eq. 11-13) are numerically load-bearing but fail silently: a projection that
grows instead of shrinks the displacement, a "descent" branch that recomputes
the ascent gradient, a normalizer that is not detached, or a loss that leaks
gradient into the old prediction all still produce finite losses that keep
training. These tests therefore transcribe the equations independently below
(explicit loops, 2-D tensors, separate reduction code) and pin the invariants:
exact agreement with the reference, the per-sample trust-region norm bound,
ascent/descent behavior of the synthetic reward, gradient routing (only
``forward_prediction``), shared-first-gradient call counts, reward probability
broadcasting, and the float64 normalizer clamp. Everything runs on CPU with no
network access, checkpoints, or package side effects.
"""

import math

import pytest
import torch
from torch.testing import assert_close

from verl_omni.trainer.diffusion.opsd_targets import (
    build_positive_negative_targets,
    build_trust_region_targets,
    clean_output,
    opsd_anchor,
    opsd_two_branch_loss,
    reward_gradient,
)

BATCH = 4
DIM = 3


def _quadratic_reward(center):
    """``reward_fn(y) = -sum((y - center)^2)`` over all non-batch dims, shape ``(B,)``."""

    def reward_fn(y):
        return -((y - center) ** 2).flatten(1).sum(dim=1)

    return reward_fn


class _CountingReward:
    """Reward wrapper that records how many times the reward is evaluated."""

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn
        self.calls = 0

    def __call__(self, y):
        self.calls += 1
        return self.reward_fn(y)


def _reference_trust_region_targets(anchor, reward_fn, *, rho, n_ascent, eta=1.0, direction=1, eps_g=1e-12):
    """Literal, loop-based transcription of paper Eq. 6-7 for 2-D ``(B, D)`` inputs."""
    budget = rho * anchor.flatten(1).norm(dim=1)
    step_len = eta * budget / n_ascent
    x = anchor.clone()
    for _ in range(n_ascent):
        x = x.detach().clone().requires_grad_(True)
        grad = torch.autograd.grad(reward_fn(x).sum(), x)[0]
        grad_norm = grad.flatten(1).norm(dim=1) + eps_g
        x = x.detach() + direction * step_len.view(-1, 1) * grad / grad_norm.view(-1, 1)
    displacement = x - anchor
    displacement_norm = displacement.flatten(1).norm(dim=1)
    scale = (budget / (displacement_norm + eps_g)).clamp(max=1.0)
    return (anchor + displacement * scale.view(-1, 1)).detach()


def _reference_branch_target(anchor, reward_fn, *, rho, n_ascent, direction, eta=1.0, eps_g=1e-12):
    """Same as the trust-region reference, used to check the shared-gradient branches independently."""
    return _reference_trust_region_targets(
        anchor,
        reward_fn,
        rho=rho,
        n_ascent=n_ascent,
        eta=eta,
        direction=direction,
        eps_g=eps_g,
    )


def _reference_two_branch_loss(
    *,
    forward_prediction,
    old_prediction,
    xt,
    t_expanded,
    target_pos,
    target_neg,
    reward_prob,
    beta,
    adv_clip_max,
    eps_gamma,
):
    """Literal transcription of paper Eq. 11-13 (float64 normalizer, detached old/targets)."""
    old = old_prediction.detach()
    v_pos = beta * forward_prediction + (1.0 - beta) * old
    v_neg = (1.0 + beta) * old - beta * forward_prediction
    y_pos = xt - t_expanded * v_pos
    y_neg = xt - t_expanded * v_neg
    reduce_dims = tuple(range(1, y_pos.dim()))
    gamma_pos = (
        (y_pos.detach().double() - target_pos.detach().double()).abs().mean(dim=reduce_dims, keepdim=True)
    ).clamp_min(eps_gamma)
    gamma_neg = (
        (y_neg.detach().double() - target_neg.detach().double()).abs().mean(dim=reduce_dims, keepdim=True)
    ).clamp_min(eps_gamma)
    gamma_pos = gamma_pos.to(y_pos.dtype)
    gamma_neg = gamma_neg.to(y_neg.dtype)
    pos = ((y_pos - target_pos.detach()) ** 2 / gamma_pos).mean(dim=reduce_dims)
    neg = ((y_neg - target_neg.detach()) ** 2 / gamma_neg).mean(dim=reduce_dims)
    reward_prob = reward_prob.detach().reshape((reward_prob.shape[0],) + (1,) * (pos.dim() - 1))
    per_sample = reward_prob * pos + (1.0 - reward_prob) * neg
    return per_sample.mean() * adv_clip_max


def _make_loss_inputs(*, dtype=torch.float32, shape=(BATCH, DIM), t_shape=None):
    torch.manual_seed(1234)
    forward_prediction = torch.randn(*shape, dtype=dtype, requires_grad=True)
    old_prediction = torch.randn(*shape, dtype=dtype)
    xt = torch.randn(*shape, dtype=dtype)
    t_shape = t_shape if t_shape is not None else (shape[0],) + (1,) * (len(shape) - 1)
    t_expanded = torch.rand(*t_shape, dtype=dtype)
    target_pos = torch.randn(*shape, dtype=dtype)
    target_neg = torch.randn(*shape, dtype=dtype)
    reward_prob = torch.rand(shape[0], dtype=dtype)
    return forward_prediction, old_prediction, xt, t_expanded, target_pos, target_neg, reward_prob


class TestCleanOutputAndAnchor:
    def test_clean_output_is_flow_interpolation(self):
        xt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.tensor([[0.25], [0.5]])
        velocity = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
        assert_close(clean_output(xt, t, velocity), torch.tensor([[0.875, 1.75], [2.0, 2.0]]))

    def test_clean_output_keeps_gradient(self):
        xt = torch.ones(2, 3, requires_grad=True)
        t = torch.full((2, 1), 0.5)
        velocity = torch.zeros(2, 3, requires_grad=True)
        out = clean_output(xt, t, velocity)
        assert out.requires_grad
        out.sum().backward()
        assert_close(velocity.grad, -0.5 * torch.ones(2, 3))

    def test_opsd_anchor_is_detached_clean_output(self):
        xt = torch.randn(BATCH, DIM)
        t = torch.rand(BATCH, 1)
        old = torch.randn(BATCH, DIM, requires_grad=True)
        anchor = opsd_anchor(xt, t, old)
        assert anchor.requires_grad is False
        assert_close(anchor, (xt - t * old).detach())


class TestRewardGradient:
    def test_matches_analytic_quadratic_gradient(self):
        center = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
        anchor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        grad = reward_gradient(anchor, _quadratic_reward(center))
        # d/dy of -sum((y - center)^2) is -2 * (y - center).
        assert_close(grad, -2.0 * (anchor.detach() - center))

    def test_rejects_tensor_without_gradient(self):
        with pytest.raises(ValueError, match="requires_grad"):
            reward_gradient(torch.zeros(BATCH, DIM), _quadratic_reward(torch.zeros(BATCH, DIM)))


class TestTrustRegionTargets:
    @pytest.mark.parametrize("rho", [0.05, 0.1])
    @pytest.mark.parametrize("n_ascent", [1, 2, 3])
    @pytest.mark.parametrize("direction", [1, -1])
    @pytest.mark.parametrize("eta", [0.5, 1.0, 2.0])
    def test_matches_reference_and_stays_inside_budget(self, rho, n_ascent, direction, eta):
        torch.manual_seed(7)
        anchor = torch.randn(BATCH, DIM) + 0.5
        reward_fn = _quadratic_reward(torch.full((BATCH, DIM), 1.5))

        target = build_trust_region_targets(anchor, reward_fn, rho=rho, n_ascent=n_ascent, eta=eta, direction=direction)
        reference = _reference_trust_region_targets(
            anchor, reward_fn, rho=rho, n_ascent=n_ascent, eta=eta, direction=direction
        )
        assert target.requires_grad is False
        assert_close(target, reference)

        displacement_norm = (target - anchor).flatten(1).norm(dim=1)
        budget = rho * anchor.flatten(1).norm(dim=1)
        assert torch.all(displacement_norm <= budget + 1e-6)

    def test_ascent_raises_and_descent_lowers_reward(self):
        torch.manual_seed(11)
        anchor = torch.randn(BATCH, DIM)
        center = anchor + 2.0
        reward_fn = _quadratic_reward(center)

        ascent = build_trust_region_targets(anchor, reward_fn, rho=0.1, n_ascent=2, direction=1)
        descent = build_trust_region_targets(anchor, reward_fn, rho=0.1, n_ascent=2, direction=-1)
        anchor_reward = reward_fn(anchor)
        assert torch.all(reward_fn(ascent) > anchor_reward)
        assert torch.all(reward_fn(descent) < anchor_reward)

    def test_target_construction_leaves_no_grad_on_the_anchor(self):
        torch.manual_seed(13)
        anchor = torch.randn(BATCH, DIM, requires_grad=True)
        reward_fn = _quadratic_reward(torch.zeros(BATCH, DIM))
        target = build_trust_region_targets(anchor, reward_fn, rho=0.1, n_ascent=2)

        assert target.requires_grad is False
        assert target.grad_fn is None
        assert anchor.grad is None
        # Construction detaches every intermediate: rewarding the target is pure.
        with torch.enable_grad():
            probe = reward_fn(target).sum()
        assert probe.requires_grad is False

    @pytest.mark.parametrize("n_ascent", [0, -1])
    def test_rejects_non_positive_n_ascent(self, n_ascent):
        anchor = torch.randn(BATCH, DIM)
        with pytest.raises(ValueError, match="n_ascent"):
            build_trust_region_targets(anchor, _quadratic_reward(torch.zeros(BATCH, DIM)), n_ascent=n_ascent)


class TestPositiveNegativeTargets:
    def test_shares_the_first_reward_gradient(self):
        torch.manual_seed(17)
        anchor = torch.randn(BATCH, DIM)
        reward_fn = _CountingReward(_quadratic_reward(torch.zeros(BATCH, DIM)))
        build_positive_negative_targets(anchor, reward_fn, rho=0.1, n_ascent=2)
        # One shared anchor gradient + one gradient per branch for the remaining steps.
        assert reward_fn.calls == 2 * 2 - 1

    @pytest.mark.parametrize("rho", [0.05, 0.1])
    @pytest.mark.parametrize("n_ascent", [1, 2, 3])
    def test_matches_independent_branch_references(self, rho, n_ascent):
        torch.manual_seed(19)
        anchor = torch.randn(BATCH, DIM)
        reward_fn = _quadratic_reward(torch.zeros(BATCH, DIM))

        target_pos, target_neg, _ = build_positive_negative_targets(anchor, reward_fn, rho=rho, n_ascent=n_ascent)
        assert_close(
            target_pos,
            _reference_branch_target(anchor, reward_fn, rho=rho, n_ascent=n_ascent, direction=1),
        )
        assert_close(
            target_neg,
            _reference_branch_target(anchor, reward_fn, rho=rho, n_ascent=n_ascent, direction=-1),
        )

    @pytest.mark.parametrize("n_ascent", [1, 2, 3])
    def test_branches_match_single_direction_builder(self, n_ascent):
        torch.manual_seed(23)
        anchor = torch.randn(BATCH, DIM)
        reward_fn = _quadratic_reward(torch.full((BATCH, DIM), 0.75))

        target_pos, target_neg, _ = build_positive_negative_targets(anchor, reward_fn, rho=0.1, n_ascent=n_ascent)
        assert_close(target_pos, build_trust_region_targets(anchor, reward_fn, rho=0.1, n_ascent=n_ascent, direction=1))
        assert_close(
            target_neg, build_trust_region_targets(anchor, reward_fn, rho=0.1, n_ascent=n_ascent, direction=-1)
        )

    def test_diagnostics_are_finite_and_within_budget(self):
        torch.manual_seed(29)
        anchor = torch.randn(BATCH, DIM)
        reward_fn = _quadratic_reward(torch.zeros(BATCH, DIM))
        target_pos, target_neg, diagnostics = build_positive_negative_targets(anchor, reward_fn, rho=0.1, n_ascent=2)

        expected_keys = {
            "displacement_pos_rel_mean",
            "displacement_neg_rel_mean",
            "budget_rel",
            "anchor_norm_mean",
            "target_pos_norm_mean",
            "target_neg_norm_mean",
            "first_grad_norm_mean",
        }
        assert set(diagnostics) == expected_keys
        for value in diagnostics.values():
            assert isinstance(value, float)
            assert math.isfinite(value)

        anchor_norm = anchor.flatten(1).norm(dim=1)
        pos_rel = ((target_pos - anchor).flatten(1).norm(dim=1) / anchor_norm).mean().item()
        neg_rel = ((target_neg - anchor).flatten(1).norm(dim=1) / anchor_norm).mean().item()
        assert diagnostics["displacement_pos_rel_mean"] == pytest.approx(pos_rel, rel=1e-6)
        assert diagnostics["displacement_neg_rel_mean"] == pytest.approx(neg_rel, rel=1e-6)
        assert diagnostics["budget_rel"] == pytest.approx(0.1, rel=1e-6)
        assert diagnostics["displacement_pos_rel_mean"] <= 0.1 + 1e-6
        assert diagnostics["displacement_neg_rel_mean"] <= 0.1 + 1e-6

    def test_construction_leaves_no_grad_on_the_anchor_or_targets(self):
        torch.manual_seed(31)
        anchor = torch.randn(BATCH, DIM, requires_grad=True)
        reward_fn = _quadratic_reward(torch.zeros(BATCH, DIM))
        target_pos, target_neg, _ = build_positive_negative_targets(anchor, reward_fn, rho=0.1, n_ascent=2)

        assert target_pos.requires_grad is False
        assert target_neg.requires_grad is False
        assert anchor.grad is None


class TestTwoBranchLoss:
    @pytest.mark.parametrize("beta", [1.0, 0.5])
    @pytest.mark.parametrize("shape", [(BATCH, DIM), (BATCH, 2, 3)])
    def test_matches_reference_loss_and_gradient(self, beta, shape):
        forward_prediction, old_prediction, xt, t_expanded, target_pos, target_neg, reward_prob = _make_loss_inputs(
            dtype=torch.float64, shape=shape
        )
        loss, metrics = opsd_two_branch_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
            beta=beta,
        )

        (grad,) = torch.autograd.grad(loss, forward_prediction)

        reference_leaf = forward_prediction.detach().clone().requires_grad_(True)
        reference_loss = _reference_two_branch_loss(
            forward_prediction=reference_leaf,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
            beta=beta,
            adv_clip_max=5.0,
            eps_gamma=1e-5,
        )
        (reference_grad,) = torch.autograd.grad(reference_loss, reference_leaf)

        assert_close(loss, reference_loss, rtol=1e-10, atol=1e-12)
        assert_close(grad, reference_grad, rtol=1e-9, atol=1e-12)

        assert metrics["opsd_total_loss"] == pytest.approx(loss.item(), rel=1e-12)
        assert torch.isfinite(grad).all()

    def test_zero_loss_and_zero_gradient_when_branches_hit_targets(self):
        forward_prediction, old_prediction, xt, t_expanded, _, _, _ = _make_loss_inputs(dtype=torch.float64)
        beta = 1.0
        # Solve for the targets that the current prediction reproduces exactly.
        with torch.no_grad():
            v_pos = beta * forward_prediction + (1.0 - beta) * old_prediction
            v_neg = (1.0 + beta) * old_prediction - beta * forward_prediction
            target_pos = xt - t_expanded * v_pos
            target_neg = xt - t_expanded * v_neg

        loss, metrics = opsd_two_branch_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=torch.full((BATCH,), 0.5, dtype=torch.float64),
            beta=beta,
        )
        (grad,) = torch.autograd.grad(loss, forward_prediction)

        assert loss.item() == 0.0
        assert metrics["opsd_pos_loss"] == 0.0
        assert metrics["opsd_neg_loss"] == 0.0
        assert_close(grad, torch.zeros_like(grad))

    def test_old_prediction_and_targets_are_detached(self):
        forward_prediction, old_prediction, xt, t_expanded, target_pos, target_neg, reward_prob = _make_loss_inputs()
        old_prediction.requires_grad_(True)
        target_pos.requires_grad_(True)
        target_neg.requires_grad_(True)
        reward_prob.requires_grad_(True)

        loss, _ = opsd_two_branch_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
        )

        # No path from the loss to any detached input: autograd reports them as unused.
        for detached_input in (old_prediction, target_pos, target_neg, reward_prob):
            assert torch.autograd.grad(loss, detached_input, allow_unused=True, retain_graph=True)[0] is None

        loss.backward()
        assert forward_prediction.grad is not None
        assert old_prediction.grad is None
        assert target_pos.grad is None
        assert target_neg.grad is None
        assert reward_prob.grad is None

    def test_large_eps_gamma_keeps_loss_and_gradient_finite(self):
        forward_prediction, old_prediction, xt, t_expanded, target_pos, target_neg, reward_prob = _make_loss_inputs(
            dtype=torch.float64
        )
        loss, metrics = opsd_two_branch_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
            eps_gamma=1e3,
        )
        (grad,) = torch.autograd.grad(loss, forward_prediction)

        assert math.isfinite(loss.item())
        assert torch.isfinite(grad).all()
        assert metrics["opsd_total_loss"] == pytest.approx(loss.item(), rel=1e-12)

        reference_loss = _reference_two_branch_loss(
            forward_prediction=forward_prediction.detach().clone().requires_grad_(True),
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
            beta=1.0,
            adv_clip_max=5.0,
            eps_gamma=1e3,
        )
        assert_close(loss, reference_loss, rtol=1e-10, atol=1e-12)

    def test_float64_epsilon_normalizer_stays_finite(self):
        # The residual is ~1e-13, just below the 1e-12 epsilon: the float64
        # normalizer must clamp to eps_gamma (not round the residual away) and
        # still produce a finite, non-zero loss.
        base = torch.full((2, 3), 0.5, dtype=torch.float64)
        xt = base + 1e-13
        residual = (xt - base).mean()
        assert residual < 1e-12
        forward_prediction = torch.zeros_like(base)
        old_prediction = torch.zeros_like(base)
        t_expanded = torch.zeros(2, 1, dtype=torch.float64)
        reward_prob = torch.ones(2, dtype=torch.float64)

        loss, metrics = opsd_two_branch_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=base,
            target_neg=base,
            reward_prob=reward_prob,
            eps_gamma=1e-12,
        )

        expected_pos_loss = residual.item() ** 2 / 1e-12
        assert loss.item() == pytest.approx(expected_pos_loss * 5.0, rel=1e-9)
        assert loss.item() > 0.0
        assert math.isfinite(metrics["opsd_pos_loss"])
        assert math.isfinite(metrics["opsd_neg_loss"])

    def test_reward_prob_broadcast_shapes_and_branch_selection(self):
        forward_prediction, old_prediction, xt, t_expanded, target_pos, target_neg, reward_prob = _make_loss_inputs(
            dtype=torch.float64, shape=(BATCH, 2, 3)
        )
        loss_flat, _ = opsd_two_branch_loss(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
        )
        forward_prediction_2, old_prediction_2, xt_2, t_expanded_2, target_pos_2, target_neg_2, _ = _make_loss_inputs(
            dtype=torch.float64, shape=(BATCH, 2, 3)
        )
        loss_reshaped, _ = opsd_two_branch_loss(
            forward_prediction=forward_prediction_2,
            old_prediction=old_prediction_2,
            xt=xt_2,
            t_expanded=t_expanded_2,
            target_pos=target_pos_2,
            target_neg=target_neg_2,
            reward_prob=reward_prob.view(BATCH, 1, 1),
        )
        assert_close(loss_reshaped, loss_flat, rtol=1e-12, atol=0.0)

        # reward_prob == 1 selects the positive branch, == 0 selects the negative one.
        forward_prediction_3, old_prediction_3, xt_3, t_expanded_3, target_pos_3, target_neg_3, _ = _make_loss_inputs(
            dtype=torch.float64, shape=(BATCH, 2, 3)
        )
        loss_pos, metrics_pos = opsd_two_branch_loss(
            forward_prediction=forward_prediction_3,
            old_prediction=old_prediction_3,
            xt=xt_3,
            t_expanded=t_expanded_3,
            target_pos=target_pos_3,
            target_neg=target_neg_3,
            reward_prob=torch.ones(BATCH, dtype=torch.float64),
        )
        assert loss_pos.item() == pytest.approx(metrics_pos["opsd_pos_loss"] * 5.0, rel=1e-12)
        assert metrics_pos["opsd_reward_prob_mean"] == pytest.approx(1.0)

    def test_metric_contract_and_advantage_scaling(self):
        forward_prediction, old_prediction, xt, t_expanded, target_pos, target_neg, reward_prob = _make_loss_inputs()
        kwargs = dict(
            forward_prediction=forward_prediction,
            old_prediction=old_prediction,
            xt=xt,
            t_expanded=t_expanded,
            target_pos=target_pos,
            target_neg=target_neg,
            reward_prob=reward_prob,
        )
        loss, metrics = opsd_two_branch_loss(**kwargs)
        loss_doubled, _ = opsd_two_branch_loss(**kwargs, adv_clip_max=10.0)

        assert set(metrics) == {
            "opsd_pos_loss",
            "opsd_neg_loss",
            "opsd_policy_loss",
            "opsd_reward_prob_mean",
            "opsd_total_loss",
        }
        for value in metrics.values():
            assert isinstance(value, float)
            assert math.isfinite(value)
        assert metrics["opsd_total_loss"] == pytest.approx(loss.item(), rel=1e-12)
        assert metrics["opsd_reward_prob_mean"] == pytest.approx(reward_prob.mean().item(), rel=1e-6)
        assert loss_doubled.item() == pytest.approx(2.0 * loss.item(), rel=1e-6)
        assert metrics["opsd_policy_loss"] * 5.0 == pytest.approx(metrics["opsd_total_loss"], rel=1e-6)
