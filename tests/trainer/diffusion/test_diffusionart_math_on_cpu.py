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
"""CPU tests for the DiffusionART tensor math (guidance, pairwise labels, drift proxy, masked MSE).

Necessity: the guidance is a per-prompt covariance normalised to unit RMS and the loss is a frozen-target
MSE whose only protection against runaway updates is the trust-region mask; a wrong sign, a missed
zero-signal case or an inverted mask condition would train silently in the wrong direction. These tests
pin each piece to the paper's equations (arXiv:2610.12572, Eq. 5-8 and App. C.2-C.3).
"""

import math

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_art import (
    art_guidance,
    art_loss,
    art_pairwise_labels,
    art_reference_drift,
)

C, H, W = 2, 3, 3


def unit_rms(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).square().mean(dim=1).sqrt()


def test_guidance_matches_covariance_formula_and_has_unit_rms():
    torch.manual_seed(0)
    endpoints = torch.randn(8, C, H, W)
    rewards = torch.randn(8)
    group_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    guidance, active = art_guidance(endpoints, rewards, group_index, 2)

    assert guidance.shape == (2, C, H, W)
    assert active.tolist() == [True, True]
    torch.testing.assert_close(unit_rms(guidance), torch.ones(2))
    for group in range(2):
        rows = group_index == group
        x = endpoints[rows].flatten(1)
        r = rewards[rows]
        cov = ((r - r.mean())[:, None] * (x - x.mean(dim=0))).mean(dim=0)  # paper Eq. 5, 1/K reduction
        expected = cov / cov.square().mean().sqrt()
        torch.testing.assert_close(guidance[group].flatten(), expected, rtol=1e-5, atol=1e-6)


def test_guidance_sign_follows_reward_endpoint_covariance():
    """Endpoints that grow along ``u`` with the reward give guidance pointing along ``u``."""
    u = torch.randn(C, H, W)
    rewards = torch.tensor([-1.0, 0.0, 2.0, 3.0])
    endpoints = rewards[:, None, None, None] * u + 0.3  # affine in the reward: covariance is exactly along u
    guidance, active = art_guidance(endpoints, rewards, torch.zeros(4, dtype=torch.long), 1)
    assert bool(active[0])
    torch.testing.assert_close(guidance[0], u / unit_rms(u[None])[0], rtol=1e-5, atol=1e-6)

    flipped, _ = art_guidance(endpoints, -rewards, torch.zeros(4, dtype=torch.long), 1)
    torch.testing.assert_close(flipped, -guidance)


def test_guidance_is_zero_for_constant_rewards_single_samples_and_zero_covariance():
    endpoints = torch.randn(5, C, H, W)
    rewards = torch.tensor([0.7, 0.7, 0.7, 1.0, 1.0])
    # group 0: constant rewards (fp32 sums of equal values could round, the exact check must catch it);
    # group 1: a single endpoint; group 2: two identical endpoints with different rewards (zero covariance).
    endpoints[4] = endpoints[3]
    group_index = torch.tensor([0, 0, 0, 1, 2])
    rewards = torch.tensor([0.1, 0.1, 0.1, 5.0, 1.0])
    endpoints = torch.cat([endpoints, endpoints[3:4]])
    group_index = torch.cat([group_index, torch.tensor([2])])
    rewards = torch.cat([rewards, torch.tensor([2.0])])

    guidance, active = art_guidance(endpoints, rewards, group_index, 3)

    assert active.tolist() == [False, False, False]
    assert torch.count_nonzero(guidance) == 0


def test_guidance_mixes_active_and_inactive_groups_in_one_batch():
    torch.manual_seed(1)
    endpoints = torch.randn(6, C, H, W)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])
    group_index = torch.tensor([0, 0, 0, 1, 1, 1])
    guidance, active = art_guidance(endpoints, rewards, group_index, 2)
    assert active.tolist() == [True, False]
    torch.testing.assert_close(unit_rms(guidance), torch.tensor([1.0, 0.0]))


def test_guidance_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="one entry per endpoint"):
        art_guidance(torch.randn(4, C, H, W), torch.randn(3), torch.zeros(4, dtype=torch.long), 1)
    with pytest.raises(ValueError, match=r"group_index must lie in"):
        art_guidance(torch.randn(4, C, H, W), torch.randn(4), torch.tensor([0, 0, 1, 2]), 2)


def test_pairwise_labels_are_disjoint_pairs_with_ties_at_half():
    scores = torch.tensor([0.9, 0.1, 0.5, 0.5, 2.0, 1.0])
    group_index = torch.tensor([0, 0, 1, 1, 2, 2])
    labels = art_pairwise_labels(scores, group_index, 3, torch.Generator().manual_seed(0))
    assert labels.tolist() == [1.0, 0.0, 0.5, 0.5, 1.0, 0.0]


def test_pairwise_labels_follow_the_random_pairing_within_a_group():
    scores = torch.tensor([3.0, 1.0, 2.0, 0.0])
    group_index = torch.zeros(4, dtype=torch.long)
    labels = art_pairwise_labels(scores, group_index, 1, torch.Generator().manual_seed(3))
    assert sorted(labels.tolist()) == [0.0, 0.0, 1.0, 1.0]  # two disjoint pairs, one winner each
    assert labels[0] == 1.0  # the top score always wins its pair
    assert labels[3] == 0.0  # the bottom score always loses its pair
    with pytest.raises(ValueError, match="even group size"):
        art_pairwise_labels(scores[:3], group_index[:3], 1)


def test_pairwise_guidance_with_k2_points_from_loser_to_winner():
    """Paper App. C.3: for K = 2 the normalised guidance is the winner-minus-loser direction."""
    endpoints = torch.randn(2, C, H, W)
    labels = art_pairwise_labels(torch.tensor([0.2, 0.9]), torch.zeros(2, dtype=torch.long), 1)
    guidance, active = art_guidance(endpoints, labels, torch.zeros(2, dtype=torch.long), 1)
    diff = endpoints[1] - endpoints[0]
    assert bool(active[0])
    torch.testing.assert_close(guidance[0], diff / unit_rms(diff[None])[0], rtol=1e-5, atol=1e-6)


def test_reference_drift_matches_paper_closed_form():
    """Eq. 7 with App. C.2: m = h^2 (1 + nu^2/2)^2 mean(d^2) / (2 nu^2 t h / (1 - t))."""
    nu = 0.8
    energy = torch.tensor([0.01, 0.02, 0.03])
    sigma = torch.tensor([0.9, 0.5, 1.0])
    sigma_next = torch.tensor([0.8, 0.4, 0.9])
    score = art_reference_drift(energy, sigma, sigma_next, nu)
    h = sigma - sigma_next
    expected_mid = (h * (1 + nu**2 / 2)) ** 2 * energy / (2 * nu**2 * sigma / (1 - sigma) * h)
    torch.testing.assert_close(score[:2], expected_mid[:2])
    # t = 1 uses mu = x + h v and sigma_ref^2 = nu^2.
    torch.testing.assert_close(score[2], h[2] ** 2 * energy[2] / (2 * nu**2))


def make_loss_inputs(batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    old = torch.randn(batch, C, H, W, generator=generator)
    raw = torch.randn(batch, C, H, W, generator=generator)
    guidance = raw / unit_rms(raw)[:, None, None, None]
    sigma = torch.tensor([1.0, 0.7, 0.4, 0.1])[:batch]
    sigma_next = torch.tensor([0.9, 0.6, 0.3, 0.0])[:batch]
    return old, guidance, sigma, sigma_next


def test_loss_at_the_old_policy_equals_sigma_times_gain_squared():
    """Paper Eq. 6: L(0) = w / D ||G||^2, i.e. w * gain^2 for a unit-RMS direction."""
    old, guidance, sigma, sigma_next = make_loss_inputs()
    velocity = old.clone().requires_grad_(True)
    terms = art_loss(velocity, old, guidance, sigma, sigma_next, guidance_gain=1.5)
    torch.testing.assert_close(terms["per_item_loss"], sigma * 1.5**2)
    torch.testing.assert_close(terms["target"], old - 1.5 * guidance)
    assert terms["keep_mask"].all() and not terms["masked"].any()
    # The output-space force is 2 w / D * (target - v): with zero drift it is pure guidance.
    (grad,) = torch.autograd.grad(terms["per_item_loss"].sum(), velocity)
    expected = -2 * sigma[:, None, None, None] * (terms["target"] - old) / (C * H * W)
    torch.testing.assert_close(grad, expected)


def test_loss_second_update_descends_along_guidance_minus_drift():
    """After a first update the descent direction is G - d (paper Eq. 6), up to the velocity sign convention."""
    old, guidance, sigma, sigma_next = make_loss_inputs()
    drift = 0.01 * torch.randn_like(old)
    velocity = (old + drift).requires_grad_(True)
    terms = art_loss(velocity, old, guidance, sigma, sigma_next, loss_space="velocity_uniform", mask_threshold=1e9)
    (grad,) = torch.autograd.grad(terms["per_item_loss"].sum(), velocity)
    # target - v = -(G + d) in the v = noise - x0 convention, so -grad is along -(G + d).
    torch.testing.assert_close(grad, 2 * (guidance + drift) / (C * H * W))
    torch.testing.assert_close(terms["movement_energy"], drift.square().flatten(1).mean(dim=1))


def test_mask_zeroes_large_outward_drift_but_keeps_inward_and_small_moves():
    old, guidance, sigma, sigma_next = make_loss_inputs(batch=3)
    # Item 0: large drift along the descent direction (-G): outward -> masked.
    # Item 1: large drift against the descent direction (+G): inward -> kept.
    # Item 2: tiny outward drift below the threshold -> kept.
    velocity = torch.stack([old[0] - 0.5 * guidance[0], old[1] + 0.5 * guidance[1], old[2] - 1e-4 * guidance[2]])
    velocity.requires_grad_(True)
    terms = art_loss(velocity, old, guidance, sigma, sigma_next, mask_metric="fixed_mse", mask_threshold=1e-5)

    assert terms["masked"].tolist() == [True, False, False]
    assert terms["keep_mask"].tolist() == [False, True, True]
    assert terms["outward_alignment"][0] > 0 and terms["outward_alignment"][1] < 0
    assert terms["per_item_loss"][0] == 0
    (grad,) = torch.autograd.grad(terms["per_item_loss"].sum(), velocity)
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[1]) > 0 and torch.count_nonzero(grad[2]) > 0

    unmasked = art_loss(
        velocity, old, guidance, sigma, sigma_next, mask_metric="fixed_mse", corrective_mask_enabled=False
    )
    assert not unmasked["masked"].any()
    assert unmasked["per_item_loss"][0] > 0


def test_sde_kl_mask_uses_the_reference_drift_proxy():
    old, guidance, sigma, sigma_next = make_loss_inputs(batch=2)
    velocity = old - 0.05 * guidance  # outward
    terms = art_loss(velocity, old, guidance, sigma, sigma_next, mask_metric="sde_kl", mask_threshold=1e-5)
    expected = art_reference_drift(terms["movement_energy"], sigma, sigma_next, 0.8)
    torch.testing.assert_close(terms["movement_score"], expected)
    assert terms["masked"].all()


def test_inactive_items_have_zero_loss_and_zero_gradient():
    old, guidance, sigma, sigma_next = make_loss_inputs()
    guidance[1] = 0.0
    velocity = (old + 0.1).requires_grad_(True)
    terms = art_loss(velocity, old, guidance, sigma, sigma_next)
    assert terms["active"].tolist() == [True, False, True, True]
    assert terms["per_item_loss"][1] == 0
    (grad,) = torch.autograd.grad(terms["per_item_loss"].sum(), velocity)
    assert torch.count_nonzero(grad[1]) == 0


def test_loss_validates_options_and_shapes():
    old, guidance, sigma, sigma_next = make_loss_inputs()
    with pytest.raises(ValueError, match="loss_space"):
        art_loss(old, old, guidance, sigma, sigma_next, loss_space="x0")
    with pytest.raises(ValueError, match="mask_metric"):
        art_loss(old, old, guidance, sigma, sigma_next, mask_metric="kl")
    with pytest.raises(ValueError, match="share one shape"):
        art_loss(old, old, guidance[:2], sigma, sigma_next)
    with pytest.raises(ValueError, match="one value per item"):
        art_loss(old, old, guidance, sigma[:2], sigma_next)
    assert math.isfinite(art_loss(old, old, guidance, sigma, sigma_next)["per_item_loss"].sum().item())


def test_group_index_from_uid_numbers_prompts_by_first_appearance():
    import numpy as np

    from verl_omni.trainer.diffusion.diffusion_art import group_index_from_uid

    group_index, num_groups = group_index_from_uid(np.array(["b", "a", "b", "c", "a"], dtype=object))
    assert group_index.tolist() == [0, 1, 0, 2, 1] and num_groups == 3
    assert group_index.dtype == torch.int64


def test_guidance_damps_near_zero_reward_endpoint_correlation_like_the_reference():
    """Reference ``historical`` direction mode: the covariance is scaled by the group's mean absolute centred reward
    and divided by ``max(rms, eps)``. A group whose scaled covariance has RMS below ``eps`` therefore gets a damped
    push (RMS < 1) instead of a unit-RMS push along what is mostly noise; ordinary groups are unit RMS."""
    from verl_omni.trainer.diffusion.diffusion_art import art_guidance

    generator = torch.Generator().manual_seed(3)
    endpoints = torch.randn(8, 6, generator=generator)
    endpoints[4:] = 1e-7 * torch.randn(
        4, 6, generator=generator
    )  # group 1: endpoints nearly identical -> tiny covariance
    rewards = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.1, 0.9, 0.3, 0.5])
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    guidance, active = art_guidance(endpoints, rewards, groups, 2, eps=1e-5)
    assert active.tolist() == [True, True]
    rms = guidance.flatten(1).square().mean(dim=1).sqrt()
    assert rms[0].item() == pytest.approx(1.0, rel=1e-5)
    assert 0.0 < rms[1].item() < 1.0  # damped, not amplified
    # Reference formula for the damped group: signal = cov / mad, direction = signal / clamp_min(rms(signal), eps).
    centered = rewards[4:] - rewards[4:].mean()
    covariance = (centered[:, None] * endpoints[4:]).mean(dim=0)
    mad = centered.abs().mean()
    signal = covariance / mad
    expected = signal / signal.square().mean().sqrt().clamp_min(1e-5)
    torch.testing.assert_close(guidance[1], expected.float(), rtol=1e-4, atol=1e-6)
    # Direction of the ordinary group is unchanged by the scaling (cov / mad has the same direction as cov).
    cov0 = ((rewards[:4] - rewards[:4].mean())[:, None] * endpoints[:4]).mean(dim=0)
    torch.testing.assert_close(guidance[0], (cov0 / cov0.square().mean().sqrt()).float(), rtol=1e-4, atol=1e-6)


def test_reference_drift_edge_handling_matches_the_reference():
    """``t = 1`` band within ``atol=eps`` of 1, variance floored at ``eps**2``, and a non-forward step is an error."""
    from verl_omni.trainer.diffusion.diffusion_art import art_reference_drift

    energy = torch.full((3,), 0.01)
    nu = 0.8
    # 1 - 2e-6 is inside the eps=1e-5 band -> t=1 form: coefficient = h, variance = nu^2.
    sigma = torch.tensor([1.0, 1.0 - 2e-6, 0.5])
    sigma_next = torch.tensor([0.9, 0.9, 0.4])
    drift = art_reference_drift(energy, sigma, sigma_next, nu)
    h = sigma - sigma_next
    expected_initial = h**2 * energy / (2 * nu**2)
    torch.testing.assert_close(drift[:2], expected_initial[:2])
    interior = (h[2] * (1 + 0.5 * nu**2)) ** 2 * energy[2] / (2 * nu**2 * 0.5 / 0.5 * h[2])
    torch.testing.assert_close(drift[2], interior)
    # Tiny variance is floored at eps**2 (not eps): a low-noise state with a tiny step (fp32 step used for the
    # expectation so the check is about the floor, not about float rounding of the inputs).
    s_tiny, s_tiny_next = torch.tensor([1e-3]), torch.tensor([1e-3 - 1e-7])
    h_tiny = (s_tiny - s_tiny_next).double()
    variance = nu**2 * 1e-3 / (1 - 1e-3) * h_tiny.item()
    assert variance < 1e-10
    tiny = art_reference_drift(torch.tensor([1.0]), s_tiny, s_tiny_next, nu)
    expected_tiny = (h_tiny * (1 + 0.5 * nu**2)) ** 2 / (2 * 1e-10)
    torch.testing.assert_close(tiny.double(), expected_tiny, rtol=1e-4, atol=0)
    with pytest.raises(ValueError, match="sigma > sigma_next"):
        art_reference_drift(energy, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.4, 0.6]), nu)
