"""FlowGRPO telemetry must stay finite for single-image microbatches."""

import math
from types import SimpleNamespace

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss


@pytest.mark.parametrize("batch_size", [1, 2])
def test_ratio_population_std_and_gradient(batch_size):
    log_prob = torch.linspace(0.01, 0.02, batch_size, requires_grad=True)
    config = SimpleNamespace(diffusion_loss=SimpleNamespace(adv_clip_max=5.0, clip_ratio=0.2))
    loss, metrics = FlowGRPOLoss.compute_loss(
        old_log_prob=torch.zeros(batch_size), log_prob=log_prob, advantages=torch.ones(batch_size), config=config
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["actor/ratio_std"] == pytest.approx(log_prob.exp().std(unbiased=False).item())
    loss.backward()
    assert torch.isfinite(log_prob.grad).all()
    assert torch.count_nonzero(log_prob.grad) == batch_size
