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

"""CPU tests for the BAGEL UniGRPO adapter registration and the UniGRPO image loss.

No model weights are loaded: these exercise the ``(architecture, algorithm)`` registry
wiring, the ``unigrpo`` loss-mode plumbing, and ``UniGRPOLoss`` numerics on tiny-random
tensors (the joint AR + image update itself needs a GPU and a real BAGEL checkpoint).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import verl_omni.pipelines  # noqa: F401  (import side effect: registers the adapters)
from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import BagelDiffusion
from verl_omni.pipelines.bagel_unigrpo.diffusers_training_adapter import BagelUniGRPO
from verl_omni.pipelines.bagel_unigrpo.joint_update import ar_grpo_loss
from verl_omni.pipelines.bagel_unigrpo.rollout import build_unigrpo_pipeline_kwargs
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.diffusion_algos import DIFFUSION_LOSS_REGISTRY, UniGRPOLoss, get_diffusion_loss_fn


def _loss_cfg(*, ratio_norm=True, mse_weight=1.5e-5, clip_ratio=1e-6, adv_clip_max=5.0):
    """Minimal stand-in for the actor config the loss reads (``config.diffusion_loss.*``)."""
    return SimpleNamespace(
        diffusion_loss=SimpleNamespace(
            clip_ratio=clip_ratio, adv_clip_max=adv_clip_max, mse_weight=mse_weight, ratio_norm=ratio_norm
        )
    )


def _tiny_batch(batch=4, latent=(3, 5), *, requires_grad=True):
    """A tiny-random image-track batch: per-sample log-probs + reverse-SDE means + velocities."""
    torch.manual_seed(0)
    log_probs = torch.randn(batch, requires_grad=requires_grad)
    old_log_probs = log_probs.detach().clone()
    prev_sample_mean = torch.randn(batch, *latent, requires_grad=requires_grad)
    old_prev_sample_mean = prev_sample_mean.detach().clone()
    std_dev_t = torch.rand(batch) + 0.1
    sqrt_dt = torch.rand(batch) + 0.1
    velocity = torch.randn(batch, *latent, requires_grad=requires_grad)
    ref_velocity = torch.randn(batch, *latent)
    advantages = torch.randn(batch)
    model_output = {
        "log_probs": log_probs,
        "prev_sample_mean": prev_sample_mean,
        "std_dev_t": std_dev_t,
        "sqrt_dt": sqrt_dt,
        "velocity": velocity,
    }
    data = {
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "old_prev_sample_mean": old_prev_sample_mean,
        "ref_velocity": ref_velocity,
    }
    return model_output, data


def test_bagel_unigrpo_adapter_registered() -> None:
    assert DiffusionModelBase.get_class_by_name("OmniBagelForConditionalGeneration", "unigrpo") is BagelUniGRPO
    # Reuses the flow_grpo image branch (scheduler / CFG / forward_and_sample) by subclassing it.
    assert issubclass(BagelUniGRPO, BagelDiffusion)


def test_unigrpo_loss_mode_registered() -> None:
    assert "unigrpo" in DIFFUSION_LOSS_REGISTRY
    assert isinstance(get_diffusion_loss_fn("unigrpo"), UniGRPOLoss)


def test_unigrpo_loss_config_accepts_unigrpo_mode() -> None:
    from verl_omni.workers.config.diffusion import DiffusionLossConfig

    cfg = DiffusionLossConfig(loss_mode="unigrpo")
    assert cfg.loss_mode == "unigrpo"
    # mse_weight / ratio_norm are the UniGRPO-specific knobs added to the loss config.
    assert hasattr(cfg, "mse_weight") and hasattr(cfg, "ratio_norm")
    with pytest.raises(ValueError):
        DiffusionLossConfig(loss_mode="not_a_real_mode")


def test_fsdp_diffusion_optimizer_config_has_param_group_lrs() -> None:
    from omegaconf import OmegaConf
    from verl.utils.config import omega_conf_to_dataclass

    cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.FSDPDiffusionOptimizerConfig",
            "lr": 1e-6,
            "param_group_lrs": {"moe_gen": 3e-5},
        }
    )
    obj = omega_conf_to_dataclass(cfg)
    assert obj.param_group_lrs == {"moe_gen": 3e-5}
    assert float(obj.lr) == pytest.approx(1e-6)


def test_unigrpo_loss_ratio_norm_backward() -> None:
    model_output, data = _tiny_batch()
    result = UniGRPOLoss()(config=_loss_cfg(ratio_norm=True), model_output=model_output, data=data)
    assert result.loss.ndim == 0 and torch.isfinite(result.loss)
    assert "actor/velocity_mse" in result.metrics and result.metrics["actor/velocity_mse"] > 0.0
    assert {"actor/ppo_kl", "actor/ratio_mean", "actor/pg_clipfrac"} <= set(result.metrics)
    result.loss.backward()
    assert model_output["log_probs"].grad is not None
    assert model_output["velocity"].grad is not None  # the velocity-MSE term reaches the velocity


def test_unigrpo_loss_plain_ratio_when_ratio_norm_off() -> None:
    model_output, data = _tiny_batch()
    result = UniGRPOLoss()(config=_loss_cfg(ratio_norm=False), model_output=model_output, data=data)
    assert torch.isfinite(result.loss)
    # ratio_norm=False falls back to the plain Flow-GRPO ratio, which is >0 for on-policy logp.
    assert result.metrics["actor/velocity_mse"] > 0.0


def test_unigrpo_loss_zero_mse_skips_velocity() -> None:
    model_output, data = _tiny_batch()
    # No velocity / ref_velocity supplied, but mse_weight == 0 so they are not required.
    model_output.pop("velocity")
    data.pop("ref_velocity")
    result = UniGRPOLoss()(config=_loss_cfg(mse_weight=0.0), model_output=model_output, data=data)
    assert result.metrics["actor/velocity_mse"] == 0.0
    assert torch.isfinite(result.loss)


def test_unigrpo_loss_requires_velocity_when_mse_positive() -> None:
    model_output, data = _tiny_batch()
    model_output.pop("velocity")
    data.pop("ref_velocity")
    with pytest.raises(KeyError):
        UniGRPOLoss()(config=_loss_cfg(mse_weight=1.5e-5), model_output=model_output, data=data)


def test_ar_grpo_loss_on_policy_ratio_is_one() -> None:
    torch.manual_seed(0)
    logp = torch.randn(6, requires_grad=True)
    old = logp.detach().clone()
    adv = torch.randn(6)
    loss, metrics = ar_grpo_loss(logp, old, adv, clip_range=1e-2)
    assert torch.isfinite(loss)
    assert metrics["ar/ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["ar/ppo_kl"] == pytest.approx(0.0, abs=1e-5)


def test_build_unigrpo_pipeline_kwargs_reads_model_config() -> None:
    model_config = SimpleNamespace(
        pipeline=SimpleNamespace(height=512, width=512, num_inference_steps=25),
        algo=SimpleNamespace(noise_level=0.8, sde_window_size=3),
    )
    kwargs = build_unigrpo_pipeline_kwargs(model_config)
    assert kwargs["num_inference_steps"] == 25
    assert kwargs["eta"] == pytest.approx(0.8)
    assert kwargs["num_sde_steps"] == 3
    assert "stop_token_ids" not in kwargs  # no module -> no EOS derived
