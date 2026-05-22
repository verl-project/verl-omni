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
"""CPU tests for every registered diffusion loss function.

Necessity: Worker-side loss functions are the mathematical core of diffusion RL
and DPO training. Each registered loss must be exercised on CPU so regressions
in tensor contracts, metrics, and validation guards are caught without GPUs.
"""

import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.trainer.diffusion.diffusion_algos import (
    DIFFUSION_LOSS_REGISTRY,
    DPOLoss,
    DiffusionLossResult,
    FlowGRPOLoss,
    GRPOGuardLoss,
    KLLoss,
    get_diffusion_loss_fn,
)
from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

BUILTIN_LOSS_NAMES = ("flow_grpo", "grpo_guard", "dpo", "kl")

FLOW_GRPO_METRIC_KEYS = (
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_higher",
    "actor/pg_clipfrac_lower",
    "actor/ratio_mean",
    "actor/ratio_std",
)


def _actor_config(*overrides: str) -> FSDPDiffusionActorConfig:
    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "ppo_micro_batch_size_per_gpu=4",
                *overrides,
            ],
        )
    return omega_conf_to_dataclass(cfg)


def _dpo_tensors(batch_pairs: int = 2, latent_shape: tuple[int, ...] = (4, 8, 8)):
    batch_size = batch_pairs * 2
    noise = torch.randn(batch_size, *latent_shape)
    latent = torch.randn(batch_size, *latent_shape)
    model_noise_pred = torch.randn(batch_size, *latent_shape)
    ref_noise_pred = torch.randn(batch_size, *latent_shape)
    sample_level_rewards = torch.tensor([1.0, 0.0] * batch_pairs, dtype=torch.float32)
    return noise, latent, model_noise_pred, ref_noise_pred, sample_level_rewards


class TestDiffusionLossRegistry:
    @pytest.mark.parametrize("loss_name", BUILTIN_LOSS_NAMES)
    def test_builtin_loss_registered(self, loss_name: str):
        assert loss_name in DIFFUSION_LOSS_REGISTRY
        fn = get_diffusion_loss_fn(loss_name)
        assert callable(fn)

    def test_registry_matches_builtin_list(self):
        for name in BUILTIN_LOSS_NAMES:
            assert name in DIFFUSION_LOSS_REGISTRY


class TestFlowGRPOLoss:
    def test_compute_loss_returns_scalar_and_metrics(self):
        actor_config = _actor_config(
            "diffusion_loss.clip_ratio=0.0001",
            "diffusion_loss.adv_clip_max=5.0",
        )
        batch_size = 8
        loss_fn = get_diffusion_loss_fn("flow_grpo")
        loss, metrics = loss_fn.compute_loss(
            old_log_prob=torch.randn(batch_size),
            log_prob=torch.randn(batch_size),
            advantages=torch.randn(batch_size),
            config=actor_config,
        )
        assert loss.shape == ()
        assert isinstance(loss.item(), float)
        assert set(metrics.keys()) >= set(FLOW_GRPO_METRIC_KEYS)

    def test_callable_returns_diffusion_loss_result(self):
        actor_config = _actor_config()
        loss_fn = get_diffusion_loss_fn("flow_grpo")
        result = loss_fn(
            config=actor_config,
            model_output={"log_probs": torch.randn(4)},
            data={"old_log_probs": torch.randn(4), "advantages": torch.randn(4)},
        )
        assert isinstance(result, DiffusionLossResult)
        assert isinstance(result.loss, torch.Tensor)
        assert result.loss.shape == ()
        assert isinstance(result.metrics, dict)

    def test_validate_inputs_requires_advantages(self):
        loss_fn = get_diffusion_loss_fn("flow_grpo")
        with pytest.raises(KeyError, match="advantages"):
            loss_fn.validate_inputs(
                loss_name="flow_grpo",
                model_output={"log_probs": torch.randn(4)},
                data={"old_log_probs": torch.randn(4)},
            )


class TestGRPOGuardLoss:
    def test_compute_loss_returns_scalar_and_metrics(self):
        actor_config = _actor_config(
            "diffusion_loss.loss_mode=grpo_guard",
            "diffusion_loss.clip_ratio=2e-6",
            "diffusion_loss.adv_clip_max=5.0",
        )
        batch_size = 4
        old_prev_sample_mean = torch.randn(batch_size, 16, 8, 8)
        loss_fn = get_diffusion_loss_fn("grpo_guard")
        loss, metrics = loss_fn.compute_loss(
            old_log_prob=torch.randn(batch_size),
            log_prob=torch.randn(batch_size),
            advantages=torch.randn(batch_size),
            config=actor_config,
            old_prev_sample_mean=old_prev_sample_mean,
            prev_sample_mean=old_prev_sample_mean + 0.01 * torch.randn_like(old_prev_sample_mean),
            std_dev_t=torch.full((batch_size, 1, 1, 1), 0.5),
            sqrt_dt=torch.full((batch_size,), 0.3),
        )
        assert loss.shape == ()
        assert isinstance(loss.item(), float)
        for key in FLOW_GRPO_METRIC_KEYS:
            assert key in metrics

    def test_callable_returns_diffusion_loss_result(self):
        actor_config = _actor_config("diffusion_loss.loss_mode=grpo_guard")
        batch_size = 4
        old_prev_sample_mean = torch.randn(batch_size, 8, 4, 4)
        loss_fn = get_diffusion_loss_fn("grpo_guard")
        result = loss_fn(
            config=actor_config,
            model_output={
                "log_probs": torch.randn(batch_size),
                "prev_sample_mean": old_prev_sample_mean,
                "std_dev_t": torch.full((batch_size, 1, 1, 1), 0.5),
                "sqrt_dt": torch.full((batch_size,), 0.3),
            },
            data={
                "old_log_probs": torch.randn(batch_size),
                "advantages": torch.randn(batch_size),
                "old_prev_sample_mean": old_prev_sample_mean,
            },
        )
        assert isinstance(result, DiffusionLossResult)
        assert result.loss.shape == ()


class TestDPOLoss:
    def test_registered_instance_type(self):
        assert isinstance(get_diffusion_loss_fn("dpo"), DPOLoss)

    def test_compute_loss_returns_scalar_and_metrics(self):
        actor_config = _actor_config(
            "diffusion_loss.loss_mode=dpo",
            "diffusion_loss.dpo_beta=100.0",
        )
        noise, latent, model_noise_pred, ref_noise_pred, rewards = _dpo_tensors()
        uid = np.array(["pair-0", "pair-0", "pair-1", "pair-1"], dtype=object)

        loss, metrics = DPOLoss.compute_loss(
            noise=noise,
            latent=latent,
            model_noise_pred=model_noise_pred,
            ref_noise_pred=ref_noise_pred,
            sample_level_rewards=rewards,
            config=actor_config,
            index=uid,
        )

        assert loss.shape == ()
        assert isinstance(loss.item(), float)
        assert "actor/dpo_loss" in metrics
        assert "actor/implicit_acc" in metrics
        assert 0.0 <= metrics["actor/implicit_acc"] <= 1.0

    def test_callable_returns_diffusion_loss_result(self):
        actor_config = _actor_config(
            "diffusion_loss.loss_mode=dpo",
            "diffusion_loss.dpo_beta=100.0",
        )
        noise, latent, model_noise_pred, ref_noise_pred, rewards = _dpo_tensors(batch_pairs=1)
        loss_fn = get_diffusion_loss_fn("dpo")
        data = TensorDict(
            {
                "ref_noise_pred": ref_noise_pred,
                "sample_level_rewards": rewards,
            },
            batch_size=2,
        )
        result = loss_fn(
            config=actor_config,
            model_output={"noise": noise, "latent": latent, "noise_pred": model_noise_pred},
            data=data,
        )
        assert isinstance(result, DiffusionLossResult)
        assert result.loss.shape == ()

    def test_rejects_odd_batch_size(self):
        actor_config = _actor_config("diffusion_loss.loss_mode=dpo")
        noise, latent, model_noise_pred, ref_noise_pred, rewards = _dpo_tensors(batch_pairs=1)
        with pytest.raises(ValueError, match="even batch"):
            DPOLoss.compute_loss(
                noise=noise[:1],
                latent=latent[:1],
                model_noise_pred=model_noise_pred[:1],
                ref_noise_pred=ref_noise_pred[:1],
                sample_level_rewards=rewards[:1],
                config=actor_config,
            )

    def test_rejects_mismatched_prompt_uids(self):
        actor_config = _actor_config("diffusion_loss.loss_mode=dpo")
        noise, latent, model_noise_pred, ref_noise_pred, rewards = _dpo_tensors()
        uid = np.array(["pair-0", "pair-1", "pair-0", "pair-1"], dtype=object)
        with pytest.raises(ValueError, match="same prompt uid"):
            DPOLoss.compute_loss(
                noise=noise,
                latent=latent,
                model_noise_pred=model_noise_pred,
                ref_noise_pred=ref_noise_pred,
                sample_level_rewards=rewards,
                config=actor_config,
                index=uid,
            )

    def test_rejects_chosen_reward_below_rejected(self):
        actor_config = _actor_config("diffusion_loss.loss_mode=dpo")
        noise, latent, model_noise_pred, ref_noise_pred, _ = _dpo_tensors()
        rewards = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="chosen sample reward"):
            DPOLoss.compute_loss(
                noise=noise,
                latent=latent,
                model_noise_pred=model_noise_pred,
                ref_noise_pred=ref_noise_pred,
                sample_level_rewards=rewards,
                config=actor_config,
            )


class TestKLLoss:
    def test_compute_loss_returns_scalar_and_metrics(self):
        batch_size = 4
        mean = torch.randn(batch_size, 16, 3)
        ref_mean = torch.randn(batch_size, 16, 3)
        std_dev_t = torch.rand(batch_size, 1, 1) + 0.1
        loss, metrics = KLLoss.compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=ref_mean,
            std_dev_t=std_dev_t,
        )
        assert loss.shape == ()
        assert loss.item() >= 0.0
        assert "actor/kl_loss" in metrics

    def test_identical_means_gives_zero(self):
        mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)
        loss, _ = KLLoss.compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=mean.clone(),
            std_dev_t=std_dev_t,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_callable_returns_diffusion_loss_result(self):
        mean = torch.randn(4, 16, 3)
        ref_mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)
        loss_fn = get_diffusion_loss_fn("kl")
        result = loss_fn(
            config=object(),
            model_output={"prev_sample_mean": mean, "std_dev_t": std_dev_t},
            data={"ref_prev_sample_mean": ref_mean},
        )
        assert isinstance(result, DiffusionLossResult)
        assert result.loss.shape == ()
        assert result.metrics["actor/kl_loss"] == pytest.approx(result.loss.item())

    def test_validate_inputs_requires_std_dev_t(self):
        loss_fn = get_diffusion_loss_fn("kl")
        with pytest.raises(KeyError, match="std_dev_t"):
            loss_fn.validate_inputs(
                loss_name="kl",
                model_output={"prev_sample_mean": torch.randn(4, 16, 3)},
                data={"ref_prev_sample_mean": torch.randn(4, 16, 3)},
            )


@pytest.mark.parametrize(
    ("loss_name", "loss_cls"),
    [
        ("flow_grpo", FlowGRPOLoss),
        ("grpo_guard", GRPOGuardLoss),
        ("dpo", DPOLoss),
        ("kl", KLLoss),
    ],
)
def test_registered_loss_class_types(loss_name: str, loss_cls: type) -> None:
    fn = get_diffusion_loss_fn(loss_name)
    assert isinstance(fn, loss_cls)
