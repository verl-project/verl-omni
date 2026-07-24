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
"""CPU tests for teacher-anchored continuous distillation losses (OPD, #293)."""

import os

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_algos import (
    DIFFUSION_LOSS_REGISTRY,
    DiffusionLossResult,
    DistillFlowMatchingMSELoss,
    DistillKLLoss,
    get_diffusion_loss_fn,
)
from verl_omni.workers.config.diffusion import DiffusionLossConfig

# ---------------------------------------------------------------------------
# DistillKLLoss.compute_loss
# ---------------------------------------------------------------------------


class TestDistillKLLoss:
    def setup_method(self):
        self.loss_fn = DistillKLLoss()

    def test_identical_means_gives_zero(self):
        mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)

        loss, _ = self.loss_fn.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=mean.clone(),
            std_dev_t=std_dev_t,
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_matches_gaussian_kl_closed_form(self):
        """Constant deviation d with unit variance gives KL = d^2 / 2."""
        mean = torch.zeros(4, 16, 3)
        teacher_mean = mean + 2.0
        std_dev_t = torch.ones(4, 1, 1)

        loss, _ = self.loss_fn.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=teacher_mean,
            std_dev_t=std_dev_t,
        )

        assert loss.item() == pytest.approx(2.0, rel=1e-5)

    def test_larger_deviation_gives_larger_loss(self):
        mean = torch.zeros(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)

        loss_small, _ = self.loss_fn.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=mean + 0.1,
            std_dev_t=std_dev_t,
        )
        loss_large, _ = self.loss_fn.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=mean + 10.0,
            std_dev_t=std_dev_t,
        )

        assert loss_large.item() > loss_small.item()

    def test_output_is_scalar_and_nonnegative(self):
        loss, _ = self.loss_fn.compute_loss(
            prev_sample_mean=torch.randn(2, 8, 4),
            teacher_prev_sample_mean=torch.randn(2, 8, 4),
            std_dev_t=torch.rand(2, 1, 1) + 0.1,
        )

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_call_returns_result_with_metrics(self):
        mean = torch.randn(4, 16, 3)
        teacher_mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)

        result = self.loss_fn(
            config=None,
            model_output={"prev_sample_mean": mean, "std_dev_t": std_dev_t},
            data={"teacher_prev_sample_mean": teacher_mean},
        )

        assert isinstance(result, DiffusionLossResult)
        assert "actor/distill_kl_loss" in result.metrics
        assert result.metrics["actor/distill_kl_loss"] == pytest.approx(result.loss.item())

    def test_validate_inputs_reports_missing_teacher_key(self):
        with pytest.raises(KeyError) as exc_info:
            self.loss_fn.validate_inputs(
                loss_name="distill_kl",
                model_output={"prev_sample_mean": torch.randn(4, 16, 3), "std_dev_t": torch.ones(4, 1, 1)},
                data={"old_log_probs": torch.randn(4)},
            )

        message = str(exc_info.value)
        assert "distill_kl" in message
        assert "teacher_prev_sample_mean" in message


# ---------------------------------------------------------------------------
# DistillFlowMatchingMSELoss.compute_loss
# ---------------------------------------------------------------------------


class TestDistillFlowMatchingMSELoss:
    def setup_method(self):
        self.loss_fn = DistillFlowMatchingMSELoss()

    def test_identical_predictions_give_zero(self):
        pred = torch.randn(4, 16, 3)

        loss, _ = self.loss_fn.compute_loss(
            noise_pred=pred,
            teacher_noise_pred=pred.clone(),
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_matches_elementwise_mse(self):
        """Constant deviation d gives MSE = d^2."""
        pred = torch.zeros(4, 16, 3)
        teacher_pred = pred + 3.0

        loss, _ = self.loss_fn.compute_loss(
            noise_pred=pred,
            teacher_noise_pred=teacher_pred,
        )

        assert loss.item() == pytest.approx(9.0, rel=1e-5)

    @pytest.mark.parametrize("shape", [(4, 16, 3), (2, 4, 8, 8), (1, 16, 3, 32, 32)])
    def test_supports_image_and_video_shapes(self, shape):
        pred = torch.randn(*shape)
        teacher_pred = torch.randn(*shape)

        loss, _ = self.loss_fn.compute_loss(
            noise_pred=pred,
            teacher_noise_pred=teacher_pred,
        )

        expected = torch.nn.functional.mse_loss(pred, teacher_pred)
        assert loss.shape == ()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_call_returns_result_with_metrics(self):
        pred = torch.randn(4, 16, 3)
        teacher_pred = torch.randn(4, 16, 3)

        result = self.loss_fn(
            config=None,
            model_output={"noise_pred": pred},
            data={"teacher_noise_pred": teacher_pred},
        )

        assert isinstance(result, DiffusionLossResult)
        assert "actor/distill_fm_mse_loss" in result.metrics
        assert result.metrics["actor/distill_fm_mse_loss"] == pytest.approx(result.loss.item())

    def test_validate_inputs_reports_missing_teacher_key(self):
        with pytest.raises(KeyError) as exc_info:
            self.loss_fn.validate_inputs(
                loss_name="distill_fm_mse",
                model_output={"noise_pred": torch.randn(4, 16, 3)},
                data={},
            )

        message = str(exc_info.value)
        assert "distill_fm_mse" in message
        assert "teacher_noise_pred" in message


# ---------------------------------------------------------------------------
# Registry and config plumbing
# ---------------------------------------------------------------------------


class TestDistillationRegistryAndConfig:
    def test_distill_kl_registered(self):
        assert "distill_kl" in DIFFUSION_LOSS_REGISTRY
        assert isinstance(get_diffusion_loss_fn("distill_kl"), DistillKLLoss)

    def test_distill_fm_mse_registered(self):
        assert "distill_fm_mse" in DIFFUSION_LOSS_REGISTRY
        assert isinstance(get_diffusion_loss_fn("distill_fm_mse"), DistillFlowMatchingMSELoss)

    @pytest.mark.parametrize("loss_mode", ["distill_kl", "distill_fm_mse"])
    def test_diffusion_loss_config_accepts_distill_modes(self, loss_mode):
        cfg = DiffusionLossConfig(loss_mode=loss_mode)
        assert cfg.loss_mode == loss_mode

    def test_diffusion_loss_config_still_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            DiffusionLossConfig(loss_mode="nonexistent_distill_loss")


# ---------------------------------------------------------------------------
# Auxiliary distillation term in the worker-side diffusion_loss entry point
# ---------------------------------------------------------------------------


def _compose_actor_config(overrides):
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    import verl_omni

    config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=4", *overrides],
        )
    return omega_conf_to_dataclass(cfg)


class TestAuxiliaryDistillLoss:
    def _build_batch(self):
        from verl.utils import tensordict_utils as tu

        torch.manual_seed(0)
        model_output = {
            "log_probs": torch.randn(4),
            "prev_sample_mean": torch.randn(4, 16, 3),
            "std_dev_t": torch.ones(4, 1, 1),
        }
        data = tu.get_tensordict(
            {
                "old_log_probs": torch.randn(4),
                "advantages": torch.randn(4),
                "teacher_prev_sample_mean": torch.randn(4, 16, 3),
            }
        )
        tu.assign_non_tensor(data, gradient_accumulation_steps=1, sp_size=1)
        return model_output, data

    def test_use_distill_loss_adds_weighted_term(self):
        from verl_omni.workers.utils.losses import diffusion_loss

        coef = 0.5
        actor_cfg = _compose_actor_config(
            overrides=["use_distill_loss=true", f"distill_loss_coef={coef}", "distill_loss_mode=distill_kl"]
        )
        model_output, data = self._build_batch()

        total_loss, metrics = diffusion_loss(actor_cfg, model_output, data)

        pg_loss, _ = get_diffusion_loss_fn("flow_grpo").compute_loss(
            old_log_prob=data["old_log_probs"],
            log_prob=model_output["log_probs"],
            advantages=data["advantages"],
            config=actor_cfg,
        )
        distill_loss_value, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=model_output["prev_sample_mean"],
            teacher_prev_sample_mean=data["teacher_prev_sample_mean"],
            std_dev_t=model_output["std_dev_t"],
        )

        expected = (pg_loss + coef * distill_loss_value).item()
        assert total_loss.item() == pytest.approx(expected, rel=1e-5)
        assert "actor/distill_kl_loss" in metrics
        assert "distill_coef" in metrics

    def test_distill_disabled_by_default(self):
        from verl_omni.workers.utils.losses import diffusion_loss

        actor_cfg = _compose_actor_config(overrides=[])
        assert actor_cfg.use_distill_loss is False

        model_output, data = self._build_batch()
        total_loss, metrics = diffusion_loss(actor_cfg, model_output, data)

        pg_loss, _ = get_diffusion_loss_fn("flow_grpo").compute_loss(
            old_log_prob=data["old_log_probs"],
            log_prob=model_output["log_probs"],
            advantages=data["advantages"],
            config=actor_cfg,
        )

        assert total_loss.item() == pytest.approx(pg_loss.item(), rel=1e-5)
        assert "actor/distill_kl_loss" not in metrics
