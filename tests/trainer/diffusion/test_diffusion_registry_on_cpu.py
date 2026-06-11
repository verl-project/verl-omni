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
"""CPU tests for diffusion algorithm registries and KL-penalty utilities."""

import unittest
from enum import Enum

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_algos import (
    DIFFUSION_ADV_ESTIMATOR_REGISTRY,
    DIFFUSION_LOSS_REGISTRY,
    DiffusionAdvantageEstimator,
    DiffusionLossResult,
    KLLoss,
    get_diffusion_adv_estimator_fn,
    get_diffusion_loss_fn,
    register_diffusion_adv_est,
    register_diffusion_loss,
)

# ---------------------------------------------------------------------------
# KLLoss.compute_loss
# ---------------------------------------------------------------------------


class TestKLPenalty:
    def setup_method(self):
        self.kl_loss = KLLoss()

    @pytest.mark.parametrize("batch_size,seq_len,channels", [(4, 16, 3), (1, 64, 16), (8, 4, 8)])
    def test_output_is_scalar(self, batch_size, seq_len, channels):
        mean = torch.randn(batch_size, seq_len, channels)
        ref_mean = torch.randn(batch_size, seq_len, channels)
        std_dev_t = torch.rand(batch_size, 1, 1) + 0.1  # strictly positive

        loss, _ = self.kl_loss.compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=ref_mean,
            std_dev_t=std_dev_t,
        )

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_identical_means_gives_zero(self):
        """When model and reference are identical the KL is 0."""
        mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)

        loss, _ = self.kl_loss.compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=mean.clone(),
            std_dev_t=std_dev_t,
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_larger_deviation_gives_larger_loss(self):
        torch.manual_seed(0)
        mean = torch.zeros(4, 16, 3)
        small_ref = mean + 0.1
        large_ref = mean + 10.0
        std_dev_t = torch.ones(4, 1, 1)

        loss_small, _ = self.kl_loss.compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=small_ref,
            std_dev_t=std_dev_t,
        )
        loss_large, _ = self.kl_loss.compute_loss(
            prev_sample_mean=mean,
            ref_prev_sample_mean=large_ref,
            std_dev_t=std_dev_t,
        )
        loss_small = loss_small.item()
        loss_large = loss_large.item()

        assert loss_large > loss_small


# ---------------------------------------------------------------------------
# register_diffusion_loss / get_diffusion_loss_fn
# ---------------------------------------------------------------------------


class TestDiffusionLossRegistry(unittest.TestCase):
    def setUp(self):
        # Snapshot the registry and restore after each test to avoid state leakage
        self._original = dict(DIFFUSION_LOSS_REGISTRY)

    def tearDown(self):
        DIFFUSION_LOSS_REGISTRY.clear()
        DIFFUSION_LOSS_REGISTRY.update(self._original)

    def test_builtin_flow_grpo_registered(self):
        assert "flow_grpo" in DIFFUSION_LOSS_REGISTRY

    def test_builtin_flow_dppo_registered(self):
        assert "flow_dppo" in DIFFUSION_LOSS_REGISTRY

    def test_builtin_dance_grpo_registered(self):
        assert "dance_grpo" in DIFFUSION_LOSS_REGISTRY

    def test_builtin_kl_registered(self):
        assert "kl" in DIFFUSION_LOSS_REGISTRY

    def test_builtin_dpo_registered(self):
        assert "dpo" in DIFFUSION_LOSS_REGISTRY

    def test_get_existing_loss_fn(self):
        fn = get_diffusion_loss_fn("flow_grpo")
        assert callable(fn)

    def test_get_unknown_loss_fn_raises(self):
        with self.assertRaises(ValueError):
            get_diffusion_loss_fn("nonexistent_loss")

    def test_loss_input_validation_reports_missing_data_key(self):
        fn = get_diffusion_loss_fn("flow_grpo")
        with pytest.raises(KeyError) as exc_info:
            fn.validate_inputs(
                loss_name="flow_grpo",
                model_output={"log_probs": torch.randn(4)},
                data={"old_log_probs": torch.randn(4)},
            )

        message = str(exc_info.value)
        assert "flow_grpo" in message
        assert "Missing data keys" in message
        assert "advantages" in message
        assert "Available data keys" in message
        assert "old_log_probs" in message

    def test_loss_input_validation_reports_missing_model_output_key(self):
        fn = get_diffusion_loss_fn("kl")
        with pytest.raises(KeyError) as exc_info:
            fn.validate_inputs(
                loss_name="kl",
                model_output={"prev_sample_mean": torch.randn(4, 16, 3)},
                data={"ref_prev_sample_mean": torch.randn(4, 16, 3)},
            )

        message = str(exc_info.value)
        assert "kl" in message
        assert "Missing model_output keys" in message
        assert "std_dev_t" in message
        assert "Available model_output keys" in message
        assert "prev_sample_mean" in message

    def test_flow_dppo_loss_input_validation_reports_missing_data_key(self):
        fn = get_diffusion_loss_fn("flow_dppo")
        with pytest.raises(KeyError) as exc_info:
            fn.validate_inputs(
                loss_name="flow_dppo",
                model_output={
                    "log_probs": torch.randn(4),
                    "prev_sample_mean": torch.randn(4, 2, 2, 2),
                    "std_dev_t": torch.ones(4, 1, 1, 1),
                    "sqrt_dt": torch.ones(4),
                },
                data={"old_log_probs": torch.randn(4), "advantages": torch.randn(4)},
            )

        message = str(exc_info.value)
        assert "flow_dppo" in message
        assert "old_prev_sample_mean" in message

    def test_register_and_retrieve_custom_fn(self):
        @register_diffusion_loss("test_loss_cpu")
        class MyLossFunc:
            def __call__(self, *, config, model_output, data):
                del config, model_output, data
                return DiffusionLossResult(loss=torch.tensor(0.0), metrics={})

        fn = get_diffusion_loss_fn("test_loss_cpu")
        assert isinstance(fn, MyLossFunc)

    def test_registered_fn_is_callable_and_returns_correct_types(self):
        import os

        from hydra import compose, initialize_config_dir
        from verl.utils.config import omega_conf_to_dataclass

        import verl_omni
        from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

        config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="dp_diffusion_actor",
                overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=4"],
            )
        actor_cfg: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

        fn = get_diffusion_loss_fn("flow_grpo")
        result = fn(
            config=actor_cfg,
            model_output={"log_probs": torch.randn(4)},
            data={"old_log_probs": torch.randn(4), "advantages": torch.randn(4)},
        )
        assert isinstance(result.loss, torch.Tensor)
        assert isinstance(result.metrics, dict)


class TestKLLoss:
    def test_computes_kl_loss_and_metrics(self):
        class _Config:
            kl_loss_coef = 0.1

        mean = torch.randn(4, 16, 3)
        ref_mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)
        loss_func = get_diffusion_loss_fn("kl")

        result = loss_func(
            config=_Config(),
            model_output={"prev_sample_mean": mean, "std_dev_t": std_dev_t},
            data={"ref_prev_sample_mean": ref_mean},
        )

        assert isinstance(result.loss, torch.Tensor)

        assert "actor/kl_loss" in result.metrics
        assert result.metrics["actor/kl_loss"] == pytest.approx(result.loss.item())


# ---------------------------------------------------------------------------
# register_diffusion_adv_est / get_diffusion_adv_estimator_fn
# ---------------------------------------------------------------------------


class TestDiffusionAdvEstRegistry(unittest.TestCase):
    def setUp(self):
        self._original = dict(DIFFUSION_ADV_ESTIMATOR_REGISTRY)

    def tearDown(self):
        DIFFUSION_ADV_ESTIMATOR_REGISTRY.clear()
        DIFFUSION_ADV_ESTIMATOR_REGISTRY.update(self._original)

    def test_builtin_flow_grpo_registered(self):
        assert DiffusionAdvantageEstimator.FLOW_GRPO.value in DIFFUSION_ADV_ESTIMATOR_REGISTRY

    def test_builtin_flow_dppo_registered(self):
        assert DiffusionAdvantageEstimator.FLOW_DPPO.value in DIFFUSION_ADV_ESTIMATOR_REGISTRY

    def test_builtin_dance_grpo_registered(self):
        assert DiffusionAdvantageEstimator.DANCE_GRPO.value in DIFFUSION_ADV_ESTIMATOR_REGISTRY

    def test_get_existing_estimator_by_string(self):
        fn = get_diffusion_adv_estimator_fn("flow_grpo")
        assert callable(fn)

    def test_get_existing_estimator_by_enum(self):
        fn = get_diffusion_adv_estimator_fn(DiffusionAdvantageEstimator.FLOW_GRPO)
        assert callable(fn)

    def test_flow_dppo_reuses_flow_grpo_estimator(self):
        assert get_diffusion_adv_estimator_fn("flow_dppo") is get_diffusion_adv_estimator_fn("flow_grpo")

    def test_get_unknown_estimator_raises(self):
        with self.assertRaises(ValueError):
            get_diffusion_adv_estimator_fn("nonexistent_estimator")

    def test_register_with_string(self):
        @register_diffusion_adv_est("cpu_test_est")
        def _est(sample_level_rewards, index, **kwargs):
            return sample_level_rewards, sample_level_rewards

        assert get_diffusion_adv_estimator_fn("cpu_test_est") is _est

    def test_register_with_enum(self):
        class _TestEnum(str, Enum):
            MY_EST = "my_est_cpu"

        @register_diffusion_adv_est(_TestEnum.MY_EST)
        def _est(sample_level_rewards, index, **kwargs):
            return sample_level_rewards, sample_level_rewards

        assert get_diffusion_adv_estimator_fn("my_est_cpu") is _est

    def test_duplicate_registration_same_function_is_idempotent(self):
        def _fn(r, i, **kw):
            return r, r

        register_diffusion_adv_est("dup_cpu_est")(_fn)
        register_diffusion_adv_est("dup_cpu_est")(_fn)  # second call must not raise
        assert get_diffusion_adv_estimator_fn("dup_cpu_est") is _fn

    def test_duplicate_registration_different_function_raises(self):
        register_diffusion_adv_est("conflict_cpu_est")(lambda r, i, **kw: (r, r))
        with self.assertRaises(ValueError):
            register_diffusion_adv_est("conflict_cpu_est")(lambda r, i, **kw: (r, r))
