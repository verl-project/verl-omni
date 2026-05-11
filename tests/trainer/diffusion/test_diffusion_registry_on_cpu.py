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
    get_diffusion_adv_estimator_fn,
    get_diffusion_loss_fn,
    kl_penalty_image,
    register_diffusion_adv_est,
    register_diffusion_loss,
)

# ---------------------------------------------------------------------------
# kl_penalty_image
# ---------------------------------------------------------------------------


class TestKLPenaltyImage:
    @pytest.mark.parametrize("batch_size,seq_len,channels", [(4, 16, 3), (1, 64, 16), (8, 4, 8)])
    def test_output_is_scalar(self, batch_size, seq_len, channels):
        mean = torch.randn(batch_size, seq_len, channels)
        ref_mean = torch.randn(batch_size, seq_len, channels)
        std_dev_t = torch.rand(batch_size, 1, 1) + 0.1  # strictly positive

        loss = kl_penalty_image(mean, ref_mean, std_dev_t)

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_identical_means_gives_zero(self):
        """When model and reference are identical the KL is 0."""
        mean = torch.randn(4, 16, 3)
        std_dev_t = torch.ones(4, 1, 1)

        loss = kl_penalty_image(mean, mean.clone(), std_dev_t)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_larger_deviation_gives_larger_loss(self):
        torch.manual_seed(0)
        mean = torch.zeros(4, 16, 3)
        small_ref = mean + 0.1
        large_ref = mean + 10.0
        std_dev_t = torch.ones(4, 1, 1)

        loss_small = kl_penalty_image(mean, small_ref, std_dev_t).item()
        loss_large = kl_penalty_image(mean, large_ref, std_dev_t).item()

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

    def test_get_existing_loss_fn(self):
        fn = get_diffusion_loss_fn("flow_grpo")
        assert callable(fn)

    def test_get_unknown_loss_fn_raises(self):
        with self.assertRaises(ValueError):
            get_diffusion_loss_fn("nonexistent_loss")

    def test_register_and_retrieve_custom_fn(self):
        @register_diffusion_loss("test_loss_cpu")
        def _my_loss(old_log_prob, log_prob, advantages, config=None):
            return torch.tensor(0.0), {}

        fn = get_diffusion_loss_fn("test_loss_cpu")
        assert fn is _my_loss

    def test_registered_fn_is_callable_and_returns_correct_types(self):
        import os

        from hydra import compose, initialize_config_dir

        import verl_omni
        from verl.utils.config import omega_conf_to_dataclass
        from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

        config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="dp_diffusion_actor",
                overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=4"],
            )
        actor_cfg: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

        fn = get_diffusion_loss_fn("flow_grpo")
        old_lp = torch.randn(4)
        lp = torch.randn(4)
        adv = torch.randn(4)
        loss, metrics = fn(old_lp, lp, adv, config=actor_cfg)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)


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

    def test_get_existing_estimator_by_string(self):
        fn = get_diffusion_adv_estimator_fn("flow_grpo")
        assert callable(fn)

    def test_get_existing_estimator_by_enum(self):
        fn = get_diffusion_adv_estimator_fn(DiffusionAdvantageEstimator.FLOW_GRPO)
        assert callable(fn)

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
