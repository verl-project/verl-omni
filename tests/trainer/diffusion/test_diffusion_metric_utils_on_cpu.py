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
"""CPU tests for verl_omni.trainer.diffusion.diffusion_metric_utils."""

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
STEPS = 10


def _make_batch(batch_size: int = BATCH_SIZE, steps: int = STEPS, include_uid: bool = False) -> DataProto:
    tensors = {
        "sample_level_rewards": torch.randn(batch_size, steps),
        "advantages": torch.randn(batch_size, steps),
        "returns": torch.randn(batch_size, steps),
    }
    non_tensors = {}
    if include_uid:
        # 2 images per unique prompt  →  batch_size / 2 unique uids
        non_tensors["uid"] = np.array([f"uid-{i // 2}" for i in range(batch_size)], dtype=object)
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


# ---------------------------------------------------------------------------
# compute_data_metrics_diffusion
# ---------------------------------------------------------------------------


class TestComputeDataMetricsDiffusion:
    BASE_KEYS = {
        "critic/rewards/mean",
        "critic/rewards/max",
        "critic/rewards/min",
        "critic/advantages/mean",
        "critic/advantages/max",
        "critic/advantages/min",
        "critic/returns/mean",
        "critic/returns/max",
        "critic/returns/min",
    }
    UID_KEYS = {
        "critic/rewards/zero_std_ratio",
        "critic/rewards/std_mean",
        "critic/rewards/group_size",
    }

    def test_returns_all_base_keys(self):
        batch = _make_batch(include_uid=False)
        metrics = compute_data_metrics_diffusion(batch)
        assert self.BASE_KEYS.issubset(set(metrics.keys()))

    def test_no_uid_keys_without_uid(self):
        batch = _make_batch(include_uid=False)
        metrics = compute_data_metrics_diffusion(batch)
        for key in self.UID_KEYS:
            assert key not in metrics

    def test_uid_keys_present_with_uid(self):
        batch = _make_batch(include_uid=True)
        metrics = compute_data_metrics_diffusion(batch)
        assert self.BASE_KEYS | self.UID_KEYS == set(metrics.keys())

    def test_all_values_are_float(self):
        batch = _make_batch(include_uid=True)
        metrics = compute_data_metrics_diffusion(batch)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_reward_ordering(self):
        batch = _make_batch(include_uid=False)
        metrics = compute_data_metrics_diffusion(batch)
        assert metrics["critic/rewards/min"] <= metrics["critic/rewards/mean"] <= metrics["critic/rewards/max"]

    def test_advantages_ordering(self):
        batch = _make_batch(include_uid=False)
        metrics = compute_data_metrics_diffusion(batch)
        assert metrics["critic/advantages/min"] <= metrics["critic/advantages/mean"] <= metrics["critic/advantages/max"]

    def test_returns_ordering(self):
        batch = _make_batch(include_uid=False)
        metrics = compute_data_metrics_diffusion(batch)
        assert metrics["critic/returns/min"] <= metrics["critic/returns/mean"] <= metrics["critic/returns/max"]

    def test_group_size_correct(self):
        """With 2 images per unique prompt, group_size should equal 2."""
        batch = _make_batch(batch_size=8, include_uid=True)
        metrics = compute_data_metrics_diffusion(batch)
        assert metrics["critic/rewards/group_size"] == pytest.approx(2.0)

    def test_zero_std_ratio_all_identical(self):
        """When all images for a prompt have the same reward, zero_std_ratio should be 1.0."""
        batch_size = 4
        steps = 5
        # All timestep rewards identical within each pair
        rewards = torch.zeros(batch_size, steps)
        tensors = {
            "sample_level_rewards": rewards,
            "advantages": torch.randn(batch_size, steps),
            "returns": torch.randn(batch_size, steps),
        }
        non_tensors = {"uid": np.array(["uid-0", "uid-0", "uid-1", "uid-1"], dtype=object)}
        batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        metrics = compute_data_metrics_diffusion(batch)
        assert metrics["critic/rewards/zero_std_ratio"] == pytest.approx(1.0)

    def test_single_sample(self):
        """Edge case: batch of 1 should not raise."""
        batch = _make_batch(batch_size=1, steps=4, include_uid=False)
        metrics = compute_data_metrics_diffusion(batch)
        assert "critic/rewards/mean" in metrics

    def test_dpo_batch_without_advantages_or_returns(self):
        """Direct-preference batches only carry scalar sample_level_rewards."""
        batch_size = 4
        tensors = {"sample_level_rewards": torch.tensor([1.0, 0.0, 1.0, 0.0])}
        non_tensors = {"uid": np.array(["p0", "p0", "p1", "p1"], dtype=object)}
        batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        metrics = compute_data_metrics_diffusion(batch)
        assert "critic/rewards/mean" in metrics
        assert "critic/advantages/mean" not in metrics
        assert "critic/returns/mean" not in metrics


# ---------------------------------------------------------------------------
# compute_timing_metrics_diffusion
# ---------------------------------------------------------------------------


class TestComputeTimingMetricsDiffusion:
    def test_timing_s_keys_present(self):
        timing_raw = {"gen": 1.0, "ref": 0.5, "update_actor": 2.0}
        metrics = compute_timing_metrics_diffusion(timing_raw, num_images=16)
        for name in timing_raw:
            assert f"timing_s/{name}" in metrics

    def test_per_image_keys_for_compute_stages(self):
        timing_raw = {"gen": 1.0, "ref": 0.5, "old_log_prob": 0.3, "adv": 0.2, "update_actor": 2.0}
        metrics = compute_timing_metrics_diffusion(timing_raw, num_images=16)
        for name in timing_raw:
            assert f"timing_per_image_ms/{name}" in metrics

    def test_non_compute_stages_excluded_from_per_image(self):
        timing_raw = {"save_checkpoint": 3.0, "update_weights": 0.1, "testing": 0.5}
        metrics = compute_timing_metrics_diffusion(timing_raw, num_images=16)
        for name in timing_raw:
            assert f"timing_per_image_ms/{name}" not in metrics

    def test_per_image_value_calculation(self):
        timing_raw = {"gen": 2.0}
        num_images = 10
        metrics = compute_timing_metrics_diffusion(timing_raw, num_images=num_images)
        expected = 2.0 * 1000 / num_images
        assert metrics["timing_per_image_ms/gen"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_throughput_metrics_diffusion
# ---------------------------------------------------------------------------


class TestComputeThroughputMetricsDiffusion:
    def test_returns_expected_keys(self):
        batch = _make_batch()
        timing_raw = {"step": 4.0}
        metrics = compute_throughput_metrics_diffusion(batch, timing_raw, n_gpus=2)
        assert {"perf/total_num_images", "perf/time_per_step", "perf/throughput"} == set(metrics.keys())

    def test_total_num_images_equals_batch_size(self):
        batch = _make_batch(batch_size=12)
        timing_raw = {"step": 1.0}
        metrics = compute_throughput_metrics_diffusion(batch, timing_raw, n_gpus=1)
        assert metrics["perf/total_num_images"] == 12

    def test_throughput_uses_rewards_when_advantages_missing(self):
        tensors = {"sample_level_rewards": torch.zeros(6)}
        batch = DataProto.from_dict(tensors=tensors, non_tensors={})
        metrics = compute_throughput_metrics_diffusion(batch, {"step": 2.0}, n_gpus=1)
        assert metrics["perf/total_num_images"] == 6

    def test_throughput_value(self):
        batch_size = 8
        n_gpus = 2
        step_time = 4.0
        batch = _make_batch(batch_size=batch_size)
        metrics = compute_throughput_metrics_diffusion(batch, {"step": step_time}, n_gpus=n_gpus)
        assert metrics["perf/throughput"] == pytest.approx(batch_size / (step_time * n_gpus))
