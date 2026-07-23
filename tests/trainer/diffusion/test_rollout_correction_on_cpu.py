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
"""CPU tests for diffusion rollout correction (bypass and decoupled modes)."""

from __future__ import annotations

import os

import pytest
import torch
from verl import DataProto
from verl.trainer.config.algorithm import RolloutCorrectionConfig

from verl_omni.trainer.diffusion import diffusion_algos, rollout_correction
from verl_omni.workers.utils.losses import _apply_bypass_rc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rc_cfg(
    *,
    bypass_mode: bool = True,
    rollout_is: str | None = "sequence",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
    rollout_rs: str | None = None,
    rollout_rs_threshold: str | None = None,
) -> RolloutCorrectionConfig:
    """Build a RolloutCorrectionConfig for _apply_bypass_rc (attribute access)."""
    return RolloutCorrectionConfig(
        bypass_mode=bypass_mode,
        loss_type="ppo_clip",
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )


def _make_rc_dict(
    *,
    rollout_is: str | None = "sequence",
    rollout_is_threshold: float | str = 2.0,
    rollout_is_batch_normalize: bool = False,
    rollout_rs: str | None = None,
    rollout_rs_threshold: str | None = None,
) -> dict:
    """Build a dict config for apply_rollout_correction_to_diffusion_batch (uses .get())."""
    return {
        "rollout_is": rollout_is,
        "rollout_is_threshold": rollout_is_threshold,
        "rollout_is_batch_normalize": rollout_is_batch_normalize,
        "rollout_rs": rollout_rs,
        "rollout_rs_threshold": rollout_rs_threshold,
    }


# ---------------------------------------------------------------------------
# Bypass mode (_apply_bypass_rc)
# ---------------------------------------------------------------------------


class TestBypassRC:
    """Tests for _apply_bypass_rc – the per-step IS/RS in bypass mode."""

    def test_is_metrics_only_no_weights_stashed(self):
        """In ppo_clip bypass mode, IS weights are NOT applied to loss."""
        batch_size = 8
        log_prob = torch.randn(batch_size)
        rollout_log_prob = log_prob + 0.01 * torch.randn(batch_size)

        rc_cfg = _make_rc_cfg(rollout_is="sequence", rollout_rs=None)
        data = {"old_log_probs": rollout_log_prob.clone(), "advantages": torch.randn(batch_size)}
        metrics = {}

        _apply_bypass_rc(log_prob, rollout_log_prob, rc_cfg, data, metrics)

        # IS metrics should be logged
        assert "rollout_corr/rollout_is_mean" in metrics
        # RS metrics should NOT be logged (RS is off)
        assert "rollout_corr/rollout_rs_masked_fraction" not in metrics
        # No rollout_is_weights should be stashed (ppo_clip skips IS)
        assert "rollout_is_weights" not in data

    def test_rs_rejection_zeros_weights(self):
        """RS rejection should stash a binary mask into rollout_is_weights."""
        batch_size = 16
        log_prob = torch.randn(batch_size)
        # Create large drift for some samples to trigger RS rejection
        rollout_log_prob = log_prob.clone()
        rollout_log_prob[:4] = log_prob[:4] + 5.0  # large positive drift
        rollout_log_prob[4:8] = log_prob[4:8] - 5.0  # large negative drift

        rc_cfg = _make_rc_cfg(
            rollout_is=None,  # IS off — pure RS
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold="0.5_2.0",
        )
        data = {"old_log_probs": rollout_log_prob.clone(), "advantages": torch.randn(batch_size)}
        metrics = {}

        _apply_bypass_rc(log_prob, rollout_log_prob, rc_cfg, data, metrics)

        assert "rollout_is_weights" in data
        weights = data["rollout_is_weights"]
        assert weights.shape == (batch_size,)
        assert weights.dtype == log_prob.dtype

        # Rejected samples should have weight 0
        rejected_mask = weights[:8] == 0.0
        assert rejected_mask.any(), "Expected some RS rejections for extreme drift"
        # Kept samples should have weight 1
        kept_mask = weights[8:] == 1.0
        assert kept_mask.all(), "Expected all non-drifted samples to be kept"

    def test_no_rc_when_both_off(self):
        """When IS and RS are both null, no weights and only off-policy metrics."""
        batch_size = 4
        log_prob = torch.randn(batch_size)
        rollout_log_prob = log_prob + 0.01 * torch.randn(batch_size)

        rc_cfg = _make_rc_cfg(rollout_is=None, rollout_rs=None)
        data: dict = {}
        metrics = {}

        _apply_bypass_rc(log_prob, rollout_log_prob, rc_cfg, data, metrics)

        # No weights stashed
        assert "rollout_is_weights" not in data
        # Off-policy metrics still computed (KL, PPL, χ²)
        assert any(k.startswith("rollout_corr/") for k in metrics)

    def test_assert_on_invalid_loss_type(self):
        """Only ppo_clip is supported — assert fails on anything else."""
        batch_size = 2
        log_prob = torch.randn(batch_size)
        rollout_log_prob = log_prob.clone()

        # RolloutCorrectionConfig is frozen — construct with invalid loss_type directly
        rc_cfg = RolloutCorrectionConfig(
            bypass_mode=True,
            loss_type="reinforce",  # invalid
            rollout_is="sequence",
            rollout_is_threshold=2.0,
        )

        with pytest.raises(AssertionError, match="ppo_clip"):
            _apply_bypass_rc(log_prob, rollout_log_prob, rc_cfg, {}, {})


# ---------------------------------------------------------------------------
# Decoupled mode (apply_rollout_correction_to_diffusion_batch)
# ---------------------------------------------------------------------------


class TestDecoupledRC:
    """Tests for apply_rollout_correction_to_diffusion_batch — 3-policy mode."""

    def test_is_weights_computed(self):
        """IS weights correct old→rollout drift in decoupled mode."""
        batch_size, window = 8, 2
        old_log_prob = torch.randn(batch_size, window)
        rollout_log_prob = old_log_prob + 0.05 * torch.randn(batch_size, window)

        batch = DataProto.from_dict(
            tensors={
                "old_log_probs": old_log_prob,
                "rollout_log_probs": rollout_log_prob,
            }
        )

        rc_cfg = _make_rc_dict(
            rollout_is="sequence",
            rollout_is_threshold=2.0,
            rollout_rs=None,
        )

        result_batch, metrics = rollout_correction.apply_rollout_correction_to_diffusion_batch(batch, rc_cfg)

        assert "rollout_is_weights" in result_batch.batch
        assert "rollout_corr/rollout_is_mean" in metrics
        weights = result_batch.batch["rollout_is_weights"]
        assert weights.shape == (batch_size, window)
        # IS weights should be positive and ≤ threshold (ignoring clipping edge cases)
        assert (weights >= 0).all()

    def test_combined_is_rs(self):
        """IS and RS combine into a single rollout_is_weights tensor."""
        batch_size, window = 6, 2
        old_log_prob = torch.randn(batch_size, window)
        # Inject extreme drift in some samples
        rollout_log_prob = old_log_prob.clone()
        rollout_log_prob[0] = old_log_prob[0] + 4.0
        rollout_log_prob[1] = old_log_prob[1] - 4.0

        batch = DataProto.from_dict(
            tensors={
                "old_log_probs": old_log_prob,
                "rollout_log_probs": rollout_log_prob,
            }
        )

        rc_cfg = _make_rc_dict(
            rollout_is="token",
            rollout_is_threshold=2.0,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold="0.5_2.0",
        )

        result_batch, metrics = rollout_correction.apply_rollout_correction_to_diffusion_batch(batch, rc_cfg)

        weights = result_batch.batch["rollout_is_weights"]
        # Combined weights = IS × RS, so rejected samples get weight 0
        assert (weights[0] == 0.0).all() or (weights[1] == 0.0).all(), (
            "Extreme drift samples should be rejected (weight=0)"
        )

    def test_missing_keys_raises(self):
        """Missing rollout_log_probs or old_log_probs should raise."""
        batch = DataProto.from_dict(tensors={"old_log_probs": torch.randn(4, 2)})
        rc_cfg = _make_rc_dict()

        with pytest.raises(ValueError, match="rollout_log_probs"):
            rollout_correction.apply_rollout_correction_to_diffusion_batch(batch, rc_cfg)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes between old and rollout should raise."""
        batch = DataProto.from_dict(
            tensors={
                "old_log_probs": torch.randn(4, 2),
                "rollout_log_probs": torch.randn(4, 3),  # wrong window size
            }
        )
        rc_cfg = _make_rc_dict()

        with pytest.raises(ValueError, match="identical shapes"):
            rollout_correction.apply_rollout_correction_to_diffusion_batch(batch, rc_cfg)


# ---------------------------------------------------------------------------
# Loss integration
# ---------------------------------------------------------------------------


class TestLossIntegration:
    """Tests that rollout_is_weights affect the flow_grpo loss correctly."""

    @pytest.fixture(scope="class")
    def actor_config(self):
        from hydra import compose, initialize_config_dir
        from verl.utils.config import omega_conf_to_dataclass

        config_dir = os.path.abspath("verl_omni/trainer/config/diffusion/actor")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="dp_diffusion_actor",
                overrides=[
                    "strategy=fsdp",
                    "diffusion_loss.clip_ratio=0.0001",
                    "diffusion_loss.adv_clip_max=5.0",
                ],
            )
        return omega_conf_to_dataclass(cfg)

    def test_rollout_is_weights_rejected_samples_zero_loss(self, actor_config):
        """RS-rejected samples (weight=0) should contribute zero to per-element loss."""
        batch_size = 8
        old_log_prob = torch.randn(batch_size)
        log_prob = torch.randn(batch_size)
        advantages = torch.ones(batch_size)

        # Half the batch rejected (weight 0), half kept (weight 1)
        rollout_is_weights = torch.cat(
            [
                torch.zeros(batch_size // 2),
                torch.ones(batch_size // 2),
            ]
        )

        loss_fn = diffusion_algos.get_diffusion_loss_fn("flow_grpo")
        loss, metrics = loss_fn.compute_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            config=actor_config,
            rollout_is_weights=rollout_is_weights,
        )

        # Compute loss without weights to verify the weighted version is different
        loss_no_weights, _ = loss_fn.compute_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            config=actor_config,
            rollout_is_weights=None,
        )

        assert loss.shape == ()
        assert isinstance(loss.item(), float)
        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss, loss_no_weights)

    def test_all_rejected_gives_zero_loss(self, actor_config):
        """When all samples are rejected (weights=0), loss should be 0."""
        batch_size = 4
        old_log_prob = torch.randn(batch_size)
        log_prob = torch.randn(batch_size)
        advantages = torch.ones(batch_size)
        rollout_is_weights = torch.zeros(batch_size)

        loss_fn = diffusion_algos.get_diffusion_loss_fn("flow_grpo")
        loss, _ = loss_fn.compute_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            config=actor_config,
            rollout_is_weights=rollout_is_weights,
        )

        assert loss.item() == 0.0

    def test_pg_metrics_present(self, actor_config):
        """Standard loss metrics are present even with rollout_is_weights."""
        batch_size = 8
        old_log_prob = torch.randn(batch_size)
        log_prob = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss_fn = diffusion_algos.get_diffusion_loss_fn("flow_grpo")
        _, metrics = loss_fn.compute_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            config=actor_config,
            rollout_is_weights=torch.ones(batch_size),
        )

        for key in ("actor/ppo_kl", "actor/pg_clipfrac", "actor/ratio_mean"):
            assert key in metrics, f"Missing metric: {key}"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


class TestConsistencyMetrics:
    def test_abs_diff_aggregates(self):
        """Mean/max |Δlogp| are exact on known tensors."""
        old = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        rollout = torch.tensor([[1.5, 2.0], [2.0, 5.5]])
        metrics = rollout_correction.compute_rollout_corr_metrics_from_logprobs(old, rollout)
        assert metrics["rollout_corr/logprob_abs_diff_mean"] == pytest.approx(0.5)
        assert metrics["rollout_corr/logprob_abs_diff_max"] == pytest.approx(1.0)
        assert "rollout_corr/kl" in metrics

    def test_per_timestep_breakdown(self):
        """|Δlogp| is grouped by timestep value, not column index."""
        old = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        rollout = torch.tensor([[1.5, 2.0], [2.0, 5.5]])
        timesteps = torch.tensor([[800.0, 700.0], [800.0, 700.0]])
        metrics = rollout_correction.compute_rollout_corr_metrics_from_logprobs(old, rollout, timesteps=timesteps)
        assert metrics["rollout_corr/logprob_abs_diff/ts_800"] == pytest.approx(0.75)
        assert metrics["rollout_corr/logprob_abs_diff/ts_700"] == pytest.approx(0.25)

    def test_no_timesteps_no_breakdown(self):
        """Without timesteps the output has no per-timestep keys (backward compat)."""
        old = torch.randn(4, 2)
        metrics = rollout_correction.compute_rollout_corr_metrics_from_logprobs(old, old + 0.1)
        assert not any("logprob_abs_diff/ts_" in k for k in metrics)

    def test_shape_mismatch_skips_breakdown(self):
        """Misaligned timesteps yield aggregates only, never a wrong grouping."""
        old = torch.randn(4, 2)
        metrics = rollout_correction.compute_rollout_corr_metrics_from_logprobs(
            old, old + 0.1, timesteps=torch.randn(4, 3)
        )
        assert "rollout_corr/logprob_abs_diff_mean" in metrics
        assert not any("logprob_abs_diff/ts_" in k for k in metrics)


class TestConfigHelpers:
    def test_rollout_correction_enabled(self):
        """rollout_correction_enabled returns True when IS or RS is set."""
        assert rollout_correction.rollout_correction_enabled({"rollout_is": "sequence"})
        assert rollout_correction.rollout_correction_enabled({"rollout_rs": "seq_mean_k1"})
        assert rollout_correction.rollout_correction_enabled({"rollout_is": "token", "rollout_rs": "token_k1"})
        assert not rollout_correction.rollout_correction_enabled({"rollout_is": None, "rollout_rs": None})
        assert not rollout_correction.rollout_correction_enabled(None)

    def test_apply_bypass_mode_sets_old_log_probs(self):
        """Bypass mode copies rollout_log_probs to old_log_probs."""
        batch_size, window = 4, 2
        rollout_log_probs = torch.randn(batch_size, window)
        batch = DataProto.from_dict(tensors={"rollout_log_probs": rollout_log_probs})

        rollout_correction.apply_bypass_mode_to_diffusion_batch(batch)
        assert torch.equal(batch.batch["old_log_probs"], rollout_log_probs)

    def test_apply_bypass_mode_raises_without_rollout_log_probs(self):
        """Bypass mode requires rollout_log_probs in the batch."""
        batch = DataProto.from_dict(tensors={})

        with pytest.raises(ValueError, match="rollout_log_probs"):
            rollout_correction.apply_bypass_mode_to_diffusion_batch(batch)
