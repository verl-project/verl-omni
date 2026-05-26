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

import torch
from tensordict import TensorDict
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn, kl_penalty_image
from verl_omni.workers.config import DiffusionActorConfig


def _apply_bypass_rc(
    log_prob: torch.Tensor,  # (B,) current policy log-prob
    old_log_prob: torch.Tensor,  # (B,) == rollout_log_prob in bypass
    rc_cfg,  # RolloutCorrectionConfig
    data: TensorDict,  # modified in-place
    metrics: dict,  # modified in-place
) -> None:
    """Compute per-step IS/RS for bypass mode and stash weights into ``data``."""
    log_prob_2d = log_prob.unsqueeze(-1)  # current policy log-prob (π_θ)
    rollout_lp_2d = old_log_prob.unsqueeze(-1)  # rollout policy log-prob (π_rollout)
    response_mask = torch.ones_like(log_prob_2d)

    # In bypass mode, RS checks current→rollout drift: pass current as old_log_prob, rollout as rollout_log_prob.
    # This matches the mathematical intent: RS mask is applied to exp(log_prob - rollout_log_prob).
    is_weights_proto, modified_mask, rc_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob_2d,  # current policy (π_θ)
        rollout_log_prob=rollout_lp_2d,  # rollout policy (π_rollout)
        response_mask=response_mask,
        rollout_is=rc_cfg.rollout_is,
        rollout_is_threshold=rc_cfg.rollout_is_threshold,
        rollout_is_batch_normalize=rc_cfg.rollout_is_batch_normalize,
        rollout_rs=rc_cfg.rollout_rs,
        rollout_rs_threshold=rc_cfg.rollout_rs_threshold,
    )

    # ppo_clip: PPO ratio handles IS, only RS mask is applied.
    assert rc_cfg.loss_type == "ppo_clip", f"Only loss_type='ppo_clip' is supported, got {rc_cfg.loss_type!r}"
    weights: torch.Tensor | None = None

    if rc_cfg.rollout_rs:
        rs_mask = modified_mask
        weights = rs_mask if weights is None else weights * rs_mask

    if weights is not None:
        existing = data.get("rollout_is_weights", None)
        data["rollout_is_weights"] = (
            weights.squeeze(-1).to(dtype=log_prob.dtype)
            if existing is None
            else existing * weights.squeeze(-1).to(dtype=log_prob.dtype)
        )

    for k, v in rc_metrics.items():
        metrics[k] = Metric(value=float(v), aggregation=AggregationType.MEAN)


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    # Rollout Correction bypass mode
    rc_cfg = config.rollout_correction
    if rc_cfg.bypass_mode:
        _apply_bypass_rc(log_prob, old_log_prob, rc_cfg, data, metrics)

    # Standard loss dispatch
    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")

    policy_loss_fn = get_diffusion_loss_fn(loss_mode)
    policy_loss_kwargs = dict(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        config=config,
    )
    if "rollout_is_weights" in data.keys():
        policy_loss_kwargs["rollout_is_weights"] = data["rollout_is_weights"]

    if loss_mode == "grpo_guard":
        # GRPO-Guard requires the rollout-time SDE proposal mean and the per-step
        # diffusion coefficient terms; pass them through alongside the standard inputs.
        policy_loss_kwargs.update(
            old_prev_sample_mean=data["old_prev_sample_mean"],
            prev_sample_mean=model_output["prev_sample_mean"],
            std_dev_t=model_output["std_dev_t"],
            sqrt_dt=model_output["sqrt_dt"],
        )
    pg_loss, pg_metrics = policy_loss_fn(**policy_loss_kwargs)

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size > 1:
        policy_loss = policy_loss * sp_size

    return policy_loss, metrics
