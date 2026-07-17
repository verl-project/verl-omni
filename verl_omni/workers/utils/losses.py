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

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.trainer.omni.omni_algos import get_omni_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, OmniActorConfig


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
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")
    loss_func = get_diffusion_loss_fn(loss_mode)

    # Rollout Correction bypass mode only applies to log-prob policy-gradient losses.
    if "log_probs" in loss_func.required_model_output_keys:
        log_prob = model_output["log_probs"]
        old_log_prob = data["old_log_probs"]
        rc_cfg = config.rollout_correction
        # Rollout Correction bypass mode: compute IS/RS weights per-step and
        # stash ``rollout_is_weights`` into ``data`` before loss dispatch.
        if rc_cfg.bypass_mode:
            _apply_bypass_rc(log_prob, old_log_prob, rc_cfg, data, metrics)

    loss_func.validate_inputs(loss_name=loss_mode, model_output=model_output, data=data)
    loss_result = loss_func(config=config, model_output=model_output, data=data)
    loss_value = loss_result.loss
    metrics_values = loss_result.metrics

    metrics_values = Metric.from_dict(metrics_values, aggregation=AggregationType.MEAN)

    metrics.update(metrics_values)
    if loss_result.add_loss_metric:
        metrics["actor/loss"] = Metric(value=loss_value, aggregation=AggregationType.MEAN)

    if config.use_kl_loss:
        loss_func = get_diffusion_loss_fn("kl")
        loss_func.validate_inputs(loss_name="kl", model_output=model_output, data=data)
        kl_result = loss_func(config=config, model_output=model_output, data=data)
        loss_value += kl_result.loss * config.kl_loss_coef
        metrics.update(Metric.from_dict(kl_result.metrics, aggregation=AggregationType.MEAN))
        metrics["kl_coef"] = config.kl_loss_coef
        if kl_result.add_loss_metric:
            metrics["actor/weighted_kl_loss"] = Metric(
                value=kl_result.loss * config.kl_loss_coef,
                aggregation=AggregationType.MEAN,
            )

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    loss_value = loss_value / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size > 1:
        loss_value = loss_value * sp_size

    return loss_value, metrics


def omni_loss(config: OmniActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for omni AR direct-preference training."""
    del dp_group
    metrics = {}

    loss_mode = config.omni_loss.loss_mode
    loss_func = get_omni_loss_fn(loss_mode)
    loss_func.validate_inputs(model_output=model_output, data=data)
    loss_result = loss_func(config=config, model_output=model_output, data=data)
    loss_value = loss_result.loss
    metrics.update(Metric.from_dict(loss_result.metrics, aggregation=AggregationType.MEAN))
    metrics["actor/loss"] = Metric(value=loss_value, aggregation=AggregationType.MEAN)

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    if gradient_accumulation_steps:
        loss_value = loss_value / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size is not None and sp_size > 1:
        loss_value = loss_value * sp_size

    return loss_value, metrics
