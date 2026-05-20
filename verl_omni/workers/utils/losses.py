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


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    # Rollout Correction bypass mode
    # In bypass mode old_log_prob == rollout_log_prob.  Compute per-step IS/RS
    # from (current, rollout) and stash weights for the loss function.
    rc_cfg = config.rollout_correction
    if rc_cfg.bypass_mode:
        # Diffusion log-probs are dense (no padding).
        log_prob_2d = log_prob.unsqueeze(-1)  # (B,) -> (B, 1)
        rollout_lp_2d = old_log_prob.unsqueeze(-1)  # (B,) -> (B, 1)
        response_mask = torch.ones_like(log_prob_2d)

        is_weights_proto, modified_mask, rc_metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob=log_prob_2d,
            rollout_log_prob=rollout_lp_2d,
            response_mask=response_mask,
            rollout_is=rc_cfg.rollout_is,
            rollout_is_threshold=rc_cfg.rollout_is_threshold,
            rollout_is_batch_normalize=rc_cfg.rollout_is_batch_normalize,
            rollout_rs=rc_cfg.rollout_rs,
            rollout_rs_threshold=rc_cfg.rollout_rs_threshold,
        )

        # ppo_clip: PPO ratio handles IS, only RS mask is applied.
        # reinforce: IS weights applied explicitly (no PPO clipping).
        ppo_clip = rc_cfg.loss_type == "ppo_clip"
        weights: torch.Tensor | None = None

        if is_weights_proto is not None and not ppo_clip:
            weights = is_weights_proto.batch["rollout_is_weights"]  # (B, 1)

        if rc_cfg.rollout_rs:
            rs_mask = modified_mask  # (B, 1), 1=keep, 0=reject
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
