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


from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn, kl_penalty_image
from verl_omni.workers.config import DiffusionActorConfig


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}
    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")

    if loss_mode == "dpo":
        policy_loss_fn = get_diffusion_loss_fn(loss_mode)
        ref_noise_pred = data.get("ref_noise_pred", None)
        if ref_noise_pred is None:
            raise KeyError("DPO loss requires `ref_noise_pred` from the reference diffusion model.")
        policy_loss_kwargs = dict(
            noise=model_output["noise"],
            model_noise_pred=model_output["noise_pred"],
            ref_noise_pred=ref_noise_pred,
            sample_level_scores=data["sample_level_scores"],
            config=config,
            index=tu.get_non_tensor_data(data, "uid", default=None),
        )
        pg_loss, pg_metrics = policy_loss_fn(**policy_loss_kwargs)
        pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
        metrics.update(pg_metrics)
        metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)

        gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
        policy_loss = pg_loss / gradient_accumulation_steps

        sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
        if sp_size > 1:
            policy_loss = policy_loss * sp_size

        return policy_loss, metrics

    # compute policy loss
    log_prob = model_output["log_probs"]
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    policy_loss_fn = get_diffusion_loss_fn(loss_mode)
    policy_loss_kwargs = dict(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        config=config,
    )
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
