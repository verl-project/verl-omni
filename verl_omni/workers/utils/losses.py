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

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")
    loss_func = get_diffusion_loss_fn(loss_mode)
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
