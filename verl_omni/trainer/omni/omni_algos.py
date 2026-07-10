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

"""Omni-specific loss functions."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl_omni.workers.config import VeOmniOmniActorConfig


@dataclass
class OmniLossResult:
    loss: torch.Tensor
    metrics: dict[str, Any]


OMNI_LOSS_REGISTRY: dict[str, Any] = {}


def register_omni_loss(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        OMNI_LOSS_REGISTRY[name] = cls()
        return cls

    return decorator


def get_omni_loss_fn(name: str):
    if name not in OMNI_LOSS_REGISTRY:
        raise ValueError(f"Unsupported omni loss mode: {name}. Supported modes are: {list(OMNI_LOSS_REGISTRY.keys())}")
    return OMNI_LOSS_REGISTRY[name]


@register_omni_loss("dpo")
class OmniDPOLoss:
    required_model_output_keys: tuple[str, ...] = (
        "policy_chosen_logps",
        "policy_rejected_logps",
        "reference_chosen_logps",
        "reference_rejected_logps",
    )

    def validate_inputs(self, *, model_output: dict[str, Any], data: TensorDict) -> None:
        del data
        missing_model_output = [key for key in self.required_model_output_keys if key not in model_output]
        if missing_model_output:
            available = sorted(str(key) for key in model_output.keys())
            raise KeyError(
                "Omni DPO loss is missing required model_output keys: "
                f"{missing_model_output}. Available model_output keys: {available}."
            )

    @staticmethod
    def compute_loss(
        *,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        if reference_free:
            ref_logratios = 0
        logits = pi_logratios - ref_logratios
        if loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
            )
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        loss = losses.mean()
        metrics = {
            "dpo_loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().detach(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        }
        return loss, metrics

    def __call__(
        self,
        *,
        config: VeOmniOmniActorConfig,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> OmniLossResult:
        self.validate_inputs(model_output=model_output, data=data)
        dpo_config = config.omni_loss
        loss, metrics = self.compute_loss(
            policy_chosen_logps=model_output["policy_chosen_logps"],
            policy_rejected_logps=model_output["policy_rejected_logps"],
            reference_chosen_logps=model_output["reference_chosen_logps"],
            reference_rejected_logps=model_output["reference_rejected_logps"],
            beta=dpo_config.beta,
            label_smoothing=dpo_config.label_smoothing,
            loss_type=dpo_config.loss_type,
            reference_free=dpo_config.reference_free,
        )
        return OmniLossResult(loss=loss, metrics=metrics)
