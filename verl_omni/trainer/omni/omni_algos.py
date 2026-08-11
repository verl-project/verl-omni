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

"""Omni AR direct-preference loss functions."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl_omni.workers.config.omni import OmniLossConfig

__all__ = [
    "OmniLossResult",
    "OMNI_LOSS_REGISTRY",
    "register_omni_loss",
    "get_omni_loss_fn",
    "OmniDPOLoss",
]


@dataclass
class OmniLossResult:
    loss: torch.Tensor
    metrics: dict[str, Any]


OMNI_LOSS_REGISTRY: dict[str, Any] = {}


def register_omni_loss(name: str) -> Callable[[type], type]:
    """Register a worker-side omni loss function class."""

    def decorator(cls: type) -> type:
        OMNI_LOSS_REGISTRY[name] = cls()
        return cls

    return decorator


def get_omni_loss_fn(name: str):
    """Return the registered omni loss function for ``name``."""
    if name not in OMNI_LOSS_REGISTRY:
        raise ValueError(f"Unsupported omni loss mode: {name}. Supported modes are: {list(OMNI_LOSS_REGISTRY.keys())}")
    return OMNI_LOSS_REGISTRY[name]


@register_omni_loss("dpo")
class OmniDPOLoss:
    """Bradley-Terry DPO on sequence-level policy vs. reference log-probs."""

    required_model_output_keys: tuple[str, ...] = ("log_probs",)
    required_data_keys: tuple[str, ...] = ("labels", "ref_log_prob")

    def validate_inputs(self, *, model_output: dict[str, Any], data: TensorDict) -> None:
        missing_model_output = [key for key in self.required_model_output_keys if key not in model_output]
        if missing_model_output:
            available = sorted(str(key) for key in model_output.keys())
            raise KeyError(
                "Omni DPO loss is missing required model_output keys: "
                f"{missing_model_output}. Available model_output keys: {available}."
            )
        missing_data = [key for key in self.required_data_keys if key not in data]
        if missing_data:
            available = sorted(str(key) for key in data.keys())
            raise KeyError(
                f"Omni DPO loss is missing required data keys: {missing_data}. Available data keys: {available}."
            )

    @staticmethod
    def _split_adjacent_logps(name: str, logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logps.shape[0] % 2 != 0:
            raise ValueError(f"{name} must contain adjacent chosen/rejected rows, got shape {tuple(logps.shape)}.")
        return logps[0::2], logps[1::2]

    @staticmethod
    def _masked_sum_row_log_probs(
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        average_log_prob: bool,
    ) -> torch.Tensor:
        if log_probs.shape[-1] == labels.shape[-1]:
            log_probs = log_probs[..., :-1]
            label_mask = labels[..., 1:] != -100
        elif log_probs.shape[-1] == labels.shape[-1] - 1:
            label_mask = labels[..., 1:] != -100
        else:
            raise ValueError(
                "Token log_probs must align with labels or shifted labels; "
                f"got log_probs shape {tuple(log_probs.shape)} and labels shape {tuple(labels.shape)}."
            )
        seq_logps = (log_probs * label_mask).sum(dim=-1)
        if average_log_prob:
            seq_logps = seq_logps / label_mask.sum(dim=-1).clamp(min=1)
        return seq_logps

    @classmethod
    def _sequence_logps_from_token_log_probs(
        cls,
        *,
        name: str,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool,
    ) -> torch.Tensor:
        if log_probs.is_nested:
            if not labels.is_nested:
                raise ValueError(f"{name} is nested, but labels are not nested.")
            rows = [
                cls._masked_sum_row_log_probs(
                    row_log_probs,
                    row_labels,
                    average_log_prob=average_log_prob,
                )
                for row_log_probs, row_labels in zip(log_probs.unbind(), labels.unbind(), strict=True)
            ]
            return torch.stack(rows)
        if labels.is_nested:
            raise ValueError(f"{name} is dense, but labels are nested.")
        return cls._masked_sum_row_log_probs(log_probs, labels, average_log_prob=average_log_prob)

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
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
        config: Any,
        model_output: dict[str, Any],
        data: TensorDict,
    ) -> OmniLossResult:
        self.validate_inputs(model_output=model_output, data=data)
        dpo_config: OmniLossConfig = config.omni_loss
        policy_logps = self._sequence_logps_from_token_log_probs(
            name="log_probs",
            log_probs=model_output["log_probs"],
            labels=data["labels"],
            average_log_prob=dpo_config.average_log_prob,
        )
        reference_logps = self._sequence_logps_from_token_log_probs(
            name="ref_log_prob",
            log_probs=data["ref_log_prob"],
            labels=data["labels"],
            average_log_prob=dpo_config.average_log_prob,
        )
        policy_chosen_logps, policy_rejected_logps = self._split_adjacent_logps("log_probs", policy_logps)
        reference_chosen_logps, reference_rejected_logps = self._split_adjacent_logps("ref_log_prob", reference_logps)
        if policy_chosen_logps.shape != reference_chosen_logps.shape:
            raise ValueError(
                "Policy and reference log-probs must have matching preference batch shape; "
                f"got {tuple(policy_chosen_logps.shape)} and {tuple(reference_chosen_logps.shape)}."
            )
        loss, metrics = self.compute_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            beta=dpo_config.beta,
            label_smoothing=dpo_config.label_smoothing,
            loss_type=dpo_config.loss_type,
        )
        return OmniLossResult(loss=loss, metrics=metrics)
