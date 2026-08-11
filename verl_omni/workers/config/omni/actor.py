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

from dataclasses import dataclass, field

from verl.base_config import BaseConfig
from verl.workers.config import FSDPActorConfig

__all__ = [
    "OmniLossConfig",
    "OmniActorConfig",
]


@dataclass
class OmniLossConfig(BaseConfig):
    """Loss hyperparameters for omni AR direct-preference and SFT training.

    Which config block to use depends on algorithm.trainer_type (OmniAlgoConfig):

    * policy_gradient (online RL: GSPO, GRPO, PPO, ...): use verl's inherited
      actor_rollout_ref.actor.policy_loss (PolicyLossConfig) and sibling
      actor fields such as clip_ratio_low, clip_ratio_high,
      loss_agg_mode, use_kl_loss, and kl_loss_coef. Those are consumed
      by verl.trainer.ppo.core_algos via get_policy_loss_fn(). This
      dataclass is not read on that path.
    * direct_preference (offline/online DPO) and sft (offline supervised
      training): use this block at YAML path actor_rollout_ref.actor.omni_loss.
      Consumed by verl_omni.trainer.omni.omni_algos and the corresponding
      offline trainer.

    Field reference (direct_preference only)
    ----------------------------------------

    loss_mode:
        Omni loss registry key. Currently "dpo" and "omni_sft" are supported.
    beta:
        DPO inverse temperature β. Scales the log-probability margin between
        policy and reference on chosen vs. rejected pairs before the sigmoid/IPO
        loss. Typical values for token-level AR DPO are ~0.01–0.5 (default 0.1).
    label_smoothing:
        Label smoothing for the Bradley-Terry sigmoid DPO loss (cDPO). 0.0
        disables smoothing; values in (0, 1) soften chosen/rejected targets.
        Ignored when loss_type="ipo".
    loss_type:
        "sigmoid" — standard DPO -log σ(β·Δlogπ); "ipo" — identity
        preference optimization (squared error on the implicit reward).
    average_log_prob:
        If True, sequence log-probs are averaged over response tokens before
        the pairwise DPO margin; if False, token log-probs are summed (TRL
        default). Passed through the engine micro-batch for log-prob aggregation.
    refer_model_precision:
        Parameter dtype for the reference (frozen) policy during ref log-prob
        computation, e.g. "bfloat16" or "float32". Policy (trainable)
        precision is controlled separately by actor.fsdp_config.model_dtype.
    ce_weight:
        Text cross-entropy weight for supervised omni SFT.
    mse_weight:
        Image velocity MSE weight for supervised omni SFT.
    ignore_index:
        Label value ignored by supervised cross entropy.
    """

    loss_mode: str = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    average_log_prob: bool = False
    refer_model_precision: str = "bfloat16"
    ce_weight: float = 1.0
    mse_weight: float = 1.0
    ignore_index: int = -100

    def __post_init__(self):
        valid_modes = {"dpo", "omni_sft"}
        if self.loss_mode not in valid_modes:
            raise ValueError(
                f"Unsupported omni loss_mode={self.loss_mode!r}; currently supported: {sorted(valid_modes)}."
            )
        if self.loss_type not in {"sigmoid", "ipo"}:
            raise ValueError(f"Invalid omni DPO loss_type={self.loss_type!r}; expected 'sigmoid' or 'ipo'.")
        if self.beta <= 0:
            raise ValueError(f"Omni DPO beta must be positive, got {self.beta}.")
        if self.ce_weight < 0:
            raise ValueError(f"Omni SFT ce_weight must be non-negative, got {self.ce_weight}.")
        if self.mse_weight < 0:
            raise ValueError(f"Omni SFT mse_weight must be non-negative, got {self.mse_weight}.")


@dataclass
class OmniActorConfig(FSDPActorConfig):
    """FSDP actor config for omni model training."""

    trainer_type: str = "direct_preference"  # "direct_preference", "policy_gradient", or "sft"
    omni_loss: OmniLossConfig = field(default_factory=OmniLossConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.trainer_type not in ["direct_preference", "policy_gradient", "sft"]:
            raise ValueError(
                f"Invalid omni trainer_type={self.trainer_type}; "
                "expected ['direct_preference', 'policy_gradient', 'sft']."
            )
        if self.trainer_type == "direct_preference" and self.omni_loss is None:
            raise ValueError("OmniActorConfig.omni_loss is required for direct_preference training.")
