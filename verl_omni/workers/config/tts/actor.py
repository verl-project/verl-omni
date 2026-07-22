# Copyright 2026 Gulp AI Inc and/or its affiliates
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

from dataclasses import dataclass

from verl.workers.config import PolicyLossConfig

__all__ = ["DPOPolicyLossConfig"]


@dataclass
class DPOPolicyLossConfig(PolicyLossConfig):
    """Policy-loss config for the online TTS DPO recipe (tts_dpo_loss).

    Selected in yaml via actor.policy_loss._target_ = verl_omni.workers.config.DPOPolicyLossConfig,
    carrying the DPO knobs the base PolicyLossConfig has no field for.
    """

    loss_mode: str = "dpo"
    dpo_beta: float = 0.1
    dpo_nll_lambda: float = 0.0

    def __post_init__(self):
        if self.loss_mode != "dpo":
            raise ValueError(f"DPOPolicyLossConfig requires loss_mode='dpo', got {self.loss_mode!r}.")
        if self.dpo_beta <= 0:
            raise ValueError(f"dpo_beta must be positive, got {self.dpo_beta}.")
        if self.dpo_nll_lambda < 0:
            raise ValueError(f"dpo_nll_lambda must be non-negative, got {self.dpo_nll_lambda}.")
