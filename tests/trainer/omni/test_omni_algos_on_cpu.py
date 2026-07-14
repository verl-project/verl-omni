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
"""CPU tests for omni trainer algorithms."""

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl_omni.trainer.omni.omni_algos import OmniDPOLoss, get_omni_loss_fn
from verl_omni.workers.config.omni.actor import OmniLossConfig, VeOmniOmniActorConfig


def _model_output() -> dict[str, torch.Tensor]:
    return {
        "policy_chosen_logps": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "policy_rejected_logps": torch.tensor([-0.3, 0.1], dtype=torch.float32),
        "reference_chosen_logps": torch.tensor([0.0, 0.2], dtype=torch.float32),
        "reference_rejected_logps": torch.tensor([-0.1, -0.2], dtype=torch.float32),
    }


def test_dpo_loss_registry():
    loss_fn = get_omni_loss_fn("dpo")
    assert isinstance(loss_fn, OmniDPOLoss)


def test_unknown_loss_mode_raises():
    with pytest.raises(ValueError, match="Unsupported omni loss mode"):
        get_omni_loss_fn("not_registered")


def test_validate_inputs_reports_missing_keys():
    loss_fn = OmniDPOLoss()
    with pytest.raises(KeyError, match="policy_rejected_logps"):
        loss_fn.validate_inputs(
            model_output={"policy_chosen_logps": torch.zeros(1)}, data=TensorDict({}, batch_size=[])
        )


def test_compute_sigmoid_dpo_loss_matches_reference():
    model_output = _model_output()
    beta = 0.3
    label_smoothing = 0.1

    loss, metrics = OmniDPOLoss.compute_loss(**model_output, beta=beta, label_smoothing=label_smoothing)

    pi_logratios = model_output["policy_chosen_logps"] - model_output["policy_rejected_logps"]
    ref_logratios = model_output["reference_chosen_logps"] - model_output["reference_rejected_logps"]
    logits = pi_logratios - ref_logratios
    expected = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(metrics["dpo_loss"], expected.mean())
    torch.testing.assert_close(metrics["reward_accuracy"], torch.tensor(1.0))
    assert "reward_margin" in metrics


def test_compute_ipo_reference_free_loss():
    model_output = _model_output()
    beta = 0.5

    loss, metrics = OmniDPOLoss.compute_loss(**model_output, beta=beta, loss_type="ipo", reference_free=True)

    logits = model_output["policy_chosen_logps"] - model_output["policy_rejected_logps"]
    expected = (logits - 1 / (2 * beta)) ** 2
    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(metrics["dpo_loss"], expected.mean())


def test_call_uses_actor_config_loss_settings():
    model_output = _model_output()
    cfg = VeOmniOmniActorConfig(
        ppo_micro_batch_size_per_gpu=2,
        rollout_n=2,
        omni_loss=OmniLossConfig(beta=0.5, loss_type="ipo", reference_free=True),
    )

    result = OmniDPOLoss()(config=cfg, model_output=model_output, data=TensorDict({}, batch_size=[]))

    expected, _ = OmniDPOLoss.compute_loss(**model_output, beta=0.5, loss_type="ipo", reference_free=True)
    torch.testing.assert_close(result.loss, expected)
    assert "dpo_loss" in result.metrics
