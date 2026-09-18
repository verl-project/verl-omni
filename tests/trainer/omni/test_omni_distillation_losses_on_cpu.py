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
"""CPU tests for the omni hidden-state distillation loss dispatch (Task 3)."""

import torch
from tensordict import NonTensorData, TensorDict

from verl_omni.trainer.distillation.losses import (
    omni_distillation_ppo_loss,
)
from verl_omni.workers.config.omni.distillation import HIDDEN_STATE_LOSS_MODES


class _LossConfig:
    def __init__(self, loss_mode="nitrobrew", use_policy_gradient=False, use_hidden_states=True, topk=True):
        self.loss_mode = loss_mode
        self.use_policy_gradient = use_policy_gradient
        self.loss_settings = type("S", (), {"use_topk": topk, "use_hidden_states": use_hidden_states})()


class _DistillConfig:
    def __init__(self, loss_mode="nitrobrew"):
        self.distillation_loss = _LossConfig(loss_mode=loss_mode)


class TestLogitsProcessorDispatch:
    def test_nitrobrew_goes_to_kernel(self, monkeypatch):
        called = {}

        def fake_multi_kl(
            student_logits=None,
            teacher_hidden_states=None,
            teacher_key_ids=None,
            teacher_unembeds=None,
            config=None,
            data_format=None,
        ):
            called["called"] = True
            assert student_logits.shape == (1, 4, 32)
            assert teacher_key_ids.shape[0] == 4
            assert set(teacher_unembeds.keys()) == {0}
            return {"distillation_losses": torch.zeros(1, 4)}

        monkeypatch.setattr("verl_omni.trainer.distillation.losses.compute_nitrobrew_multi_kl", fake_multi_kl)

        hidden = torch.randn(1, 4, 8)
        student_logits = torch.randn(1, 4, 32)
        data = TensorDict(
            {
                "teacher_hidden_states": hidden,
                "teacher_unembeds": NonTensorData({"t0": torch.randn(32, 8)}),
                "teacher_key_to_id": NonTensorData({"t0": 0}),
            },
            batch_size=[1],
        )

        out = omni_distillation_ppo_loss(
            config=None,
            distillation_config=_DistillConfig("nitrobrew"),
            data=data,
            student_logits=student_logits,
            data_format="thd",
        )
        assert called["called"]
        assert "distillation_losses" in out

    def test_estimator_mode_delegates(self, monkeypatch):
        """non-hidden loss_mode must fall through to verl's distillation_ppo_loss."""

        # provide a stub to prove delegation
        def fake_verl_ppo_loss(config, distillation_config, model_output, data, dp_group, student_logits, data_format):
            return "DELEGATED"

        monkeypatch.setattr("verl_omni.trainer.distillation.losses.distillation_ppo_loss", fake_verl_ppo_loss)
        out = omni_distillation_ppo_loss(
            config=None,
            distillation_config=_DistillConfig("kl"),
            student_logits=torch.randn(1, 2, 4),
            data_format="thd",
        )
        assert out == "DELEGATED"


class TestSingleTeacherRouting:
    def test_zero_key_ids_with_single_teacher(self):
        from verl_omni.trainer.distillation.losses import _per_token_teacher_key_ids

        hidden = torch.randn(1, 4, 8)
        data = TensorDict({"teacher_hidden_states": hidden}, batch_size=[1])
        ids = _per_token_teacher_key_ids(data, hidden, {"t0": 0})
        assert ids.shape[0] == 4


class TestAggregateRegistered:
    def test_nitrobrew_in_verl_registry(self):
        """importing the module must register nitrobrew aggregate into verl's registry."""
        from verl.trainer.distillation.losses import get_distillation_loss_fn, get_distillation_loss_settings

        for mode in HIDDEN_STATE_LOSS_MODES:
            fn = get_distillation_loss_fn(mode)
            settings = get_distillation_loss_settings(mode)
            assert callable(fn)
            assert set(settings.names) == set(HIDDEN_STATE_LOSS_MODES)
