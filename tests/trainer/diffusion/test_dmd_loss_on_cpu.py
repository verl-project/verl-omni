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

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_algos import DiffusionLossResult, DMDLoss, get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.utils.losses import diffusion_loss


def make_actor():
    return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2"))


def make_batch(size, stage="student", accumulation=1, sp_size=1):
    batch = TensorDict({}, batch_size=[size])
    tu.assign_non_tensor_data(batch, "dmd_stage", stage)
    tu.assign_non_tensor_data(batch, "gradient_accumulation_steps", accumulation)
    tu.assign_non_tensor_data(batch, "sp_size", sp_size)
    return batch


class TestDMDLoss:
    def test_registration_does_not_alias_original_dmd(self):
        assert isinstance(get_diffusion_loss_fn("dmd2"), DMDLoss)
        with pytest.raises(ValueError, match="Unsupported"):
            get_diffusion_loss_fn("dmd")
        with pytest.raises(ValueError, match="loss_mode"):
            DiffusionLossConfig(loss_mode="dmd")

    def test_student_gradient_and_detached_scores(self):
        student = torch.tensor([[1.0, 3.0], [4.0, 8.0]], requires_grad=True)
        teacher = torch.zeros_like(student, requires_grad=True)
        fake = torch.ones_like(student, requires_grad=True)
        result = DMDLoss()(
            config=make_actor(),
            model_output={
                "generated_x0": student,
                "teacher_x0": teacher,
                "fake_x0": fake,
            },
            data=make_batch(2),
        )
        assert isinstance(result, DiffusionLossResult)
        result.loss.backward()
        expected = torch.tensor([[0.5, 0.5], [1 / 6, 1 / 6]]) / student.numel()
        torch.testing.assert_close(student.grad, expected)
        assert teacher.grad is None and fake.grad is None
        assert result.metrics["dmd/nonfinite"] == 0

    def test_fake_stage_uses_detached_flow_target(self):
        student = torch.ones(2, 3, requires_grad=True)
        prediction = torch.zeros(2, 3, requires_grad=True)
        noise = torch.full_like(student, 2.0, requires_grad=True)
        loss, _ = diffusion_loss(
            make_actor(),
            {
                "generated_x0": student,
                "noise_pred": prediction,
                "noise": noise,
            },
            make_batch(2, "fake_score"),
        )
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(1.0))
        assert student.grad is None and noise.grad is None
        torch.testing.assert_close(prediction.grad, torch.full_like(prediction, -2 / prediction.numel()))

    @pytest.mark.parametrize("stage,missing", [("student", "teacher_x0"), ("fake_score", "noise_pred")])
    def test_stage_inputs_fail_closed(self, stage, missing):
        output = {key: torch.ones(1, 2) for key in ("generated_x0", "teacher_x0", "fake_x0", "noise_pred", "noise")}
        del output[missing]
        with pytest.raises(KeyError, match=missing):
            diffusion_loss(make_actor(), output, make_batch(1, stage))

    def test_unknown_stage_is_not_student_fallback(self):
        with pytest.raises(ValueError, match="dmd_stage"):
            diffusion_loss(make_actor(), {}, make_batch(1, "discriminator"))

    def test_normalization_setting_reaches_registered_loss(self):
        batch = make_batch(1)
        tu.assign_non_tensor_data(batch, "dmd_normalization_epsilon", 0.5)
        output = {"generated_x0": torch.zeros(1, 2), "teacher_x0": torch.zeros(1, 2), "fake_x0": torch.ones(1, 2)}
        loss, _ = diffusion_loss(make_actor(), output, batch)
        torch.testing.assert_close(loss, torch.tensor(2.0))

    @pytest.mark.parametrize("stage", ["student", "fake_score"])
    def test_unequal_microbatches_preserve_loss_and_gradient(self, stage):
        full = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]], requires_grad=True)
        accumulated = full.detach().clone().requires_grad_()
        if stage == "student":
            output = {"generated_x0": full, "fake_x0": torch.ones_like(full), "teacher_x0": torch.zeros_like(full)}
        else:
            output = {"noise_pred": full, "noise": torch.ones_like(full), "generated_x0": torch.zeros_like(full)}
        loss, _ = diffusion_loss(make_actor(), output, make_batch(3, stage))
        loss.backward()
        total = 0.0
        for start, stop in ((0, 2), (2, 3)):
            micro = {key: tensor[start:stop] for key, tensor in output.items()}
            micro["generated_x0" if stage == "student" else "noise_pred"] = accumulated[start:stop]
            micro_loss, _ = diffusion_loss(make_actor(), micro, make_batch(stop - start, stage, 3 / (stop - start)))
            total += micro_loss.detach()
            micro_loss.backward()
        torch.testing.assert_close(total, loss.detach())
        torch.testing.assert_close(accumulated.grad, full.grad)

    def test_sequence_parallel_factor_applied_once(self):
        output = {"generated_x0": torch.ones(1, 2), "teacher_x0": torch.zeros(1, 2), "fake_x0": torch.ones(1, 2)}
        single, _ = diffusion_loss(make_actor(), output, make_batch(1))
        scaled, _ = diffusion_loss(make_actor(), output, make_batch(1, accumulation=2, sp_size=4))
        torch.testing.assert_close(scaled, single * 2)
