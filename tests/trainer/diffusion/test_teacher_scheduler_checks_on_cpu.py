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
"""CPU tests for the two-stage teacher scheduler validation (RFC #293, §5.2)."""

import pytest
import torch

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.teacher_scheduler_checks import (
    build_cpu_scheduler,
    validate_request_timesteps,
    validate_scheduler_grids,
)
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


@pytest.fixture
def adapter(diffusion_model_config, fake_sd3_checkpoint):
    """The SD3 FlowGRPO training adapter, looked up the way the runtime looks it up."""
    return DiffusionModelBase.get_class(diffusion_model_config(fake_sd3_checkpoint("probe")))


class TestStage1Grids:
    def test_identical_configs_pass(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher = diffusion_model_config(fake_sd3_checkpoint("teacher"))

        actor_scheduler, teacher_scheduler = validate_scheduler_grids(actor, teacher, adapter, "default")

        assert torch.equal(actor_scheduler.timesteps, teacher_scheduler.timesteps)
        assert torch.equal(actor_scheduler.sigmas, teacher_scheduler.sigmas)

    def test_shifted_grid_rejected(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        """`shift` is one of ~10 interacting fields; the resolved grid is what is decidable."""
        actor_path = fake_sd3_checkpoint("student", shift=1.0)
        teacher_path = fake_sd3_checkpoint("teacher", shift=3.0)
        actor = diffusion_model_config(actor_path)
        teacher = diffusion_model_config(teacher_path)

        with pytest.raises(ValueError) as excinfo:
            validate_scheduler_grids(actor, teacher, adapter, "ocr_expert")

        message = str(excinfo.value)
        assert "ocr_expert" in message
        assert teacher_path in message
        assert "first mismatch at index" in message

    def test_step_count_mismatch_rejected(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher = diffusion_model_config(
            fake_sd3_checkpoint("teacher"), pipeline=DiffusionPipelineConfig(num_inference_steps=20)
        )

        with pytest.raises(ValueError, match="resolved timesteps differ"):
            validate_scheduler_grids(actor, teacher, adapter, "default")

    def test_grids_built_on_cpu(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher = diffusion_model_config(fake_sd3_checkpoint("teacher"))

        actor_scheduler, _ = validate_scheduler_grids(actor, teacher, adapter, "default")

        assert actor_scheduler.timesteps.device.type == "cpu"
        assert actor_scheduler.sigmas.device.type == "cpu"

    def test_does_not_go_through_build_scheduler(
        self, adapter, fake_sd3_checkpoint, diffusion_model_config, monkeypatch
    ):
        """build_scheduler() resolves onto get_device_name(); the driver may have no CUDA."""

        def forbidden(*args, **kwargs):
            raise AssertionError("stage 1 must not call adapter.build_scheduler")

        monkeypatch.setattr(adapter, "build_scheduler", forbidden)
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher = diffusion_model_config(fake_sd3_checkpoint("teacher"))

        validate_scheduler_grids(actor, teacher, adapter, "default")


class TestStage2RequestTimesteps:
    @pytest.fixture
    def schedulers(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher = diffusion_model_config(fake_sd3_checkpoint("teacher"))
        return validate_scheduler_grids(actor, teacher, adapter, "default")

    def test_on_grid_timesteps_pass(self, schedulers):
        actor_scheduler, teacher_scheduler = schedulers
        # two rows starting at different SDE windows, as the rollout emits them
        all_timesteps = torch.stack([actor_scheduler.timesteps[:4], actor_scheduler.timesteps[2:6]])

        validate_request_timesteps(all_timesteps, actor_scheduler, teacher_scheduler, "default")

    def test_off_grid_timestep_rejected(self, schedulers):
        actor_scheduler, teacher_scheduler = schedulers
        off_grid = actor_scheduler.timesteps[0] + 0.5
        all_timesteps = torch.tensor([[off_grid]])

        with pytest.raises(ValueError, match="not on the shared scheduler grid") as excinfo:
            validate_request_timesteps(all_timesteps, actor_scheduler, teacher_scheduler, "default")

        # converted from the bare index error rather than surfacing it
        assert isinstance(excinfo.value.__cause__, IndexError | RuntimeError)
        assert "default" in str(excinfo.value)

    def test_index_mismatch_rejected(self, adapter, fake_sd3_checkpoint, diffusion_model_config):
        """Stage 2 stands on its own; it is not merely a corollary of stage 1.

        A 10-step and a 20-step grid share the timestep at sigma 0.9 but place it
        at different indices, which is exactly what would corrupt a replay.
        """
        actor = diffusion_model_config(fake_sd3_checkpoint("student"))
        teacher = diffusion_model_config(
            fake_sd3_checkpoint("teacher"), pipeline=DiffusionPipelineConfig(num_inference_steps=20)
        )
        actor_scheduler = build_cpu_scheduler(actor, adapter)
        teacher_scheduler = build_cpu_scheduler(teacher, adapter)
        shared = set(actor_scheduler.timesteps.tolist()) & set(teacher_scheduler.timesteps.tolist())
        assert shared, "expected the two grids to overlap"

        with pytest.raises(ValueError, match="maps to sigma index"):
            validate_request_timesteps(
                torch.tensor([sorted(shared, reverse=True)]), actor_scheduler, teacher_scheduler, "default"
            )
