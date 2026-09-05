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
"""CPU tests for atomic multi-role distillation checkpoint orchestration."""

import os
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.distillation.contracts import PhaseRequest
from verl_omni.trainer.diffusion.distillation.control_plane import (
    DistillationTrainerControlPlane,
    FakeBatchProvider,
    FakeDistillationHooks,
    FakePhaseExecutor,
)
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationBatchProvider, DistillationRayTrainer
from verl_omni.trainer.diffusion.distillation.recipes import build_plan


class CheckpointExecutor(FakePhaseExecutor):
    def __init__(self, *, fail_save=False):
        super().__init__()
        self.role_state = {
            "student": 1.0,
            "fake_score": 2.0,
            "student_ema": 1.5,
            "student_optimizer": 3,
            "fake_optimizer": 4,
            "student_scheduler": 5,
            "fake_scheduler": 6,
        }
        self.fail_save = fail_save
        self.loaded_path = None

    def save_checkpoint(self, local_path, global_step):
        os.makedirs(local_path, exist_ok=True)
        torch.save({"global_step": global_step, "role_state": self.role_state}, os.path.join(local_path, "roles.pt"))
        if self.fail_save:
            raise RuntimeError("injected save failure")

    def load_checkpoint(self, local_path):
        state = torch.load(os.path.join(local_path, "roles.pt"), weights_only=False)
        self.role_state = state["role_state"]
        self.loaded_path = local_path


class StatefulLoader:
    def __init__(self):
        self.position = 0

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


class TestDistillationBatchProvider:
    def test_reuse_student_returns_cached_batch_without_advancing(self):
        batches = [
            {"values": torch.tensor([[1.0]]), "responses": torch.tensor([[9.0]])},
            {"values": torch.tensor([[2.0]]), "responses": torch.tensor([[8.0]])},
        ]
        provider = DistillationBatchProvider(batches)
        student = PhaseRequest("student", 0, 0, "fresh", ("student",), True)
        reused = PhaseRequest("fake_score", 0, 0, "reuse_student", ("fake_score",), False)
        fresh = PhaseRequest("fake_score", 0, 1, "fresh", ("fake_score",), False)
        student_batch = provider.next(student)
        reused_batch = provider.next(reused)
        fresh_batch = provider.next(fresh)
        torch.testing.assert_close(student_batch["values"], reused_batch["values"])
        torch.testing.assert_close(fresh_batch["values"], torch.tensor([[2.0]]))
        assert "responses" not in student_batch

    def test_reuse_before_student_fails(self):
        provider = DistillationBatchProvider([{"values": torch.tensor([[1.0]])}])
        request = PhaseRequest("fake_score", 0, 0, "reuse_student", ("fake_score",), False)
        with pytest.raises(RuntimeError, match="before a student batch"):
            provider.next(request)


class TestDistillationCheckpoint:
    @staticmethod
    def make_trainer(tmp_path, *, fail_save=False):
        plan = build_plan(
            "dmd2",
            {"model_path": "/m", "fake_update_ratio": 1},
            frozenset({"distribution_matching"}),
        )
        executor = CheckpointExecutor(fail_save=fail_save)
        hooks = FakeDistillationHooks()
        control_plane = DistillationTrainerControlPlane(
            plan=plan,
            executor=executor,
            batch_provider=FakeBatchProvider(num_batches=10),
            hooks=hooks,
        )
        control_plane.run_cycle()

        trainer = DistillationRayTrainer(
            plan=plan, executor=executor, batch_provider=FakeBatchProvider(10), hooks=hooks
        )
        trainer._control_plane = control_plane
        trainer._production = True
        trainer.global_steps = control_plane.counters.global_step
        trainer.train_dataloader = StatefulLoader()
        trainer.train_dataloader.position = 7
        trainer.config = OmegaConf.create(
            {
                "trainer": {
                    "default_local_dir": str(tmp_path),
                    "default_hdfs_dir": None,
                    "resume_mode": "auto",
                    "resume_from_path": None,
                }
            }
        )
        return trainer, executor

    def test_round_trip_restores_roles_counters_dataloader_and_rng(self, tmp_path):
        trainer, executor = self.make_trainer(tmp_path)
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        trainer._save_checkpoint()
        expected_random = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = float(torch.rand(()))

        executor.role_state = {"corrupt": True}
        trainer.control_plane.counters.global_step = 99
        trainer.control_plane.counters.optimizer_steps = {"student": 99}
        trainer.train_dataloader.position = 99
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)

        restored_step = trainer._load_checkpoint()
        assert restored_step == 1
        assert trainer.control_plane.counters.global_step == 1
        assert trainer.control_plane.counters.optimizer_steps == {"student": 1, "fake_score": 1}
        assert trainer.control_plane.counters.completed_cycles == 1
        assert trainer.train_dataloader.position == 7
        assert executor.role_state["student_optimizer"] == 3
        assert executor.role_state["fake_scheduler"] == 6
        assert random.random() == expected_random
        assert float(np.random.random()) == expected_numpy
        assert float(torch.rand(())) == expected_torch

        checkpoint = tmp_path / "global_step_1"
        assert (checkpoint / "manifest.json").is_file()
        assert (checkpoint / "trainer_state.pt").is_file()
        assert (checkpoint / "data.pt").is_file()
        assert (checkpoint / "rng.pt").is_file()
        assert executor.loaded_path == str(checkpoint / "workers")

    def test_failed_save_never_publishes_a_checkpoint(self, tmp_path):
        trainer, _ = self.make_trainer(tmp_path, fail_save=True)
        with pytest.raises(RuntimeError, match="injected save failure"):
            trainer._save_checkpoint()
        assert not (tmp_path / "global_step_1").exists()
        assert not list(tmp_path.glob(".global_step_1_*"))
        assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()

    def test_equivalent_plan_mappings_have_identical_fingerprints(self, tmp_path):
        trainer, _ = self.make_trainer(tmp_path)
        fingerprint = trainer.checkpoint_fingerprint()
        trainer.plan = replace(trainer.plan, objective=dict(reversed(list(trainer.plan.objective.items()))))
        assert trainer.checkpoint_fingerprint() == fingerprint

    def test_changed_plan_is_rejected_before_worker_restore(self, tmp_path):
        trainer, executor = self.make_trainer(tmp_path)
        trainer._save_checkpoint()
        trainer.plan = build_plan(
            "dmd2",
            {"model_path": "/m", "fake_update_ratio": 2},
            frozenset({"distribution_matching"}),
        )
        with pytest.raises(ValueError, match="does not match the active run"):
            trainer._load_checkpoint()
        assert executor.loaded_path is None

    def test_incomplete_checkpoint_is_rejected(self, tmp_path):
        trainer, _ = self.make_trainer(tmp_path)
        incomplete = tmp_path / "global_step_1"
        incomplete.mkdir()
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")
        with pytest.raises(FileNotFoundError, match="Incomplete distillation checkpoint"):
            trainer._load_checkpoint()
