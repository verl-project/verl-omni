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

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.trainer.diffusion.ray_diffusion_trainer import DistributionMatchingRayTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls

CONFIG_DIR = str(Path(verl_omni.__file__).parent / "trainer" / "config")


def make_config(overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="diffusion_trainer",
            overrides=[
                "algorithm.trainer_type=distribution_matching",
                "algorithm.sample_source=offline",
                "actor_rollout_ref.model.algorithm=dmd2",
                "actor_rollout_ref.model.model_type=diffusion_dmd_model",
                "actor_rollout_ref.model.lora_rank=2",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "data.train_batch_size=8",
                "trainer.total_training_steps=3",
                "trainer.save_freq=-1",
                "trainer.test_freq=-1",
                "trainer.val_before_train=false",
                "trainer.resume_mode=disable",
                *overrides,
            ],
        )


class FakeTracking:
    records = []

    def __init__(self, **kwargs):
        pass

    def log(self, data, step):
        self.records.append((step, data))


class FakeDMDWorker:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.stages = []

    def update_actor(self, data):
        self.stages.append(tu.get_non_tensor_data(data, "dmd_stage", default=None))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return tu.get_tensordict(
            {}, {"metrics": {"dmd/update_applied": outcome, "dmd/skip_nonfinite": 1 - outcome, "loss": 0.25}}
        )


def empty_batch():
    return TensorDict({}, batch_size=[8])


def make_trainer(outcomes):
    trainer = object.__new__(DistributionMatchingRayTrainer)
    trainer.config = make_config()
    trainer.dmd_config = omega_conf_to_dataclass(trainer.config.dmd)
    trainer.global_steps = 0
    trainer.failed = False
    trainer.optimizer_steps = {"student": 0, "fake_score": 0}
    trainer.data_epoch = 0
    trainer.total_training_steps = 3
    trainer.actor_rollout_wg = FakeDMDWorker(outcomes)
    trainer.next_batch = empty_batch
    trainer.export_student = MagicMock()
    return trainer


class TestDMDConfiguration:
    def test_route_and_production_preflight(self):
        config = make_config()
        DistributionMatchingRayTrainer.validate_config(config)
        assert _get_trainer_cls(config) is DistributionMatchingRayTrainer

    @pytest.mark.parametrize(
        "override",
        [
            "algorithm.sample_source=online",
            "actor_rollout_ref.model.algorithm=dmd",
            "actor_rollout_ref.actor.use_kl_loss=true",
            "actor_rollout_ref.actor.use_distill_loss=true",
            "distillation.enabled=true",
            "actor_rollout_ref.actor.ppo_epochs=2",
            "actor_rollout_ref.model.lora_rank=0",
            "actor_rollout_ref.actor.strategy=fsdp",
            "data.train_batch_size=7",
            "actor_rollout_ref.model.model_type=diffusion_model",
            "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
        ],
    )
    def test_unsupported_modes_fail_before_workers(self, override):
        with pytest.raises(ValueError):
            DistributionMatchingRayTrainer.validate_config(make_config([override]))

    def test_fingerprint_is_mapping_order_independent(self):
        trainer = make_trainer([1] * 9)
        first = trainer.configuration_fingerprint()
        from omegaconf import OmegaConf

        fields = OmegaConf.to_container(trainer.config.dmd)
        trainer.config.dmd = dict(reversed(list(fields.items())))
        assert trainer.configuration_fingerprint() == first


class TestDMDCycles:
    def test_normal_and_skipped_cycles_keep_separate_success_counts(self, monkeypatch):
        monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
        FakeTracking.records = []
        trainer = make_trainer([1, 1, 1, 0, 1, 0, 1, 0, 1])
        trainer.fit()
        assert trainer.global_steps == 3
        assert trainer.optimizer_steps == {"student": 2, "fake_score": 4}
        assert trainer.actor_rollout_wg.stages == ["student", "fake_score", "fake_score"] * 3
        assert [step for step, _ in FakeTracking.records] == [1, 2, 3]
        assert "fake_score/0/loss" in FakeTracking.records[0][1]
        assert "fake_score/1/loss" in FakeTracking.records[0][1]
        trainer.export_student.assert_called_once()

    def test_all_skipped_budget_terminates_without_claiming_training_success(self, monkeypatch):
        monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
        trainer = make_trainer([0] * 9)
        with pytest.raises(RuntimeError, match="without successful updates"):
            trainer.fit()
        assert trainer.global_steps == 3
        assert trainer.optimizer_steps == {"student": 0, "fake_score": 0}
        trainer.export_student.assert_not_called()

    def test_partial_exception_is_not_a_numerical_retry(self, monkeypatch):
        monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
        trainer = make_trainer([1, RuntimeError("injected rank failure"), 1])
        with pytest.raises(RuntimeError, match="injected rank failure"):
            trainer.fit()
        assert trainer.global_steps == 0
        assert trainer.optimizer_steps["student"] == 1  # This cannot roll back a real optimizer update.
        assert trainer.actor_rollout_wg.stages == ["student", "fake_score"]
        trainer.export_student.assert_not_called()
        with pytest.raises(RuntimeError, match="must be reconstructed"):
            trainer.fit()
        assert trainer.actor_rollout_wg.stages == ["student", "fake_score"]

    def test_malformed_fractional_outcome_is_not_accepted(self):
        trainer = make_trainer([0.5])
        with pytest.raises(RuntimeError, match="Malformed"):
            trainer.update_stage("student", 0)


class TestDMDCheckpoint:
    def test_counter_mismatch_rejected_before_worker_load(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.n_gpus_per_node = 1
        trainer.config.trainer.resume_mode = "resume_path"
        trainer.config.trainer.resume_from_path = str(tmp_path)
        trainer.actor_rollout_wg = MagicMock()
        torch.save(
            {
                "version": 1,
                "configuration": trainer.configuration_fingerprint(),
                "global_step": 1,
                "optimizer_steps": {"student": 1, "fake_score": 2},
            },
            tmp_path / "trainer.pt",
        )
        (tmp_path / "data.pt").touch()
        actor = tmp_path / "actor"
        actor.mkdir()
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_1_rank_0.pt").touch()
        torch.save(
            {"version": 1, "world_size": 1, "optimizer_steps": {"student": 0, "fake_score": 2}},
            actor / "dmd_state_rank_0.pt",
        )
        with pytest.raises(ValueError, match="counters do not match"):
            trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_not_called()

    def test_missing_shards_are_not_published(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.default_local_dir = str(tmp_path)
        trainer.actor_rollout_wg = MagicMock()
        with pytest.raises(ValueError, match="missing model_world"):
            trainer._save_checkpoint()
        assert not (tmp_path / "global_step_0").exists()
        assert not list(tmp_path.glob(".global_step_*"))

    def test_failed_save_preserves_latest_and_publishes_no_partial_cycle(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.default_local_dir = str(tmp_path)
        trainer.global_steps = 2
        trainer.train_dataloader = MagicMock()
        trainer.actor_rollout_wg = MagicMock()
        trainer.actor_rollout_wg.save_checkpoint.side_effect = RuntimeError("failed shard")
        tracker = tmp_path / "latest_checkpointed_iteration.txt"
        tracker.write_text("1")
        with pytest.raises(RuntimeError, match="failed shard"):
            trainer._save_checkpoint()
        assert tracker.read_text() == "1"
        assert not (tmp_path / "global_step_2").exists()
        assert not list(tmp_path.glob(".global_step_*"))

    def test_old_checkpoint_is_rejected_before_loading_workers(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.resume_mode = "resume_path"
        trainer.config.trainer.resume_from_path = str(tmp_path)
        trainer.actor_rollout_wg = MagicMock()
        (tmp_path / "trainer.pt").touch()
        with pytest.raises(ValueError, match="Incomplete/old"):
            trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
