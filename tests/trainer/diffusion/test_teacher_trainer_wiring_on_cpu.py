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
"""CPU tests for wiring the diffusion teacher into the trainer (RFC #293)."""

import os

import pytest
from hydra import compose, initialize_config_dir
from verl.trainer.ppo.ray_trainer import Role

import verl_omni
from verl_omni.trainer.main_diffusion import TaskRunner
from verl_omni.workers.teacher_workers import DiffusionTeacherWorker

CONFIG_DIR = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config")

TEACHER_OVERRIDES = [
    "actor_rollout_ref.teacher.enabled=true",
    "+actor_rollout_ref.teacher.models.default.model.path=/ckpt/teacher",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
]


@pytest.fixture
def task_runner(monkeypatch):
    """A TaskRunner with ray.remote neutered, so registration is observable on CPU."""
    from verl_omni.trainer import main_diffusion

    monkeypatch.setattr(main_diffusion.ray, "remote", lambda cls: cls)
    runner = TaskRunner()
    runner.role_worker_mapping = {}
    runner.mapping = {}
    return runner


def compose_config(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=list(overrides))


class TestTaskRunnerRegistration:
    def test_teacher_registered_to_global_pool(self, task_runner):
        task_runner.add_teacher_worker(compose_config(*TEACHER_OVERRIDES))

        assert task_runner.role_worker_mapping[Role.TeacherModel] is DiffusionTeacherWorker
        # colocated: the teacher shares the actor's pool, which is pure config
        assert task_runner.mapping[Role.TeacherModel] == "global_pool"

    def test_default_off_registers_nothing(self, task_runner):
        task_runner.add_teacher_worker(compose_config())

        assert Role.TeacherModel not in task_runner.role_worker_mapping
        assert task_runner.mapping == {}

    def test_role_name_matches_worker_prefix(self):
        """The spawn prefix is str(Role), so the teacher must not collide with a peer."""
        assert str(Role.TeacherModel) == "teacher"
        assert str(Role.TeacherModel) not in {str(Role.ActorRollout), str(Role.ActorRolloutRef), str(Role.RefPolicy)}


class FakeResourcePool:
    pass


class FakeResourcePoolManager:
    def __init__(self, pool):
        self.pool = pool
        self.resource_pool_dict = {"global_pool": pool}

    def create_resource_pool(self):
        pass

    def get_resource_pool(self, role):
        return self.pool


class SpawnReached(Exception):
    """Raised in place of Ray spawning, which needs a live cluster."""


class TestColocatedSpawnEntry:
    """The teacher joins the fused class_dict rather than getting its own group.

    A standalone pool must *not* go through ``create_colocated_worker_cls`` --
    both upstream and this file's own comment say per-role pools need separate
    worker groups -- which is why placement is the next runtime PR, not a flag.
    """

    @staticmethod
    def assemble(monkeypatch, role_worker_mapping):
        """Run _init_colocated_workers up to the point where Ray would be needed."""
        from verl_omni.trainer.diffusion import ray_diffusion_trainer
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

        def stop_before_spawn(class_dict):
            raise SpawnReached

        monkeypatch.setattr(ray_diffusion_trainer, "create_colocated_worker_cls", stop_before_spawn)

        pool = FakeResourcePool()
        trainer = object.__new__(PolicyGradientRayTrainer)
        trainer.config = compose_config(*TEACHER_OVERRIDES)
        trainer.role_worker_mapping = role_worker_mapping
        trainer.resource_pool_manager = FakeResourcePoolManager(pool)
        trainer.hybrid_engine = True
        trainer.use_reference_policy = False
        trainer.device_name = "cpu"

        with pytest.raises(SpawnReached):
            trainer._init_colocated_workers()
        return trainer.resource_pool_to_cls[pool]

    def test_teacher_added_to_class_dict(self, monkeypatch):
        class_dict = self.assemble(
            monkeypatch,
            {Role.ActorRollout: DiffusionTeacherWorker, Role.TeacherModel: DiffusionTeacherWorker},
        )

        assert set(class_dict) == {str(Role.ActorRollout), str(Role.TeacherModel)}

    def test_default_off_leaves_class_dict_unchanged(self, monkeypatch):
        class_dict = self.assemble(monkeypatch, {Role.ActorRollout: DiffusionTeacherWorker})

        assert set(class_dict) == {str(Role.ActorRollout)}


class FakeHandle:
    """Stands in for a Ray actor handle."""


class FakeSpawnedGroup:
    """One prefixed view. All views share the same handle list, as verl's spawn does."""

    def __init__(self, handles, calls, name, fails=False):
        self._workers = handles
        self.calls = calls
        self.name = name
        self.fails = fails

    def init_model(self):
        self.calls.append(self.name)
        if self.fails:
            raise RuntimeError("teacher ran out of memory at init")


class FakeWorkerGroupCls:
    def __init__(self, handles, calls, failing=()):
        self.handles = handles
        self.calls = calls
        self.failing = set(failing)
        self.spawned = []

    def __call__(self, resource_pool, ray_cls_with_init, **kwargs):
        return self

    def spawn(self, prefix_set):
        groups = {
            prefix: FakeSpawnedGroup(self.handles, self.calls, prefix, fails=prefix in self.failing)
            for prefix in prefix_set
        }
        self.spawned.extend(groups.values())
        return groups


class FakeManager:
    def __init__(self):
        self.scored = []

    def compute_teacher_outputs(self, batch):
        self.scored.append(batch)
        return batch


class TestInitOrderingAndTeardown:
    @staticmethod
    def build_trainer(monkeypatch, role_worker_mapping, failing=(), kills=None):
        from verl_omni.trainer.diffusion import ray_diffusion_trainer
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

        monkeypatch.setattr(ray_diffusion_trainer, "create_colocated_worker_cls", lambda class_dict: object())
        if kills is not None:
            monkeypatch.setattr(ray_diffusion_trainer.ray, "kill", kills.append)

        calls = []
        handles = [FakeHandle(), FakeHandle()]
        pool = FakeResourcePool()
        trainer = object.__new__(PolicyGradientRayTrainer)
        trainer.config = compose_config(*TEACHER_OVERRIDES)
        trainer.role_worker_mapping = role_worker_mapping
        trainer.resource_pool_manager = FakeResourcePoolManager(pool)
        trainer.hybrid_engine = True
        trainer.use_reference_policy = False
        trainer.ref_in_actor = False
        trainer.device_name = "cpu"
        trainer.teacher_wg = None
        trainer.teacher_manager = None
        trainer.ray_worker_group_cls = FakeWorkerGroupCls(handles, calls, failing=failing)
        monkeypatch.setattr(PolicyGradientRayTrainer, "_build_teacher_manager", lambda self: FakeManager())
        return trainer, calls, handles

    def test_init_order(self, monkeypatch):
        """actor/rollout must stay last: it is created last so vLLM can size its KV cache."""
        trainer, calls, _ = self.build_trainer(
            monkeypatch, {Role.ActorRollout: DiffusionTeacherWorker, Role.TeacherModel: DiffusionTeacherWorker}
        )

        trainer._init_colocated_workers()

        assert calls == [str(Role.TeacherModel), str(Role.ActorRollout)]
        assert trainer.teacher_manager is not None

    def test_default_off_no_manager(self, monkeypatch):
        trainer, calls, _ = self.build_trainer(monkeypatch, {Role.ActorRollout: DiffusionTeacherWorker})

        trainer._init_colocated_workers()

        assert calls == [str(Role.ActorRollout)]
        assert trainer.teacher_manager is None
        assert trainer.teacher_wg is None

    def test_teacher_init_failure_teardown(self, monkeypatch):
        """Spawned views share one handle list, so each actor must be killed exactly once."""
        kills = []
        trainer, calls, handles = self.build_trainer(
            monkeypatch,
            {Role.ActorRollout: DiffusionTeacherWorker, Role.TeacherModel: DiffusionTeacherWorker},
            failing=(str(Role.TeacherModel),),
            kills=kills,
        )

        with pytest.raises(RuntimeError, match="ran out of memory"):
            trainer._init_colocated_workers()

        assert kills == handles  # each handle once, despite two views sharing the list
        assert calls == [str(Role.TeacherModel)]  # actor/rollout never started
