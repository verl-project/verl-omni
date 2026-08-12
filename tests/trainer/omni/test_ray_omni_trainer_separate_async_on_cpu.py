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
"""CPU tests for ``OmniPPOTrainerSeparateAsync``: registration, the
``parameter_sync_step`` key fix, and the LoRA-aware worker/manager wiring.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from hydra import compose, initialize_config_dir
from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync

_CONFIG_DIR = str((Path(__file__).parents[3] / "verl_omni" / "trainer" / "config").resolve())

_BASE_OVERRIDES = [
    "trainer.v1.trainer_mode=omni_separate_async",
    "data.train_batch_size=4",
    "actor_rollout_ref.actor.ppo_mini_batch_size=4",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.n_gpus_per_node=2",
    "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    "actor_rollout_ref.model.path=/dummy/model",
]


def _compose_config(extra_overrides=()):
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name="omni_trainer", overrides=[*_BASE_OVERRIDES, *extra_overrides])


def test_registered_and_subclasses_separate_async():
    assert get_trainer_cls("omni_separate_async") is OmniPPOTrainerSeparateAsync
    assert issubclass(OmniPPOTrainerSeparateAsync, PPOTrainerSeparateAsync)


def test_parameter_sync_step_follows_validated_key():
    # PPOTrainer.__init__ reads v1.omni_separate_async.parameter_sync_step (absent
    # -> 1); the parent gates syncs on v1.separate_async. Both the trainer AND the
    # ReplayBuffer (which normalizes staleness by this knob) must report the
    # cadence the parent actually runs.
    trainer = OmniPPOTrainerSeparateAsync(_compose_config())
    assert trainer.parameter_sync_step == 4  # upstream separate_async default
    assert trainer.replay_buffer.parameter_sync_step == 4

    trainer = OmniPPOTrainerSeparateAsync(_compose_config(["trainer.v1.separate_async.parameter_sync_step=2"]))
    assert trainer.parameter_sync_step == 2
    assert trainer.replay_buffer.parameter_sync_step == 2


def test_init_tokenizer_wires_omni_model_config():
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    trainer = OmniPPOTrainerSeparateAsync.__new__(OmniPPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"model": {"path": "/dummy"}}})

    fake_cfg = SimpleNamespace(tokenizer="fake_tok", processor="fake_proc")
    with patch(
        "verl_omni.trainer.omni.ray_omni_trainer_separate_async.omega_conf_to_dataclass",
        return_value=fake_cfg,
    ):
        trainer._init_tokenizer()

    assert trainer.tokenizer == "fake_tok"
    assert trainer.processor == "fake_proc"


class TestLoraAwareWiring:
    def test_actor_worker_is_omni_detach_worker(self):
        from verl.trainer.ppo.utils import Role

        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        trainer = OmniPPOTrainerSeparateAsync(_compose_config())
        trainer._init_resource_pool_mgr()
        role = Role.ActorRolloutRef if Role.ActorRolloutRef in trainer.role_worker_mapping else Role.ActorRollout
        modified = trainer.role_worker_mapping[role].__ray_metadata__.modified_class
        assert modified.__name__ == OmniDetachActorWorker.__name__
        assert modified.__module__.startswith("verl_omni"), modified.__module__

    def test_detach_worker_mro_keeps_omni_lora_methods(self):
        # The omni worker must win the MRO: its update_weights carries the
        # adapter-only LoRA send; DetachActorWorker contributes CPU save/restore.
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker
        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        assert OmniDetachActorWorker.update_weights is ActorRolloutRefWorker.update_weights
        assert OmniDetachActorWorker.get_lora_peft_config is ActorRolloutRefWorker.get_lora_peft_config
        assert hasattr(OmniDetachActorWorker, "save_model_to_cpu")
        assert hasattr(OmniDetachActorWorker, "restore_model_from_cpu")

    def test_worker_exposes_upstream_v1_log_prob_methods(self):
        # The upstream v1 trainer calls these on the worker group (ref /
        # non-bypass paths); an unregistered name only fails at Ray dispatch
        # time on GPU. The plain omni worker does not extend verl's v1 worker,
        # so it must define both itself (the detach worker would silently mask
        # a deletion by inheriting verl's copies through DetachActorWorker).
        from verl.single_controller.base.decorator import MAGIC_ATTR

        from verl_omni.workers.engine_workers import ActorRolloutRefWorker
        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        for cls in (ActorRolloutRefWorker, OmniDetachActorWorker):
            for name in ("compute_log_prob", "compute_ref_log_prob"):
                assert hasattr(getattr(cls, name), MAGIC_ATTR), (cls.__name__, name)
        assert "compute_log_prob" in ActorRolloutRefWorker.__dict__
        assert "compute_ref_log_prob" in ActorRolloutRefWorker.__dict__

    def test_init_model_routes_through_omni_worker(self):
        # RFC #320: the omni init_model path is the single source of truth for
        # the detach worker (not verl's parallel implementation).
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker
        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        assert OmniDetachActorWorker.init_model is ActorRolloutRefWorker.init_model

    def test_save_restore_round_trip_routes_through_strategy_handlers(self):
        # RFC #320: DetachActorWorker's CPU save/restore must work on the omni
        # worker. Real handlers need GPU DTensors, so they are faked; the test
        # pins the wiring — save stores the copy handler's output keyed by n,
        # restore feeds it back to the engine module under the fsdp2 tuple
        # protocol, clear drops it.
        from types import SimpleNamespace

        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        worker = object.__new__(OmniDetachActorWorker)
        worker._strategy_handlers = None
        module = object()
        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=module))
        worker.config = SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2"))

        saved = ("sharded_state", "global_spec")
        restored = []
        with patch.object(
            OmniDetachActorWorker,
            "_get_strategy_handlers",
            return_value=(lambda m: saved, lambda m, state, spec: restored.append((m, state, spec))),
        ):
            worker.save_model_to_cpu("step0")
            assert worker.cpu_saved_models["step0"] is saved
            worker.restore_model_from_cpu("step0")
            worker.clear_cpu_model("step0")

        assert restored == [(module, "sharded_state", "global_spec")]
        assert "step0" not in worker.cpu_saved_models

    def test_setup_installs_lora_aware_checkpoint_manager(self):
        from verl.checkpoint_engine import CheckpointEngineRegistry

        from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager

        trainer = OmniPPOTrainerSeparateAsync(_compose_config())
        with (
            patch.object(PPOTrainerSeparateAsync, "_setup"),
            # The nccl backend registers via GPU-only import side effects.
            patch.object(CheckpointEngineRegistry, "get", return_value=MagicMock()),
        ):
            trainer.actor_rollout_wg = MagicMock()
            trainer.standalone_server_manager = MagicMock()
            trainer.standalone_server_manager.get_replicas.return_value = []
            trainer._setup()
        assert isinstance(trainer.standalone_checkpoint_manager, OmniCheckpointEngineManager)
