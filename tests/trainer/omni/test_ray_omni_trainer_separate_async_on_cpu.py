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
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, get_trainer_cls
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync

_CONFIG_DIR = str((Path(__file__).parents[3] / "verl_omni" / "trainer" / "config").resolve())

_BASE_OVERRIDES = [
    "trainer.v1.trainer_mode=omni_separate_async",
    # Parent asserts train_batch_size == parameter_sync_step * ppo_mini_batch_size.
    "data.train_batch_size=16",
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
    # -> 1); the parent gates syncs on v1.separate_async. The trainer must report
    # the cadence the parent actually runs. ReplayBuffer does not read this knob.
    trainer = OmniPPOTrainerSeparateAsync(_compose_config())
    assert trainer.parameter_sync_step == 4  # upstream separate_async default

    trainer = OmniPPOTrainerSeparateAsync(
        _compose_config(
            [
                "trainer.v1.separate_async.parameter_sync_step=2",
                "data.train_batch_size=8",
            ]
        )
    )
    assert trainer.parameter_sync_step == 2


def test_omni_separate_async_inherits_verl_tq_checkpoint_and_hybrid_switch():
    # Omni does not override save/load; after TransferQueue 0.1.9 the pin's
    # TQ checkpoint path (verl#7037) is live. Dynamic GPU lending on v1 is
    # hybrid_rollout.enable_switch (verl#7373), default off.
    assert OmniPPOTrainerSeparateAsync._save_checkpoint is PPOTrainer._save_checkpoint
    assert OmniPPOTrainerSeparateAsync._load_checkpoint is PPOTrainer._load_checkpoint
    trainer = OmniPPOTrainerSeparateAsync(_compose_config())
    assert trainer.hybrid_rollout_config.enable_switch is False


def test_pin_ships_dynamic_resource_controller():
    # verl#6556 is in the pin; v1 trainers do not construct it (see RFC #52).
    from verl.experimental.fully_async_policy.dynamic_schedule import DynamicResourceController, build_policy

    policy = build_policy("default", deactivate_ratio=0.3, only_hybrid=False)
    assert DynamicResourceController is not None
    assert policy is not None


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
        # Save stores the copy handler's output keyed by n, restore feeds it back
        # to the engine module under the fsdp2 tuple protocol, clear drops it.
        # The omni worker wraps the fsdp2 save handler so snapshots own their
        # storage: verl's helper returns views of CPU-resident (param_offload)
        # parameters, and the decoupled-PPO dance keeps several slots live at once.
        from types import SimpleNamespace

        import torch
        from verl.utils import fsdp_utils

        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        worker = object.__new__(OmniDetachActorWorker)
        worker._strategy_handlers = None
        module = object()
        live_shard = torch.zeros(2)
        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=module))
        worker.config = SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2"))

        restored = []

        def restore(m, state, spec):
            restored.append((m, {name: tensor.clone() for name, tensor in state.items()}, spec))

        with (
            patch.object(fsdp_utils, "fsdp2_sharded_save_to_cpu", lambda m: ({"adapter": live_shard}, "global_spec")),
            patch.object(fsdp_utils, "fsdp2_sharded_load_from_cpu", restore),
        ):
            worker.save_model_to_cpu(0)
            live_shard.add_(1.0)  # a later local update must not rewrite the snapshot through the alias
            worker.restore_model_from_cpu(0)
            worker.clear_cpu_model(0)

        assert len(restored) == 1
        assert restored[0][0] is module
        assert torch.equal(restored[0][1]["adapter"], torch.zeros(2))
        assert restored[0][2] == "global_spec"
        assert 0 not in worker.cpu_saved_models

    def test_strategy_handlers_wrap_only_the_aliasing_save(self):
        # fsdp1/megatron save helpers already copy; wrapping them would double
        # the CPU footprint of every snapshot.
        from types import SimpleNamespace

        from verl.utils import fsdp_utils

        from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

        worker = object.__new__(OmniDetachActorWorker)
        worker._strategy_handlers = None
        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=object()))
        worker.config = SimpleNamespace(actor=SimpleNamespace(strategy="fsdp"))

        copy_handler, restore_handler = worker._get_strategy_handlers()
        assert copy_handler is fsdp_utils.fsdp1_sharded_save_to_cpu
        assert restore_handler is fsdp_utils.fsdp1_sharded_load_from_cpu

        worker = object.__new__(OmniDetachActorWorker)
        worker._strategy_handlers = None
        worker.actor = SimpleNamespace(engine=SimpleNamespace(module=object()))
        worker.config = SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2"))

        copy_handler, restore_handler = worker._get_strategy_handlers()
        assert copy_handler is not fsdp_utils.fsdp2_sharded_save_to_cpu
        assert copy_handler.__wrapped__ is fsdp_utils.fsdp2_sharded_save_to_cpu
        assert restore_handler is fsdp_utils.fsdp2_sharded_load_from_cpu

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
