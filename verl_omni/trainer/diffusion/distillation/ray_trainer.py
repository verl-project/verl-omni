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
"""Ray trainer for the architecture-neutral distillation runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tracking import Tracking

from verl_omni.pipelines.model_base import DiffusionModelBase, DistributionMatchingModelAdapter
from verl_omni.trainer.diffusion.diffusion_trainer_utils import (
    _to_diffusion_worker_tensordict,
    validate_distillation_config,
)
from verl_omni.trainer.diffusion.distillation.contracts import DistillationPlan, PhaseRequest, TrainerCounters
from verl_omni.trainer.diffusion.distillation.control_plane import DistillationTrainerControlPlane
from verl_omni.trainer.diffusion.distillation.recipes import build_plan_from_config
from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.diffusion_distillation_worker import DiffusionDistillationWorkerGroup

__all__ = ["DistillationRayTrainer"]


def checkpoint_json_value(value: Any):
    """Serialize immutable plan containers independently of insertion order and hash seed."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Unsupported checkpoint fingerprint value: {type(value).__name__}.")


class DistillationBatchProvider:
    """Stateful dataloader adapter implementing phase batch reuse semantics."""

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self._iterator = iter(dataloader)
        self._student_batch: Optional[TensorDict] = None
        self.last_data_proto: Optional[DataProto] = None

    def reset_iterator(self) -> None:
        """Recreate the iterator after a dataloader state restore."""
        self._iterator = iter(self.dataloader)
        self._student_batch = None
        self.last_data_proto = None

    def fresh_batch(self) -> TensorDict:
        """Read the next batch, restarting the dataloader at epoch boundaries."""
        try:
            batch_dict = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch_dict = next(self._iterator)
        batch = DataProto.from_single_dict(batch_dict)
        self.last_data_proto = batch
        return _to_diffusion_worker_tensordict(batch)

    def next(self, phase: PhaseRequest) -> TensorDict:
        """Return a fresh or student-reused batch according to the phase contract."""
        if phase.batch_policy == "reuse_student":
            if self._student_batch is None:
                raise RuntimeError("A reuse_student phase ran before a student batch was cached.")
            return self._student_batch.copy()
        batch = self.fresh_batch()
        if phase.kind == "student":
            self._student_batch = batch.copy()
        return batch


class DistillationRayTrainer(BaseRayDiffusionTrainer):
    """Ray driver over the generic control plane and multi-role worker group."""

    def __init__(
        self,
        config=None,
        tokenizer=None,
        role_worker_mapping=None,
        resource_pool_manager=None,
        ray_worker_group_cls=None,
        processor=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
        *,
        plan: Optional[DistillationPlan] = None,
        capabilities: Optional[frozenset[str]] = None,
        executor: Optional[Any] = None,
        batch_provider: Optional[Any] = None,
        hooks: Optional[Any] = None,
    ) -> None:
        if isinstance(config, DistillationPlan) and plan is None:
            plan = config
            config = None
        if config is not None:
            validate_distillation_config(config)
        if plan is not None and config is not None and capabilities is not None:
            raise ValueError("Pass either an explicit plan or config+capabilities, not both.")

        self._production = config is not None and OmegaConf.select(config, "trainer") is not None
        if self._production:
            role_worker_mapping = role_worker_mapping or {}
            ray_worker_group_cls = ray_worker_group_cls or RayWorkerGroup
            super().__init__(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
                device_name=device_name,
            )
        else:
            self.config = config
            self.tokenizer = tokenizer
            self.processor = processor
            self.role_worker_mapping = role_worker_mapping
            self.resource_pool_manager = resource_pool_manager
            self.ray_worker_group_cls = ray_worker_group_cls
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.collate_fn = collate_fn
            self.train_sampler = train_sampler
            self.device_name = device_name

        if plan is None and config is not None:
            if capabilities is None and self._production:
                model_config: DiffusionModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
                adapter_cls = DiffusionModelBase.get_class(model_config)
                if not issubclass(adapter_cls, DistributionMatchingModelAdapter):
                    raise TypeError(
                        f"{adapter_cls.__name__} must mix in DistributionMatchingModelAdapter for distillation."
                    )
                capabilities = adapter_cls.distillation_capabilities()
            if capabilities is not None:
                plan = build_plan_from_config(config, capabilities)

        self.plan = plan
        self.capabilities = capabilities
        self.executor = executor
        self.batch_provider = batch_provider
        self.hooks = hooks or self
        self._control_plane: Optional[DistillationTrainerControlPlane] = None
        self.distillation_worker_group = None
        self.global_steps = 0
        self._logger = None
        if self._production and self.plan is not None:
            self.validate_runtime_config()

    def validate_runtime_config(self) -> None:
        """Reject unsupported role storage and distributed batch layouts."""
        distribution_matching = self.config.distillation.distribution_matching
        strategy = self.config.actor_rollout_ref.actor.strategy
        if strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(f"Distillation role groups require strategy 'fsdp' or 'fsdp2', got {strategy!r}.")
        if any(group.placement != "colocated" for group in self.plan.role_layout.groups):
            raise NotImplementedError("The current runtime supports colocated role groups only.")
        if any(binding.role == "discriminator" for binding in self.plan.role_layout.bindings):
            raise NotImplementedError("The DMD2 adversarial discriminator data plane lands in PR 4.")
        if distribution_matching.role_storage == "shared_base_adapters":
            model_config = self.config.actor_rollout_ref.model
            lora_rank = model_config.get("lora_rank", model_config.get("lora", {}).get("rank", 0))
            if lora_rank <= 0:
                raise ValueError("shared_base_adapters requires actor_rollout_ref.model.lora_rank > 0.")
            if strategy == "fsdp" and not self.config.actor_rollout_ref.actor.fsdp_config.use_orig_params:
                raise ValueError("shared_base_adapters with FSDP1 requires actor.fsdp_config.use_orig_params=true.")

        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        sequence_parallel_size = self.config.actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size
        if world_size % sequence_parallel_size != 0:
            raise ValueError("Trainer world size must be divisible by the Ulysses sequence-parallel size.")
        data_parallel_size = world_size // sequence_parallel_size
        global_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        if global_batch_size % data_parallel_size != 0:
            raise ValueError(
                f"Distillation batch size {global_batch_size} must be divisible by data-parallel size "
                f"{data_parallel_size}."
            )

    def configure_role_steps(self) -> None:
        """Set the fake-score scheduler horizon in its own optimizer-step units."""
        distribution_matching = self.config.distillation.distribution_matching
        fake_repeats = sum(phase.repeats for phase in self.plan.update_schedule.phases if phase.kind == "fake_score")
        warmup_fake_repeats = sum(
            phase.repeats for phase in self.plan.update_schedule.warmup_phases if phase.kind == "fake_score"
        )
        fake_total_steps = self.total_training_steps * fake_repeats
        fake_total_steps += self.plan.update_schedule.warmup_cycles * warmup_fake_repeats
        with open_dict(distribution_matching.fake_score_optim):
            distribution_matching.fake_score_optim.total_training_steps = fake_total_steps

    def init_workers(self) -> None:
        """Create the colocated multi-role Ray worker group or validate injected fakes."""
        if self.executor is not None or self.batch_provider is not None:
            if self.executor is None or self.batch_provider is None:
                raise ValueError("executor and batch_provider must be supplied together.")
            if self.plan is None:
                raise ValueError("A validated DistillationPlan is required when an executor is bound.")
            return
        if not self._production:
            raise NotImplementedError("Production multi-role workers require the composed diffusion trainer config.")
        if self.plan is None:
            raise ValueError("No distribution-matching architecture adapter produced a DistillationPlan.")
        if Role.Actor not in self.role_worker_mapping:
            raise ValueError("Distillation training requires a Role.Actor worker mapping.")

        self.configure_role_steps()
        self.resource_pool_manager.create_resource_pool()
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        worker_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config,
            plan=self.plan,
        )
        worker_group_kwargs = {"device_name": self.device_name}
        register_timeout = OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout")
        if register_timeout is not None:
            worker_group_kwargs["ray_wait_register_center_timeout"] = register_timeout
        master_port_range = OmegaConf.select(self.config.trainer, "ray_master_port_range")
        if master_port_range is not None:
            worker_group_kwargs["master_port_range"] = list(master_port_range)
        profile_steps = OmegaConf.select(self.config, "global_profiler.steps")
        if profile_steps is not None:
            worker_group_kwargs["profile_steps"] = profile_steps
            if OmegaConf.select(self.config, "global_profiler.tool") == "nsys":
                worker_options = OmegaConf.select(
                    self.config,
                    "global_profiler.global_tool_config.nsys.worker_nsight_options",
                )
                if worker_options is None:
                    raise ValueError("Nsight worker options are required when global profiling uses nsys.")
                worker_group_kwargs["worker_nsight_options"] = OmegaConf.to_container(worker_options)
        self.distillation_worker_group = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls,
            **worker_group_kwargs,
        )
        self.distillation_worker_group.init_model()
        self.executor = DiffusionDistillationWorkerGroup(self.distillation_worker_group)
        self.batch_provider = DistillationBatchProvider(self.train_dataloader)

    def build_control_plane(self) -> DistillationTrainerControlPlane:
        """Construct the pure control plane from the bound worker data plane."""
        self.init_workers()
        assert self.plan is not None
        assert self.executor is not None
        assert self.batch_provider is not None
        self._control_plane = DistillationTrainerControlPlane(
            plan=self.plan,
            executor=self.executor,
            batch_provider=self.batch_provider,
            hooks=self.hooks,
        )
        return self._control_plane

    @property
    def control_plane(self) -> DistillationTrainerControlPlane:
        if self._control_plane is None:
            return self.build_control_plane()
        return self._control_plane

    def after_completed_step(self, counters: TrainerCounters, metrics: dict, executor: Any) -> None:
        """Checkpoint completed student cycles at configured boundaries."""
        self.global_steps = counters.global_step
        if not self._production:
            return
        save_freq = self.config.trainer.save_freq
        if save_freq > 0 and (self.global_steps % save_freq == 0 or self.global_steps >= self.total_training_steps):
            checkpoint_start = time.perf_counter()
            self._save_checkpoint()
            metrics.setdefault("system", {})["perf/checkpoint_s"] = time.perf_counter() - checkpoint_start

    def checkpoint_fingerprint(self) -> str:
        """Hash canonical plan and optimizer configuration for resume validation."""
        payload = {"plan": asdict(self.plan)}
        if self.config is not None and OmegaConf.select(self.config, "distillation.distribution_matching") is not None:
            payload["distribution_matching"] = OmegaConf.to_container(
                self.config.distillation.distribution_matching, resolve=True
            )
            payload["model"] = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True)
            payload["student_optimizer"] = OmegaConf.to_container(
                self.config.actor_rollout_ref.actor.optim, resolve=True
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=checkpoint_json_value).encode()).hexdigest()

    @staticmethod
    def driver_rng_state() -> dict[str, Any]:
        """Capture driver RNG state separately from worker sampling streams."""
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    @staticmethod
    def restore_driver_rng_state(state: dict[str, Any]) -> None:
        """Restore the driver RNG streams saved at a completed cycle."""
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

    def _save_checkpoint(self) -> None:
        """Atomically publish worker, driver, dataloader, and RNG state."""
        if self.executor is None or not hasattr(self.executor, "save_checkpoint"):
            raise RuntimeError("The bound distillation executor cannot save checkpoints.")
        root = os.path.abspath(self.config.trainer.default_local_dir)
        os.makedirs(root, exist_ok=True)
        final_path = os.path.join(root, f"global_step_{self.global_steps}")
        if os.path.exists(final_path):
            raise FileExistsError(f"Refusing to overwrite existing checkpoint {final_path}.")
        temporary_path = tempfile.mkdtemp(prefix=f".global_step_{self.global_steps}_", dir=root)
        try:
            self.executor.save_checkpoint(os.path.join(temporary_path, "workers"), self.global_steps)
            torch.save(self.control_plane.state_dict(), os.path.join(temporary_path, "trainer_state.pt"))
            torch.save(self.train_dataloader.state_dict(), os.path.join(temporary_path, "data.pt"))
            torch.save(self.driver_rng_state(), os.path.join(temporary_path, "rng.pt"))
            with open(os.path.join(temporary_path, "manifest.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "plan_name": self.plan.name,
                        "plan_version": self.plan.version,
                        "global_step": self.global_steps,
                        "export_role": self.plan.export.role,
                        "fingerprint": self.checkpoint_fingerprint(),
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(temporary_path, final_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        latest_tmp = os.path.join(root, ".latest_checkpointed_iteration.tmp")
        with open(latest_tmp, "w", encoding="utf-8") as file:
            file.write(str(self.global_steps))
        os.replace(latest_tmp, os.path.join(root, "latest_checkpointed_iteration.txt"))

    def _load_checkpoint(self) -> int:
        """Restore the last atomically completed distillation cycle."""
        if not self._production or self.config.trainer.resume_mode == "disable":
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("Distillation checkpoint restore from HDFS is not implemented.")
        checkpoint_root = os.path.abspath(self.config.trainer.default_local_dir)
        if self.config.trainer.resume_mode == "auto":
            checkpoint_path = find_latest_ckpt_path(checkpoint_root)
            if checkpoint_path is None:
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            checkpoint_path = os.path.abspath(self.config.trainer.resume_from_path)
        else:
            raise ValueError(f"Unsupported trainer.resume_mode {self.config.trainer.resume_mode!r}.")

        manifest_path = os.path.join(checkpoint_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Incomplete distillation checkpoint: missing {manifest_path}.")
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("plan_name") != self.plan.name or manifest.get("plan_version") != self.plan.version:
            raise ValueError("Checkpoint recipe identity does not match the active DistillationPlan.")
        if manifest.get("fingerprint") != self.checkpoint_fingerprint():
            raise ValueError("Checkpoint distillation plan or optimizer configuration does not match the active run.")

        self.executor.load_checkpoint(os.path.join(checkpoint_path, "workers"))
        trainer_state = torch.load(os.path.join(checkpoint_path, "trainer_state.pt"), weights_only=False)
        self.control_plane.load_state_dict(trainer_state)
        self.train_dataloader.load_state_dict(torch.load(os.path.join(checkpoint_path, "data.pt"), weights_only=False))
        if hasattr(self.batch_provider, "reset_iterator"):
            self.batch_provider.reset_iterator()
        self.restore_driver_rng_state(torch.load(os.path.join(checkpoint_path, "rng.pt"), weights_only=False))
        self.global_steps = self.control_plane.counters.global_step
        return self.global_steps

    @staticmethod
    def flatten_metrics(metrics: dict[str, dict]) -> dict[str, float]:
        """Flatten phase metrics for the existing tracking backends."""
        return {key: value for phase_metrics in metrics.values() for key, value in phase_metrics.items()}

    def profile_workers(self, *, start: bool, step: int) -> None:
        """Start or stop the configured distributed profiler."""
        if self.distillation_worker_group is None:
            return
        if start:
            self.distillation_worker_group.start_profile(role="distillation", profile_step=step)
        else:
            self.distillation_worker_group.stop_profile()

    def fit(self, num_cycles: Optional[int] = None) -> None:
        """Run injected CPU cycles or the production dataloader-backed control plane."""
        if not self._production:
            if num_cycles is None:
                num_cycles = 0
            self.control_plane.run(num_cycles)
            return

        control_plane = self.control_plane
        self._load_checkpoint()
        self._logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        target_steps = (
            self.total_training_steps
            if num_cycles is None
            else min(self.total_training_steps, control_plane.counters.global_step + num_cycles)
        )
        progress_bar = tqdm(
            total=target_steps,
            initial=control_plane.counters.global_step,
            desc="Distillation Training",
        )
        profile_steps = OmegaConf.select(self.config, "global_profiler.steps", default=None)
        while control_plane.counters.global_step < target_steps:
            before_global_step = control_plane.counters.global_step
            profile_step = before_global_step + 1
            do_profile = profile_steps is not None and profile_step in profile_steps
            self.profile_workers(start=do_profile, step=profile_step)
            try:
                control_plane.run_cycle()
            finally:
                if do_profile:
                    self.profile_workers(start=False, step=profile_step)
            self.global_steps = control_plane.counters.global_step
            metrics = self.flatten_metrics(control_plane.metrics)
            metrics["training/global_step"] = float(self.global_steps)
            metrics["training/completed_cycles"] = float(control_plane.counters.completed_cycles)
            self._logger.log(data=metrics, step=self.global_steps)
            if self.global_steps > before_global_step:
                progress_bar.update(1)
            if hasattr(self.train_dataset, "on_batch_end"):
                self.train_dataset.on_batch_end(batch=self.batch_provider.last_data_proto)
        progress_bar.close()
