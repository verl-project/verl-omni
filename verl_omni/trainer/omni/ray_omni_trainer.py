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
"""Omni Ray trainer implementations."""

from __future__ import annotations

import logging
import os
import uuid
import warnings
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import marked_timer
from verl.utils.fs import local_mkdir_safe
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tracking import Tracking

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl_omni.trainer.diffusion.diffusion_trainer_utils import NoOpCheckpointManager
from verl_omni.trainer.omni.omni_algos import (
    get_omni_loss_fn,
)
from verl_omni.utils.dataset.offline_mllm_dpo_dataset import get_batch_modality
from verl_omni.utils.metrics_utils import GroupedMetricMean
from verl_omni.workers.config import OmniModelConfig

sys_logger = logging.getLogger(__name__)

__all__ = ["OmniPPOTrainerSync", "OmniDirectPreferenceRayTrainer"]


@register_trainer("omni_sync")
class OmniPPOTrainerSync(PPOTrainerSync):
    """``PPOTrainerSync`` subclass that wires tokenizer/processor from ``OmniModelConfig``."""

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor


class OmniDirectPreferenceRayTrainer:
    """Standalone Omni AR direct-preference Ray trainer.

    Supports ref-in-actor (LoRA base weights as reference) and an optional
    external ref worker when ``lora_rank == 0``.
    """

    def __init__(
        self,
        config,
        tokenizer=None,
        role_worker_mapping: dict[Role, type] | None = None,
        resource_pool_manager: ResourcePoolManager | None = None,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.get("hybrid_engine", True)
        self.role_worker_mapping = role_worker_mapping or {}
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else OmegaConf.select(self.config, "trainer.device", default=None)
        self.checkpoint_manager = None

        self.is_offline = config.algorithm.get("sample_source", "online") == "offline"
        if not self.is_offline:
            raise NotImplementedError(
                "OmniDirectPreferenceRayTrainer currently supports algorithm.sample_source=offline only."
            )
        if config.actor_rollout_ref.model.get("model_type", "language_model") != "omni_model":
            raise ValueError("OmniDirectPreferenceRayTrainer requires actor_rollout_ref.model.model_type=omni_model.")
        loss_mode = config.actor_rollout_ref.actor.omni_loss.loss_mode
        if loss_mode != "dpo":
            raise NotImplementedError("OmniDirectPreferenceRayTrainer currently supports omni_loss.loss_mode=dpo only.")
        self.use_reference_policy = True
        self._has_old_adapter = "old" in tuple(
            config.actor_rollout_ref.model.get("policy_state_adapters", ("default",))
        )
        if self._has_old_adapter:
            raise NotImplementedError("OmniDirectPreferenceRayTrainer does not support old-policy adapters yet.")
        self._loss_fn = get_omni_loss_fn(loss_mode)
        self.global_batch_size = self.config.data.train_batch_size
        self.ref_in_actor = self._infer_ref_in_actor()
        self.use_rm = False

        if train_dataset is not None or self.config.data.get("train_files", None) is not None:
            self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler, val_sampler)

    def _infer_ref_in_actor(self) -> bool:
        model_cfg = self.config.actor_rollout_ref.model
        lora_rank = model_cfg.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = model_cfg.get("lora_rank", 0)
        return lora_rank > 0 or model_cfg.get("lora_adapter_path") is not None

    def _create_dataloader(
        self,
        train_dataset,
        val_dataset,
        collate_fn,
        train_sampler: Optional[Sampler],
        val_sampler: Optional[Sampler] = None,
    ) -> None:
        """Create stateful train/validation dataloaders for offline omni DPO."""
        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler, get_collate_fn

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler_config = self.config.data.get("train_sampler", self.config.data.get("sampler", None))
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset, sampler_config=train_sampler_config)
        if collate_fn is None:
            collate_fn = get_collate_fn(self.config.data)

        num_workers = self.config.data["dataloader_num_workers"]
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        if val_sampler is None:
            val_sampler_config = self.config.data.get("val_sampler", None)
            if val_sampler_config is not None:
                val_sampler = create_rl_sampler(self.config.data, self.val_dataset, sampler_config=val_sampler_config)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False if val_sampler is not None else self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
            sampler=val_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as exc:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {exc}")

    def init_workers(self) -> None:
        """Initialize actor/ref workers for offline omni direct-preference training."""
        self._init_colocated_workers()
        self.reward_loop_manager = None
        self.llm_server_manager = None
        self.enable_agent_reward_loop = False
        self.checkpoint_manager = NoOpCheckpointManager()

    def _init_colocated_workers(self):
        """Create Ray pools and colocated actor/ref worker groups."""
        if self.resource_pool_manager is None:
            raise ValueError("resource_pool_manager must be provided before initializing workers.")
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if Role.Actor in self.role_worker_mapping:
            actor_role = Role.Actor
        elif Role.ActorRolloutRef in self.role_worker_mapping:
            actor_role = Role.ActorRolloutRef
        else:
            actor_role = Role.ActorRollout
        if self.hybrid_engine or actor_role == Role.Actor:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config, "global_profiler.steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                worker_nsight_options = OmegaConf.select(
                    self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                )
                assert worker_nsight_options is not None, (
                    "global_profiler.global_tool_config.nsys.worker_nsight_options must be set "
                    "when using nsys with global_profiler.steps"
                )
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(worker_nsight_options)
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg
        return actor_rollout_resource_pool

    def _save_checkpoint(self) -> None:
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        actor_checkpoint_cfg = self.config.actor_rollout_ref.actor.checkpoint
        if (
            hasattr(actor_checkpoint_cfg, "async_save")
            and actor_checkpoint_cfg.async_save
            or "async_save" in actor_checkpoint_cfg
            and actor_checkpoint_cfg["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self) -> int:
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        return self.global_steps

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for actor/ref workers if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy and not self.ref_in_actor:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for actor/ref workers if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy and not self.ref_in_actor:
                self.ref_policy_wg.stop_profile()

    def _shutdown_dataloaders(self) -> None:
        for attr in ("train_dataloader", "val_dataloader"):
            loader = getattr(self, attr, None)
            if loader is None:
                continue
            iterator = getattr(loader, "_iterator", None) or getattr(loader, "_DataLoader__iterator", None)
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    sys_logger.debug("Ignoring error shutting down %s workers: %s", attr, exc)
            try:
                loader._iterator = None
            except Exception:
                pass

    def _batch_dict_to_dataproto(self, batch_dict: dict, meta_info: dict) -> DataProto:
        tensor_dict = dict(batch_dict)
        non_tensor_dict = {key: value for key, value in meta_info.items() if key not in tensor_dict}
        if isinstance(tensor_dict.get("multi_modal_inputs"), dict):
            non_tensor_dict["multi_modal_inputs"] = tensor_dict.pop("multi_modal_inputs")
        data = tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)
        return DataProto.from_tensordict(data)

    def _omni_dpo_meta_info(self, *, global_batch_size: int | None = None) -> dict:
        if global_batch_size is None:
            global_batch_size = self.global_batch_size
        micro_batch_size_per_gpu = self.config.data.get("micro_batch_size_per_gpu", None)
        if micro_batch_size_per_gpu is None:
            micro_batch_size_per_gpu = self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", None)
        if self.config.algorithm.get("paired_preference", False) and micro_batch_size_per_gpu is not None:
            micro_batch_size_per_gpu = min(micro_batch_size_per_gpu, global_batch_size)
        micro_batch_size_per_gpu = self._expanded_preference_batch_size(micro_batch_size_per_gpu)
        return {
            "use_remove_padding": self.config.actor_rollout_ref.model.get("use_remove_padding", True),
            "use_dynamic_bsz": self.config.data.get("use_dynamic_bsz", False),
            "max_token_len_per_gpu": self.config.data.get("max_token_len_per_gpu", None),
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self._expanded_preference_batch_size(global_batch_size),
            "pad_mode": DatasetPadMode(self.config.data.get("pad_mode", "no_padding")),
            "pad_token_id": self.config.actor_rollout_ref.model.get("pad_token_id", 0),
        }

    def _expanded_preference_batch_size(self, batch_size: int | None) -> int | None:
        if batch_size is None:
            return None
        paired = self.config.algorithm.get("paired_preference", False)
        return batch_size * 2 if paired else batch_size * self.config.actor_rollout_ref.rollout.n

    def _logical_preference_batch_size(self, batch_dict: dict) -> int:
        batch_value = batch_dict.get("input_ids")
        if batch_value is None:
            batch_value = batch_dict.get("labels")
        if batch_value is None:
            raise KeyError("Omni DPO batch must contain `input_ids` or `labels` to infer batch size.")
        expanded_batch_size = len(batch_value)
        if self.config.algorithm.get("paired_preference", False):
            if expanded_batch_size % 2 != 0:
                raise ValueError("Omni DPO validation batch must contain adjacent chosen/rejected rows.")
            return expanded_batch_size // 2
        rollout_n = self.config.actor_rollout_ref.rollout.n
        return expanded_batch_size // rollout_n

    def _infer_reference_policy(self, batch: DataProto) -> Optional[DataProto]:
        """Compute reference-policy log-probs for batch."""
        batch_td = batch.to_tensordict()
        metadata = {
            "compute_loss": False,
            "average_log_prob": self.config.actor_rollout_ref.actor.omni_loss.average_log_prob,
            "use_dynamic_bsz": False,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        else:
            output = self.ref_policy_wg.infer_ref_batch(batch_td)
        if output is None:
            return None

        ref_logps = tu.get_tensordict({"ref_log_prob": tu.get(output, "log_probs").float()})
        return DataProto.from_tensordict(ref_logps)

    def _infer_actor_policy(self, batch: DataProto):
        """Compute actor log-probs without running a training update."""

        batch_td = batch.to_tensordict()
        tu.assign_non_tensor(
            batch_td,
            compute_loss=False,
            average_log_prob=self.config.actor_rollout_ref.actor.omni_loss.average_log_prob,
            use_dynamic_bsz=False,
        )
        output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        if output is None:
            return None
        return {"log_probs": tu.get(output, "log_probs").float()}

    def _validate(self):
        """Evaluate held-out offline omni DPO pairs with reward accuracy and margin."""

        if not self.is_offline:
            raise NotImplementedError("OmniDirectPreferenceRayTrainer validation currently supports offline DPO only.")

        val_dataloader = self.val_dataloader
        loss_fn = self._loss_fn
        metric_keys = ("dpo_loss", "reward_accuracy", "reward_margin", "chosen_rewards", "rejected_rewards")
        metric_aggregator = GroupedMetricMean(metric_keys=metric_keys, group_attribute="modality")

        with torch.no_grad():
            for batch_dict in val_dataloader:
                modality = get_batch_modality(batch_dict)
                logical_batch_size = self._logical_preference_batch_size(batch_dict)
                meta_info = self._omni_dpo_meta_info(global_batch_size=logical_batch_size)
                batch = self._batch_dict_to_dataproto(batch_dict, meta_info)

                ref_infer_res = self._infer_reference_policy(batch)
                if ref_infer_res is None:
                    raise RuntimeError("Reference policy returned no log-probs during omni DPO validation.")
                batch = batch.union(ref_infer_res)

                policy_output = self._infer_actor_policy(batch)
                if policy_output is None:
                    raise RuntimeError("Actor policy returned no log-probs during omni DPO validation.")

                dpo_result = loss_fn(
                    config=self.config.actor_rollout_ref.actor,
                    model_output=policy_output,
                    data=batch.to_tensordict(),
                )
                logical_count = len(batch.batch) // 2
                metric_aggregator.update(
                    dpo_result.metrics,
                    weight=logical_count,
                    attributes={"modality": modality},
                )

        return metric_aggregator.to_prefixed_dict("val")

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch_td = batch.to_tensordict()

        ppo_mini_batch_size = self._expanded_preference_batch_size(
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        )
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        if self.config.algorithm.get("paired_preference", False) and shuffle:
            message = (
                "Shuffle is not supported for omni direct preference because chosen/rejected "
                "branches must stay grouped by preference pair. Setting shuffle to False."
            )
            sys_logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            shuffle = False

        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        if "metrics" in actor_output and hasattr(actor_output["metrics"], "to_dict"):
            actor_output = actor_output["metrics"].to_dict()
        else:
            actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if (actor_mfu := actor_output.pop("actor/mfu", None)) is not None:
            actor_output["perf/mfu/actor"] = actor_mfu
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def fit(self):
        """Offline omni DPO loop with SFT-style TensorDict batch construction."""
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracking.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dataloaders()
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        meta_info = self._omni_dpo_meta_info()

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch = self._batch_dict_to_dataproto(batch_dict, meta_info)
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    reward_tensor = batch.batch["sample_level_scores"]
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["sample_level_scores"] = reward_tensor
                    batch.batch["sample_level_rewards"] = batch.batch["sample_level_scores"]

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_infer_res = self._infer_reference_policy(batch)
                            if ref_infer_res is not None:
                                batch = batch.union(ref_infer_res)

                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(self.global_steps)

                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics_diffusion(batch=batch))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                num_images = batch.batch["sample_level_scores"].shape[0]
                metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
                metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                tracking.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._shutdown_dataloaders()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
        self._shutdown_dataloaders()
