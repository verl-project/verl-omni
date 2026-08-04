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
import uuid
import warnings
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl.protocol import DataProto
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tracking import Tracking

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    BaseRayDiffusionTrainer,
    DirectPreferenceRayTrainer,
)
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


class OmniDirectPreferenceRayTrainer(DirectPreferenceRayTrainer):
    """Omni AR direct-preference trainer on the shared Ray preference loop.

    Supports ref-in-actor (LoRA base weights as reference) and an optional
    external ref worker when ``lora_rank == 0``.
    """

    def __init__(self, config, *args, **kwargs):
        BaseRayDiffusionTrainer.__init__(self, config, *args, **kwargs)
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
            return super()._validate()

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
