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
"""Offline DPO trainer for omni-model preference data."""

import uuid
from functools import partial
from pprint import pprint
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tracking import Tracking
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


class NoOpCheckpointManager:
    """Checkpoint-engine facade used when offline training has no rollout replicas."""

    def update_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sleep_replicas(self) -> None:
        return None


def omni_dpo_loss(config, model_output, data, dp_group=None):
    """Compute sequence-level DPO loss for adjacent chosen/rejected pairs."""
    del dp_group

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    data = data.select("response_mask", "ref_log_prob").to_padded_tensor()
    response_mask = data["response_mask"].to(bool)
    ref_log_prob = data["ref_log_prob"]

    if log_prob.shape[0] % 2 != 0:
        raise ValueError(f"Offline DPO expects an even number of chosen/rejected samples, got {log_prob.shape[0]}.")

    policy_logps = (log_prob * response_mask).sum(dim=-1)
    ref_logps = (ref_log_prob * response_mask).sum(dim=-1)

    policy_chosen_logps = policy_logps[0::2]
    policy_rejected_logps = policy_logps[1::2]
    ref_chosen_logps = ref_logps[0::2]
    ref_rejected_logps = ref_logps[1::2]

    logits = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    beta = config.policy_loss.get("dpo_beta", 0.1)
    losses = -F.logsigmoid(beta * logits)
    loss = losses.mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "dpo_loss": Metric(value=loss.detach(), aggregation=AggregationType.MEAN),
        "dpo_accuracy": Metric(
            value=(chosen_rewards > rejected_rewards).float().mean(), aggregation=AggregationType.MEAN
        ),
        "dpo_margin": Metric(value=(chosen_rewards - rejected_rewards).mean(), aggregation=AggregationType.MEAN),
        "chosen_logps": Metric(value=policy_chosen_logps.detach().mean(), aggregation=AggregationType.MEAN),
        "rejected_logps": Metric(value=policy_rejected_logps.detach().mean(), aggregation=AggregationType.MEAN),
    }
    return loss, metrics


class RayOmniDPOTrainer(RayPPOTrainer):
    """Actor-only offline DPO trainer for omni models.

    The dataset already contains chosen/rejected responses. The trainer
    recomputes reference log-probs from the base model and updates only the
    actor; it never starts rollout replicas or vLLM servers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rm = False
        self.use_critic = False
        self.use_teacher_policy = False
        self.use_reference_policy = True
        if not self.ref_in_actor:
            raise ValueError("Offline omni DPO currently requires LoRA so the reference policy can run in actor.")

    def init_workers(self):
        """Initialize only actor workers for offline DPO."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if Role.Actor not in self.role_worker_mapping:
            raise ValueError("Offline omni DPO requires Role.Actor in role_worker_mapping.")

        actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(Role.Actor),
        )
        self.resource_pool_to_cls[actor_resource_pool][str(Role.Actor)] = actor_cls

        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
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

        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        self.actor_rollout_wg = all_wg[str(Role.Actor)]
        self.actor_rollout_wg.init_model()
        self.actor_rollout_wg.set_loss_fn(partial(omni_dpo_loss, config=self.config.actor_rollout_ref.actor))
        self.ref_policy_wg = self.actor_rollout_wg
        self.checkpoint_manager = NoOpCheckpointManager()

    def _validate(self):
        print("Skipping validation generation because offline rollout is disabled.")
        return {"val/offline/skipped": 1.0}

    def _start_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.stop_profile()

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.get("multi_turn", {}).get("enable", False)
        if rollout_config.get("temperature", None) is not None:
            batch.meta_info["temperature"] = rollout_config.temperature

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        pair_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = pair_mini_batch_size * 2
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed

        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": False},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if (actor_mfu := actor_output.pop("actor/mfu", None)) is not None:
            actor_output["perf/mfu/actor"] = actor_mfu
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def fit(self):
        """Run actor-only offline DPO training."""
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        logger = Tracking(
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
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
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

                batch = DataProto.from_single_dict(batch_dict)
                if "response_mask" not in batch.batch:
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    if self.config.trainer.get("balance_batch", False):
                        print("Skipping balance_batch for offline DPO to preserve chosen/rejected adjacency.")

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    images_seqlens = []
                    for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                        if isinstance(multi_modal_input, dict) and "images_seqlens" in multi_modal_input:
                            images_seqlens.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens

                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        ref_log_prob = self._compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

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
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

        self._shutdown_dump_executor()
