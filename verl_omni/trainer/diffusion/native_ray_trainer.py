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
"""Diffusion trainer for native actor-side rollout and joint policy updates.

Reuses ``PolicyGradientRayTrainer``'s worker init, reward, advantage, checkpoint, and
logging helpers, but replaces the vLLM/AgentLoop rollout with a native worker
``generate`` call that samples on the live FSDP actor module (via a flat bf16 replica) and
re-anchors ``old_logp`` there. Selected by ``algorithm.trainer_type=unigrpo`` from
``main_diffusion._get_trainer_cls``.
"""

from __future__ import annotations

import uuid
from pprint import pprint

import numpy as np
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_reward_extra_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl_omni.trainer.diffusion.diffusion_trainer_utils import NoOpCheckpointManager
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    PolicyGradientRayTrainer,
    compute_advantage,
)
from verl_omni.workers.config.reward import reward_role_required


class NativeRayDiffusionTrainer(PolicyGradientRayTrainer):
    """Run native rollout, reward, advantage, and the model-owned policy update."""

    def init_workers(self):
        """Colocated actor workers + a standalone reward loop; no vLLM rollout / checkpoint sync."""
        actor_rollout_resource_pool = self._init_colocated_workers()
        from verl_omni.reward_loop import OmniRewardLoopManager

        reward_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if reward_role_required(self.config)
            else None
        )
        # PickScore etc. run as a colocated reward loop over the actor pool; no reward-model server.
        self.reward_loop_manager = OmniRewardLoopManager(
            config=self.config,
            rm_resource_pool=reward_pool,
            accelerator_resource_pool=actor_rollout_resource_pool,
        )
        # Native samples on the live FSDP module, so there is no vLLM server, no async rollout
        # manager, no streaming reward, and nothing to sync weights to.
        self.enable_agent_reward_loop = False
        self.llm_server_manager = None
        self.async_rollout_manager = None
        self.checkpoint_manager = NoOpCheckpointManager()

    def _validate(self):
        """Native rollout has no standard server-side validation endpoint."""
        return {"val/native/skipped": 1.0}

    def _generate_native(self, gen_batch_output: DataProto) -> DataProto:
        """Run the native worker ``generate`` and return a DataProto of images + rollout samples."""
        from verl.utils import tensordict_utils as tu

        # The gen batch's only field is a variable-length ``prompt_token_ids`` stored as non-tensor
        # data, so ``DataProto.batch`` is None and ``DataProto.to_tensordict()`` (which dereferences
        # ``self.batch.to_dict()``) would raise ``AttributeError: 'NoneType' object has no attribute
        # 'to_dict'``. Build the dispatch TensorDict directly from the tensor + non-tensor batch;
        # ``get_tensordict`` wraps the lists as NonTensorStacks and infers the batch size.
        td_dict: dict = {}
        if gen_batch_output.batch is not None:
            td_dict.update(gen_batch_output.batch.to_dict())
        for key, val in gen_batch_output.non_tensor_batch.items():
            td_dict[key] = list(val)
        gen_td = tu.get_tensordict(td_dict)
        gen_output = self.actor_rollout_wg.generate(gen_td)
        return DataProto.from_tensordict(gen_output)

    def fit(self):
        """Native training loop: worker generate -> reward -> flow_grpo advantage -> joint update."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        adv_estimator = self.config.algorithm.adv_estimator
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict = {}
                timing_raw: dict = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                # The gen batch carries only non-tensor prompt data (variable-length prompt_token_ids),
                # so DataProto.batch is None. Seed an empty tensor batch of the right size so union()
                # with the generated responses (and every downstream tensor op) has a batch to merge into.
                if gen_batch_output.batch is None:
                    from tensordict import TensorDict

                    gen_batch_output.batch = TensorDict({}, batch_size=[len(gen_batch_output)])

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Native rollout on the live FSDP actor module (no vLLM).
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_output = self._generate_native(gen_batch_output)
                    batch = gen_batch_output.union(gen_output)

                    # Reward is always computed here (native never streams it during rollout).
                    with marked_timer("reward", timing_raw, color="yellow"):
                        batch_reward = self._compute_reward_colocate(batch)
                        batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["sample_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        # One advantage per sample (num_timesteps == 1): the joint update reads a
                        # per-sample scalar, not a per-denoise-step vector like image-only flow_grpo.
                        sample_level_scores = batch.batch["sample_level_scores"]
                        batch.batch["sample_level_rewards"] = (
                            sample_level_scores if sample_level_scores.ndim > 1 else sample_level_scores.unsqueeze(-1)
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=adv_estimator,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            global_std=self.config.algorithm.global_std,
                            config=self.config.algorithm,
                        )

                    # Joint AR + image update on the FSDP module (record_old_logp already anchored
                    # old_logp inside generate). num_updates_per_batch == number of framework
                    # mini-batches (ppo_mini_batch_size vs the per-step sample count).
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
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    save_freq = self.config.trainer.get("rollout_data_save_freq", 1)
                    if rollout_data_dir and save_freq > 0 and self.global_steps % save_freq == 0:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics_diffusion(batch=batch))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                num_images = batch.batch["advantages"].shape[0]
                metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
                metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics.update(compute_reward_extra_metrics_diffusion(reward_extra_infos_dict))
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


__all__ = ["NativeRayDiffusionTrainer"]
