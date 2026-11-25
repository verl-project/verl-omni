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
"""Base class of the omni v1 trainers (``omni_sync``/``omni_separate_async``).

``PPOTrainer._setup`` exposes no rendezvous-port hook and ``_compute_advantage``
no post-scoring hook; both below are faithful copies of the pinned verl bodies
with their deltas DIFF-marked — re-diff them against the verl pin whenever the
pin moves.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from typing import Any

import numpy as np
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import CriticConfig, DistillationConfig, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_to_nested

from verl_omni.trainer.omni.advantage_gates import apply_advantage_gates
from verl_omni.utils.config import resolve_lora_config
from verl_omni.utils.net_utils import worker_group_port_ranges

__all__ = ["OmniPPOTrainer"]

logger = logging.getLogger(__name__)


def _reward_extra_step_metrics(reward_extra_infos: list[Mapping[str, Any] | None]) -> dict[str, float]:
    """Per-step aggregation of every numeric reward-extra column.

    Args:
        reward_extra_infos: One scorer-returned metrics mapping per episode
            (``extra_fields["reward_extra_info"]``), or ``None`` where a
            sample carries none.

    Returns:
        ``reward/<key>``: the column's mean over its finite values (columns
        are per-axis and sparse — a key exists only on the episodes that
        measured it — so missing and non-finite entries are skipped), and
        ``reward/<key>_never_share``: the non-finite share when any value is
        NaN or inf, so an ``inf`` latency (never spoke / never yielded) or
        an all-NaN column surfaces instead of silently averaging away.
    """
    columns: dict[str, list[float]] = defaultdict(list)
    for info in reward_extra_infos:
        if not isinstance(info, Mapping):
            continue
        for key, value in info.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                continue
            columns[str(key)].append(float(value))
    metrics: dict[str, float] = {}
    for key, values in sorted(columns.items()):
        finite = [value for value in values if math.isfinite(value)]
        non_finite = len(values) - len(finite)
        if finite:
            metrics[f"reward/{key}"] = sum(finite) / len(finite)
        if non_finite:
            metrics[f"reward/{key}_never_share"] = non_finite / len(values)
    return metrics


class OmniPPOTrainer(PPOTrainer):
    """``PPOTrainer`` whose worker groups get rendezvous port sub-ranges.

    Must be the first base so this ``_setup`` precedes ``PPOTrainer._setup`` in the MRO.
    """

    def _setup(self):
        # Faithful copy of verl's PPOTrainer._setup; deltas are DIFF-marked.
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        # DIFF vs upstream: resolve_lora_config also resolves pipeline-owned lora config shapes.
        self.ref_in_actor = resolve_lora_config(self.config.actor_rollout_ref.model).enabled

        # 1. define actor and rollout class
        if Role.Actor in self.role_worker_mapping:
            actor_role = Role.Actor
        elif Role.ActorRolloutRef in self.role_worker_mapping:
            actor_role = Role.ActorRolloutRef
        else:
            actor_role = Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        if actor_role == Role.Actor and self.use_reference_policy and not self.ref_in_actor:
            ref_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[ref_resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu

            # Wire the critic profiler config via the hydra path (real dataclass tool_config), so the
            # standalone critic TrainingWorker gets a working DistProfiler instead of a silent no-op.
            critic_omega_profiler_config = self.config.critic.get("profiler", {})
            critic_profiler_config = (
                omega_conf_to_dataclass(critic_omega_profiler_config) if critic_omega_profiler_config else None
            )

            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
                profiler_config=critic_profiler_config,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        # DIFF vs upstream: one disjoint sub-range of ray_master_port_range per group.
        pools = [(pool, class_dict) for pool, class_dict in self.resource_pool_to_cls.items() if class_dict]
        master_port_range = OmegaConf.select(self.config.trainer, "ray_master_port_range")
        port_ranges = worker_group_port_ranges(master_port_range, len(pools))
        for (resource_pool, class_dict), port_range in zip(pools, port_ranges, strict=True):
            if port_range is not None:
                wg_kwargs["master_port_range"] = port_range
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        if self.use_reference_policy and not self.ref_in_actor:
            if actor_role == Role.Actor:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                self.ref_policy_wg = all_wg[str(actor_role)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        # Trainers that opt out of colocated rollout replicas on the training
        # GPUs (v1 separate_async with actor_rollout_ref.hybrid_engine=False)
        # set ``self._enable_hybrid_replicas = False`` before super()._setup();
        # they get an empty manager here and serve rollout from their
        # standalone replicas only.
        if getattr(self, "_enable_hybrid_replicas", True):
            self.llm_server_manager: LLMServerManager = LLMServerManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
            )
        else:
            self.llm_server_manager = LLMServerManager.create_empty(config=self.config)

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_engine_config.backend = "naive"
        self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("all initialize finished, ready to fit")

    # ------------------------------ metrics ------------------------------

    def _compute_metrics(self, batch: KVBatchMeta, metrics: dict, timing_raw: dict, global_steps: int, epoch) -> None:
        """verl's per-step metrics plus the reward scorer's numeric extras."""
        super()._compute_metrics(batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
        metrics.update(self._reward_extra_metrics(batch))

    def _reward_extra_metrics(self, batch: KVBatchMeta) -> dict[str, float]:
        """Aggregate every numeric reward-extra column of this step's batch.

        verl's ``_compute_metrics`` reports only its fixed column set; the
        reward scorer's numeric extras (model-agnostic — any scorer's
        returned keys) ride each sample's ``extra_fields["reward_extra_info"]``
        in the TransferQueue store and are folded into the step metrics here
        (see :func:`_reward_extra_step_metrics` for the aggregation).
        """
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        extra_fields = data.pop("extra_fields").tolist()
        non_padding = [not tag.get("is_padding", False) for tag in batch.tags]
        reward_extra_infos = [
            extra_field.get("reward_extra_info") if isinstance(extra_field, Mapping) else None
            for extra_field, keep in zip(extra_fields, non_padding, strict=True)
            if keep
        ]
        return _reward_extra_step_metrics(reward_extra_infos)

    # ------------------------------ advantage ------------------------------

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """verl's advantage computation, then the agent loop's advantage gates.

        MiniCPM-o 4.5 under duplex GRPO is the only pipeline whose agent loop
        exports the ``advantage_mask`` / ``advantage_exclude`` columns; every
        other pipeline carries neither, gets verl's batch unchanged, and only
        gains the two zero-valued ``advantage/*`` step metrics.
        """
        # Faithful copy of verl's PPOTrainer._compute_advantage; the delta is DIFF-marked.
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob"]
        if self.use_critic:
            fields.append("values")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # DIFF vs upstream: the agent loop's advantage gates (verl has no equivalent).
        metrics.update(self._apply_advantage_gates(batch, data))

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _apply_advantage_gates(self, batch: KVBatchMeta, data: DataProto) -> dict[str, float]:
        """Apply the batch's ``extra_fields`` advantage-gate columns to the scored batch.

        A padded row carries a deep copy of a real sample's ``extra_fields``;
        it is nulled here so the sequence stays aligned with ``data``'s rows
        while only real samples gate.
        """
        extras = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        extra_fields = extras.pop("extra_fields").tolist()
        non_padding = [not tag.get("is_padding", False) for tag in batch.tags]
        aligned = [
            extra_field if keep and isinstance(extra_field, Mapping) else None
            for extra_field, keep in zip(extra_fields, non_padding, strict=True)
        ]
        return apply_advantage_gates(
            data,
            aligned,
            adv_estimator=self.config.algorithm.adv_estimator,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
        )
