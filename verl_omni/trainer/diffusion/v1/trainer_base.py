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
"""V1 policy-gradient diffusion trainer base.

This is the verl-omni counterpart of upstream ``verl.trainer.ppo.v1.trainer_base``.
It reuses upstream verl v1 infrastructure (ReplayBuffer, LLMServerManager,
CheckpointEngineManager, RewardLoopManager, hook lifecycle) but keeps the
diffusion ``DataProto`` compute contract (image/video responses, dense SDE-step
log-probs, image-level rewards, Flow-GRPO advantages) instead of the token PPO
compute path. It must NOT be subclassed from upstream ``PPOTrainer`` because the
token-level assumptions (response_mask, token rewards, critic values) do not hold
for diffusion.
"""

import json
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.ppo.metric_utils import compute_variance_proxy_metrics, process_validation_metrics
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.skip import SkipManager
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager

from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_reward_extra_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl_omni.trainer.diffusion.ray_diffusion_trainer import compute_advantage
from verl_omni.trainer.diffusion.rollout_correction import (
    apply_bypass_mode_to_diffusion_batch,
    apply_rollout_correction_to_diffusion_batch,
    rollout_correction_enabled,
)
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl_omni.trainer.diffusion.v1.tq_utils import (
    diffusion_tq_batch_to_dataproto,
    sort_diffusion_tq_keys,
)
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


DIFFUSION_TRAINER_REGISTRY: dict[str, type] = {}


def register_diffusion_trainer(name: str):
    """Class decorator that registers a PolicyGradientDiffusionTrainerV1 subclass."""

    def decorator(cls):
        DIFFUSION_TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_diffusion_trainer_cls(name: str):
    """Return the diffusion v1 trainer class registered under ``name``."""
    try:
        return DIFFUSION_TRAINER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(DIFFUSION_TRAINER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown diffusion trainer '{name}'. Available: {available}.") from None


class PolicyGradientDiffusionTrainerV1(ABC):
    """Base class for v1 policy-gradient diffusion trainers.

    Mirrors the upstream ``PPOTrainer`` v1 lifecycle (replay buffer, TransferQueue
    rollout handoff, mode hooks, ``fit``/``step``) while keeping the diffusion
    ``DataProto`` compute contract. Subclasses only define mode-specific hooks
    (when rollout replicas sleep, abort, resume, or receive updated weights).
    """

    def __init__(self, config):
        self.config = config
        self.trainer_mode = config.trainer.v1.trainer_mode
        self.parameter_sync_step = config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        self.use_reference_policy = need_reference_policy(config)
        self.use_rm = need_reward_model(config)
        self.replay_buffer = self._build_replay_buffer()

        # ref_in_actor: reference policy is the actor without lora applied.
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        self.checkpoint_manager = None
        self.global_steps = 0

    # ------------------------------ replay buffer ------------------------------

    def _build_replay_buffer(self) -> ReplayBuffer:
        sampler_config = self.config.trainer.v1.sampler
        return ReplayBuffer(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
        )

    # ------------------------------ lifecycle ------------------------------

    def init(self):
        """Initialize workers, rollout server, reward loop, checkpoint engine."""
        self._setup()
        self.on_init_end()

    def fit(self, agent_loop_manager: AgentLoopManager):
        """Run the v1 training loop, mirroring upstream ``PPOTrainer.fit``."""
        self.agent_loop_manager = agent_loop_manager

        # initialize SkipManager for V1 rollout skip support (no-op until configured).
        SkipManager.init(self.config)

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            val_metrics = self._validate()
            self.on_validate_end()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // self.steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Diffusion Training")

        self.global_steps += 1
        SkipManager.set_step(self.global_steps)
        self.on_train_begin()
        last_val_metrics = None
        while current_epoch < self.config.trainer.total_epochs and self.global_steps <= self.total_training_steps:
            is_last_step = self.global_steps >= self.total_training_steps
            metrics: dict = {}
            self.timing_raw: dict = {}
            with marked_timer("step", self.timing_raw):
                self.on_step_begin()
                batch = self.step(metrics, self.timing_raw)
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                        self._save_checkpoint()
                self.on_step_end()

            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    val_metrics = self._validate()
                    self.on_validate_end()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            self._compute_metrics(batch, metrics, self.timing_raw, self.global_steps, current_epoch)

            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, self.timing_raw, rollout_data_dir)

            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

            self.logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

        self.on_train_end()
        self._shutdown_dump_executor()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """Feed one train batch and run ``parameter_sync_step`` local updates."""
        train_batch_size = self.config.data.train_batch_size
        assert train_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({train_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        sample_batch_size = train_batch_size // self.parameter_sync_step

        with marked_timer("feed", timing_raw):
            self._add_batch_to_generate()

        combined_keys: list = []
        combined_tags: list = []
        combined_partition_id = "train"
        for _ in range(self.parameter_sync_step):
            iter_metrics: dict = {}
            batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
            metrics.update(iter_metrics)
            combined_keys.extend(batch.keys)
            combined_tags.extend(batch.tags)
            combined_partition_id = batch.partition_id

        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """Sample one mini-batch from the replay buffer and run the diffusion PG pipeline."""
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch_meta, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            self.on_sample_end()

        # Convert TQ rows to diffusion DataProto; from here on the driver owns the
        # DataProto compute contract (no KVBatchMeta passed to diffusion workers).
        data = diffusion_tq_batch_to_dataproto(batch_meta, pad_token_id=self.tokenizer.pad_token_id or 0)

        # [OPTIONAL] colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None and self.use_rm:
            with marked_timer("reward", timing_raw, color="yellow"):
                self.checkpoint_manager.sleep_replicas()
                data = self._compute_reward_colocate(data)
                self.checkpoint_manager.update_weights(self.global_steps)

        data = self._balance_batch(data, metrics=metrics)

        # Bypass mode: skip old_log_prob recompute (2 policies).
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = bool(rollout_corr_config and rollout_corr_config.get("bypass_mode", False))
        if bypass_recomputing_logprobs:
            apply_bypass_mode_to_diffusion_batch(data)
        else:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self._compute_old_log_prob(data)
                data = data.union(old_log_prob)

        assert "old_log_probs" in data.batch, f'"old_log_probs" not in {data.batch.keys()}'

        if not bypass_recomputing_logprobs and rollout_correction_enabled(rollout_corr_config):
            with marked_timer("rollout_corr", timing_raw, color="cyan"):
                data, rollout_corr_metrics = apply_rollout_correction_to_diffusion_batch(data, rollout_corr_config)
                metrics.update(rollout_corr_metrics)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(data)
                data = data.union(ref_log_prob)

        with marked_timer("adv", timing_raw, color="brown"):
            data = self._compute_advantage(data)

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(data)
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_metrics)

        # Persist computed fields back to TransferQueue so the sampled keys carry
        # the full trajectory for metrics/dumping (keys are cleared after step).
        # Slice to the original key count in case ``_balance_batch`` appended pad rows.
        from verl_omni.trainer.diffusion.v1.tq_utils import put_dataproto_fields_to_tq

        n_keys = len(batch_meta.keys)
        if len(data) > n_keys:
            data_for_tq = data.select_idxs(list(range(n_keys)))
        else:
            data_for_tq = data
        put_dataproto_fields_to_tq(
            batch_meta,
            data_for_tq,
            fields=["old_log_probs", "advantages", "returns", "sample_level_scores", "sample_level_rewards"],
        )
        return batch_meta

    # ------------------------------ abstract hooks ------------------------------

    def on_init_end(self):
        """Called after initialization ends."""
        return

    def on_train_begin(self):
        """Called before the training loop starts."""
        return

    def on_train_end(self):
        """Called after the training loop ends."""
        return

    def on_validate_begin(self):
        """Called before validation."""
        return

    def on_validate_end(self):
        """Called after validation."""
        return

    def on_step_begin(self):
        """Called at the beginning of each training step."""
        return

    def on_sample_begin(self):
        """Called at the beginning of sampling from the replay buffer."""
        return

    @abstractmethod
    def on_step_end(self):
        """Called at the end of each training step."""
        return

    @abstractmethod
    def on_sample_end(self):
        """Called after sampling a batch from the replay buffer."""
        return

    # ------------------------------ diffusion cache hooks (no-op) ------------------------------

    def release_rollout_cache_for_weight_sync(self) -> None:
        """No-op for pure diffusion models (no KV cache)."""
        return

    def resume_rollout_cache_after_weight_sync(self) -> None:
        """No-op for pure diffusion models (no KV cache)."""
        return

    # ------------------------------ setup ------------------------------

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_rollout_resource_pool = self._init_colocated_workers()
        self._init_online_rollout_stack(actor_rollout_resource_pool)
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("diffusion v1 trainer initialized, ready to fit")

    def _init_tokenizer(self):
        import json

        from verl.utils import hf_processor, hf_tokenizer

        from verl_omni.utils.fs import resolve_model_local_dir

        local_path = resolve_model_local_dir(
            self.config.actor_rollout_ref.model.path,
            use_shm=self.config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)

        if self.config.actor_rollout_ref.model.tokenizer_path is None:
            tokenizer_path = os.path.join(local_path, "tokenizer")
            self.config.actor_rollout_ref.model.tokenizer_path = (
                tokenizer_path if os.path.exists(tokenizer_path) else local_path
            )
        self.tokenizer = hf_tokenizer(
            self.config.actor_rollout_ref.model.tokenizer_path, trust_remote_code=trust_remote_code
        )

        # Resolve the diffusion architecture and let the pipeline adapter prepare
        # processor files before loading the processor (mirrors main_diffusion.py).
        model_config = self.config.actor_rollout_ref.model
        architecture = model_config.get("architecture")
        if architecture is None:
            model_index_path = os.path.join(local_path, "model_index.json")
            try:
                with open(model_index_path) as model_index_file:
                    architecture = json.load(model_index_file)["_class_name"]
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Unable to infer the diffusion architecture from {model_index_path}. "
                    "Set actor_rollout_ref.model.architecture explicitly."
                ) from exc

        from verl_omni.pipelines.model_base import DiffusionModelBase

        prepared_processor_path = DiffusionModelBase.get_class_by_name(
            architecture,
            model_config.algorithm,
            model_config.get("external_lib"),
        ).prepare_processor_files(local_path)
        processor_path = os.path.join(local_path, "processor")
        if prepared_processor_path is not None:
            processor_path = prepared_processor_path
        if not os.path.exists(processor_path):
            processor_path = local_path
        self.processor = hf_processor(processor_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        from verl_omni.utils.dataset.rl_dataset import (
            create_rl_dataset,
            create_rl_sampler,
            get_collate_fn,
        )

        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=get_collate_fn(self.config.data),
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=get_collate_fn(self.config.data),
        )

        self.steps_per_epoch = len(self.train_dataset) // self.config.data.train_batch_size
        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total diffusion training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        total_training_steps * self.parameter_sync_step
                    )
        except Exception as e:
            logger.warning(f"Could not set total_training_steps in config: {e}")

    def _init_dump_executor(self):
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _init_resource_pool_mgr(self):
        self.role_worker_mapping = {}
        self.mapping = {}
        role = Role.ActorRolloutRef if self.use_reference_policy and not self.ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes}

        if self.use_rm and self.config.reward.reward_model.enable_resource_pool:
            reward_pool = [self.config.reward.reward_model.n_gpus_per_node] * self.config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            if self.use_rm:
                self.config.reward.reward_model.nnodes = self.config.trainer.nnodes
                self.config.reward.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _init_colocated_workers(self):
        """Create the colocated actor/ref worker group (diffusion has no critic)."""
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.config.trainer.device}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg
        return actor_rollout_resource_pool

    def _init_online_rollout_stack(self, actor_rollout_resource_pool):
        """Initialize reward loop, LLM server, and checkpoint engine managers."""
        from verl_omni.reward_loop import OmniRewardLoopManager

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = OmniRewardLoopManager(config=self.config, rm_resource_pool=resource_pool)

        # Streaming agent reward loop when there is no rm, or the rm has a separate pool.
        self.enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

    # ------------------------------ client handles ------------------------------

    def get_llm_client(self):
        return self.llm_server_manager.get_client()

    def get_reward_handles(self):
        if self.enable_agent_reward_loop:
            return self.reward_loop_manager.reward_loop_workers
        return None

    # ------------------------------ prompt submission ------------------------------

    def _fetch_one_gen_batch(self):
        if self.train_dataloader_it is None:
            self.train_dataloader_it = iter(self.train_dataloader)
        try:
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        batch_dict["index"] = np.arange(len(batch_dict["raw_prompt"]))
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None) -> tu.TensorDict:
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or train_batch_size
        if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size ({gen_batch_size})"
            )
        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // gen_batch_size)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        rollout_seed_cfg = self.config.actor_rollout_ref.rollout.get("seed")
        if rollout_seed_cfg is not None:
            tu.assign_non_tensor_data(batch, "rollout_seed", int(rollout_seed_cfg) + self.global_steps - 1)
        return batch

    def _submit_batch_to_rollout(self, batch) -> int:
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        if self.trainer_mode != "sync":
            from tensordict.tensorclass import NonTensorData

            tq.kv_batch_put(
                keys=list(batch["uid"]),
                partition_id="train",
                tags=tags,
                fields=batch.select(*[k for k in batch.keys() if not isinstance(batch.get(k), NonTensorData)]),
            )
        else:
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        batch = self._next_train_batch(num_prompts)
        return self._submit_batch_to_rollout(batch)

    def _add_batch_to_generate(self):
        batch = self._next_train_batch()
        self._submit_batch_to_rollout(batch)

    # ------------------------------ diffusion compute ------------------------------

    def _compute_reward_colocate(self, data: DataProto) -> DataProto:
        """Compute reward score with a colocated reward model on a DataProto batch."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        return self.reward_loop_manager.compute_rm_score(data)

    def _balance_batch(self, data: DataProto, metrics: dict) -> DataProto:
        """Ensure the DataProto is divisible by the actor dp group size.

        Diffusion has no critic, so the only divisibility constraint comes from
        the actor update. The sampled trajectory count is
        ``sample_batch_size * rollout.n``; users must configure
        ``train_batch_size`` so that this is divisible by the actor dp size
        (same assumption as the legacy diffusion trainer). When padding is
        unavoidable, pad rows are appended at the end so the original keys stay
        aligned with the first ``len(batch_meta)`` rows for TQ write-back.
        """
        dp_size = 1
        if hasattr(self.actor_rollout_wg, "_query_dispatch_info"):
            info = self.actor_rollout_wg._query_dispatch_info("actor")
            dp_size = max(info.values()) + 1 if info else 1
        actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
        batch_multiple = math.lcm(dp_size, actor_global_mini_batch_size)
        if len(data) % batch_multiple != 0:
            data, _ = pad_dataproto_to_divisor(data, size_divisor=batch_multiple)
        return data

    def _compute_old_log_prob(self, data: DataProto) -> DataProto:
        """Recompute old log-probs over diffusion latents with the actor engine."""
        batch_td = data.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        tu.assign_non_tensor(
            batch_td,
            compute_loss=False,
            height=self.config.actor_rollout_ref.model.pipeline.height,
            width=self.config.actor_rollout_ref.model.pipeline.width,
            vae_scale_factor=self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        )
        output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        log_probs = tu.get(output, "log_probs")
        old_log_prob_dict = {"old_log_probs": log_probs.float()}
        prev_sample_mean = tu.get(output, "prev_sample_mean")
        if prev_sample_mean is not None:
            old_log_prob_dict["old_prev_sample_mean"] = prev_sample_mean.float()
        return DataProto.from_tensordict(tu.get_tensordict(old_log_prob_dict))

    def _compute_ref_log_prob(self, data: DataProto) -> DataProto:
        """Compute reference log-probs over diffusion latents."""
        batch_td = data.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        metadata = {
            "compute_loss": False,
            "height": self.config.actor_rollout_ref.model.pipeline.height,
            "width": self.config.actor_rollout_ref.model.pipeline.width,
            "vae_scale_factor": self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        else:
            output = self.ref_policy_wg.infer_ref_batch(batch_td)
        log_probs = tu.get(output, "log_probs")
        prev_sample_mean = tu.get(output, "prev_sample_mean")
        ref_dict = {"ref_log_prob": log_probs.float()}
        if prev_sample_mean is not None:
            ref_dict["ref_prev_sample_mean"] = prev_sample_mean.float()
        return DataProto.from_tensordict(tu.get_tensordict(ref_dict))

    def _compute_advantage(self, data: DataProto) -> DataProto:
        """Compute diffusion (Flow-GRPO) advantages on the driver."""
        reward_tensor, reward_extra_infos_dict = extract_reward(data)
        data.batch["sample_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        num_timesteps = data.batch["old_log_probs"].shape[1]
        data.batch["sample_level_rewards"] = data.batch["sample_level_scores"].expand(-1, num_timesteps)

        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        data = compute_advantage(
            data,
            adv_estimator=self.config.algorithm.adv_estimator,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            global_std=self.config.algorithm.global_std,
            config=self.config.algorithm,
        )
        return data

    def _update_actor(self, data: DataProto) -> DataProto:
        """Update the diffusion actor network."""
        rollout_config = self.config.actor_rollout_ref.rollout
        data.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch_td = data.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=self.config.actor_rollout_ref.actor.ppo_epochs,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            height=self.config.actor_rollout_ref.model.pipeline.height,
            width=self.config.actor_rollout_ref.model.pipeline.width,
            vae_scale_factor=self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if (actor_mfu := actor_output.pop("actor/mfu", None)) is not None:
            actor_output["perf/mfu/actor"] = actor_mfu
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    # ------------------------------ validation ------------------------------

    def _validate(self) -> dict:
        """Validation via TransferQueue: dispatch, sample, convert, reward, dump images."""
        sample_inputs: list[str] = []
        sample_outputs: list[torch.Tensor] = []
        sample_gts: list = []
        sample_scores: list[float] = []
        sample_turns: list = []
        sample_uids: list = []
        data_sources: list = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        for batch_dict in self.val_dataloader:
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch_dict["index"] = np.arange(len(batch_dict["raw_prompt"]))
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": self.global_steps}
                for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)

            batch_meta, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=len(batch)
            )
            data = diffusion_tq_batch_to_dataproto(batch_meta, pad_token_id=self.tokenizer.pad_token_id or 0)

            if self.use_rm and self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                data = self._compute_reward_colocate(data)
                self.checkpoint_manager.update_weights(self.global_steps)

            input_ids = data.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_images = data.batch["responses"]
            sample_inputs.extend(input_texts)
            sample_outputs.append(output_images)
            uids = data.non_tensor_batch.get("uid")
            sample_uids.extend(list(uids) if uids is not None else [None] * len(data))

            reward_tensor, reward_extra_info = extract_reward(data)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            if "__num_turns__" in data.non_tensor_batch:
                sample_turns.append(data.non_tensor_batch["__num_turns__"])
            data_sources.append(data.non_tensor_batch.get("data_source", ["unknown"] * len(data)))

            rm_meta = data.non_tensor_batch.get("reward_model")
            if rm_meta is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in rm_meta.tolist()])
            else:
                sample_gts.extend([None] * len(data))

            tq.kv_clear(keys=batch_meta.keys, partition_id=batch_meta.partition_id)

        sample_outputs = torch.cat(sample_outputs, dim=0) if sample_outputs else torch.empty(0)
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir and len(sample_outputs) > 0:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        data_sources_arr = np.concatenate(data_sources, axis=0) if data_sources else np.array([])
        return self._val_metrics_update(data_sources_arr, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return
        if "wandb" in self.config.trainer.logger:
            import wandb

            outputs = [wandb.Image(image.float(), file_type="jpg") for image in outputs]
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump validation/rollout samples as images + JSONL (runs in background)."""
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            self.global_steps,
        )
        self._dump_futures.append(future)
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        os.makedirs(dump_path, exist_ok=True)
        visual_folder = os.path.join(dump_path, f"{global_steps}")
        os.makedirs(visual_folder, exist_ok=True)

        output_paths = []
        images_pil = outputs.cpu().float()
        # images: [N, C, H, W] -> [N, H, W, C]
        if images_pil.dim() == 4:
            images_pil = images_pil.permute(0, 2, 3, 1).numpy()
        else:
            images_pil = images_pil.numpy()
        images_pil = (images_pil * 255).round().clip(0, 255).astype("uint8")
        for i, image in enumerate(images_pil):
            image_path = os.path.join(visual_folder, f"{i}.jpg")
            Image.fromarray(image).save(image_path)
            output_paths.append(image_path)

        filename = os.path.join(dump_path, f"{global_steps}.jsonl")
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": output_paths,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")
        print(f"Dumped diffusion generations to {filename}")

    def _log_rollout_data(self, batch_meta: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout rows from TQ and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            data = diffusion_tq_batch_to_dataproto(batch_meta, pad_token_id=self.tokenizer.pad_token_id or 0)
            inputs = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
            outputs = data.batch["responses"]
            scores = data.batch["sample_level_scores"].sum(-1).cpu().tolist() if "sample_level_scores" in data.batch else (
                data.batch["rm_scores"].sum(-1).cpu().tolist() if "rm_scores" in data.batch else [0.0] * len(data)
            )
            rm_meta = data.non_tensor_batch.get("reward_model")
            gts = [item.get("ground_truth", None) for item in rm_meta.tolist()] if rm_meta is not None else [None] * len(data)

            sort_idx = sort_diffusion_tq_keys(list(batch_meta.keys))
            inputs = [inputs[i] for i in sort_idx]
            outputs = outputs[torch.tensor(sort_idx)]
            gts = [gts[i] for i in sort_idx]
            scores = [scores[i] for i in sort_idx]
            reward_extra_infos_dict = {"uid": [batch_meta.keys[i] for i in sort_idx]}
            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val
        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()
        return metric_dict

    # ------------------------------ metrics ------------------------------

    def _compute_metrics(self, batch_meta: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        data = diffusion_tq_batch_to_dataproto(batch_meta, pad_token_id=self.tokenizer.pad_token_id or 0)
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(compute_data_metrics_diffusion(batch=data))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        num_images = (
            data.batch["advantages"].shape[0]
            if "advantages" in data.batch
            else data.batch["sample_level_scores"].shape[0]
        )
        metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
        metrics.update(compute_throughput_metrics_diffusion(batch=data, timing_raw=timing_raw, n_gpus=n_gpus))
        reward_extra_infos_dict = {}
        if "reward_extra_info" in data.non_tensor_batch:
            infos = data.non_tensor_batch["reward_extra_info"].tolist()
            keys = infos[0].keys() if infos and isinstance(infos[0], dict) else []
            for key in keys:
                reward_extra_infos_dict[key] = [info.get(key) for info in infos]
        metrics.update(compute_reward_extra_metrics_diffusion(reward_extra_infos_dict))
        if "advantages" in data.batch:
            gradient_norm = metrics.get("actor/grad_norm", None)
            metrics.update(compute_variance_proxy_metrics(batch=data, gradient_norm=gradient_norm))

        # off-policy staleness metrics (model-version units)
        non_padding = np.array([not tag.get("is_padding", False) for tag in batch_meta.tags], dtype=bool)
        if non_padding.any():
            min_gs = np.array([tag["min_global_steps"] for tag in batch_meta.tags], dtype=int)[non_padding]
            max_gs = np.array([tag["max_global_steps"] for tag in batch_meta.tags], dtype=int)[non_padding]
            spans = max_gs - min_gs + 1
            staleness = (global_steps - 1) - max_gs
            staleness_worst = (global_steps - 1) - min_gs
            metrics.update(
                {
                    "training/off_policy/trajectory_spans/mean": spans.mean(),
                    "training/off_policy/trajectory_spans/max": spans.max(),
                    "training/off_policy/trajectory_spans/min": spans.min(),
                    "training/off_policy/trajectory_staleness/mean": staleness.mean(),
                    "training/off_policy/trajectory_staleness/max": staleness.max(),
                    "training/off_policy/trajectory_staleness/min": staleness.min(),
                    "training/off_policy/trajectory_staleness_worst/mean": staleness_worst.mean(),
                    "training/off_policy/trajectory_staleness_worst/max": staleness_worst.max(),
                    "training/off_policy/trajectory_staleness_worst/min": staleness_worst.min(),
                }
            )

    # ------------------------------ checkpoint ------------------------------

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving diffusion checkpoint to {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        local_mkdir_safe(local_global_step_folder)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))

        latest = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(latest, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        self.global_steps = 0
        if self.config.trainer.resume_mode == "disable":
            return
        if self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")
            return

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming diffusion from {global_step_folder}, global_steps={self.global_steps}")
        self.actor_rollout_wg.load_checkpoint(
            os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_path):
            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
        else:
            logger.warning(f"No dataloader state at {dataloader_path}, starting from scratch")








