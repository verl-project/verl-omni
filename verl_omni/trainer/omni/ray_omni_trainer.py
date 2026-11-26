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

import asyncio
import concurrent.futures
import logging
import os
import uuid
import warnings
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from tensordict import NonTensorData, NonTensorStack, TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.distillation import is_distillation_enabled
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
from verl_omni.trainer.distillation.self_distillation import (
    align_teacher_sequence_outputs,
    build_multimodal_teacher_prompt,
    build_neutral_teacher_outputs,
    extract_sequence_rewards,
    select_teacher_contexts,
    summarize_self_distillation_metrics,
)
from verl_omni.trainer.omni.omni_algos import (
    get_omni_loss_fn,
)
from verl_omni.trainer.omni.trainer_base import OmniPPOTrainer
from verl_omni.utils.config import resolve_lora_config
from verl_omni.utils.dataset.offline_mllm_dpo_dataset import get_batch_modality
from verl_omni.utils.metrics_utils import GroupedMetricMean
from verl_omni.utils.net_utils import worker_group_port_ranges
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.config.omni.distillation import is_self_distillation_enabled

sys_logger = logging.getLogger(__name__)

__all__ = ["OmniPPOTrainerSync", "OmniDirectPreferenceRayTrainer"]


def _run_coro_sync(coro):
    """Run an async coroutine from synchronous trainer code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _extract_non_tensor_list(data: Any, field: str, batch_size: int) -> list[Any] | None:
    """Return a plain Python list of length ``batch_size`` for a non-tensor TransferQueue field."""
    if field not in data.keys():
        return None
    raw = data[field]
    if isinstance(raw, NonTensorStack):
        items = [item.data if isinstance(item, NonTensorData) else item for item in raw]
    elif isinstance(raw, NonTensorData):
        items = [raw.data] * batch_size
    elif isinstance(raw, np.ndarray):
        items = raw.tolist()
    else:
        items = [item.data if isinstance(item, NonTensorData) else item for item in list(raw)]
    return [item.item() if isinstance(item, np.generic) else item for item in items]


def _extract_sequence_rows(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Unbind a jagged nested tensor or 2D tensor into per-sample 1D tensors."""
    if tensor.is_nested:
        return list(tensor.unbind())
    return [tensor[i] for i in range(tensor.size(0))]


def _extract_inline_multimodal_data(raw_prompt: Any) -> dict[str, list[Any]]:
    """Collect already-decoded media payloads directly from structured chat messages."""
    if not isinstance(raw_prompt, list):
        return {}
    images: list[Any] = []
    videos: list[Any] = []
    audios: list[Any] = []
    for msg in raw_prompt:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image" and item.get("image") is not None:
                images.append(item["image"])
            elif item_type == "video" and item.get("video") is not None:
                videos.append(item["video"])
            elif item_type == "audio" and item.get("audio") is not None:
                audios.append(item["audio"])
    result: dict[str, list[Any]] = {}
    if images:
        result["images"] = images
    if videos:
        result["videos"] = videos
    if audios:
        result["audios"] = audios
    return result


@register_trainer("omni_sync")
class OmniPPOTrainerSync(OmniPPOTrainer, PPOTrainerSync):
    """``PPOTrainerSync`` subclass that wires tokenizer/processor from ``OmniModelConfig``."""

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        for role in (Role.ActorRolloutRef, Role.ActorRollout):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)

    def _ensure_teacher_unembeds(self):
        if getattr(self, "_teacher_unembeds", None) is not None:
            return

        from verl_omni.utils.lm_head import load_lm_head_weight

        teacher_models = self.distillation_config.teacher_models
        unembeds = {}
        for key, teacher_config in teacher_models.items():
            w = load_lm_head_weight(teacher_config.model_path)
            unembeds[key] = w.to(dtype=torch.bfloat16).contiguous()

        self._teacher_unembeds = unembeds
        self.actor_rollout_wg.set_teacher_unembeds(unembeds)

    async def _resolve_self_teacher_multimodal_data(
        self,
        raw_prompt: Any,
        existing_mm_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Resolve raw images/videos/audios for reprompting when TransferQueue omits ``multi_modal_data``."""
        if isinstance(existing_mm_data, dict) and any(existing_mm_data.get(k) for k in ("images", "videos", "audios")):
            return existing_mm_data
        if not isinstance(raw_prompt, list):
            return {}
        inline_mm = _extract_inline_multimodal_data(raw_prompt)
        # If messages hold raw path/URL strings and a processor is present, use dataset preprocessing.
        needs_loading = any(isinstance(val, str) for bucket in inline_mm.values() for val in bucket)
        if needs_loading and getattr(self, "processor", None) is not None:
            try:
                from verl.utils.dataset.rl_dataset import get_dataset_class

                dataset_cls = get_dataset_class(self.config.data)
                image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
                if hasattr(dataset_cls, "process_multi_modal_info"):
                    images, videos, audios = await dataset_cls.process_multi_modal_info(
                        raw_prompt, image_patch_size=image_patch_size, config=self.config.data
                    )
                else:
                    images, videos = await dataset_cls.process_vision_info(
                        raw_prompt, image_patch_size=image_patch_size, config=self.config.data
                    )
                    audios = None
                loaded: dict[str, Any] = {}
                if images is not None:
                    loaded["images"] = images
                if videos is not None:
                    loaded["videos"] = videos
                if audios is not None:
                    loaded["audios"] = audios
                if loaded:
                    return loaded
            except Exception as exc:
                sys_logger.warning(
                    "Falling back to inline multimodal payloads during self-distillation reprompting: %s",
                    exc,
                )
        return inline_mm

    def _compute_self_teacher_log_probs(self, batch, metrics: dict):
        """Build reprompted multimodal teacher sequences and score them on the student's rollout server."""
        self_cfg = self.distillation_config.self_teacher
        if self_cfg.teacher_source == "ema":
            raise NotImplementedError(
                "distillation.self_teacher.teacher_source='ema' requires syncing EMA weights to "
                "rollout servers, which is not yet wired in OmniPPOTrainerSync. Use "
                "teacher_source='policy' for on-policy rollout-server self-distillation."
            )

        desired_fields = [
            "prompts",
            "responses",
            "response_mask",
            "uid",
            "rm_scores",
            "raw_prompt",
            "extra_info",
            "reward_model",
            "reward_extra_info",
            "multi_modal_data",
            "mm_processor_kwargs",
            "multi_modal_inputs",
        ]
        if batch.fields is not None:
            available = set(batch.fields)
            select_fields = [f for f in desired_fields if f in available]
        else:
            select_fields = None

        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=select_fields)
        batch_size = len(batch)

        prompt_rows = _extract_sequence_rows(data["prompts"])
        response_rows = _extract_sequence_rows(data["responses"])
        response_mask_rows = _extract_sequence_rows(data["response_mask"])
        student_prompt_lens = [int(p.numel()) for p in prompt_rows]

        uids = _extract_non_tensor_list(data, "uid", batch_size)
        if uids is None:
            uids = [str(i) for i in range(batch_size)]

        if "rm_scores" in data.keys():
            rewards = extract_sequence_rewards(data["rm_scores"])
        else:
            rewards = [0.0] * batch_size

        responses_text = [
            self.tokenizer.decode(
                resp[mask.bool()].tolist() if mask.numel() == resp.numel() else resp.tolist(),
                skip_special_tokens=True,
            )
            for resp, mask in zip(response_rows, response_mask_rows, strict=True)
        ]

        raw_prompts = _extract_non_tensor_list(data, "raw_prompt", batch_size)
        extra_infos = _extract_non_tensor_list(data, "extra_info", batch_size)
        reward_extra_infos = _extract_non_tensor_list(data, "reward_extra_info", batch_size)
        reward_models = _extract_non_tensor_list(data, "reward_model", batch_size)
        multi_modal_datas = _extract_non_tensor_list(data, "multi_modal_data", batch_size)
        mm_processor_kwargs_list = _extract_non_tensor_list(data, "mm_processor_kwargs", batch_size)
        multi_modal_inputs_list = _extract_non_tensor_list(data, "multi_modal_inputs", batch_size)

        decisions = select_teacher_contexts(
            uids=uids,
            response_texts=responses_text,
            rewards=rewards,
            config=self_cfg,
            extra_infos=extra_infos,
            reward_extra_infos=reward_extra_infos,
            reward_models=reward_models,
        )

        loss_cfg = self.distillation_config.distillation_loss
        use_topk = bool(loss_cfg.loss_settings.use_topk)
        width = int(loss_cfg.topk) if use_topk else 1
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        sampling_params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": width if use_topk else 0,
            "detokenize": False,
        }
        default_mm_kwargs = dict(self.config.data.get("mm_processor_kwargs", {}) or {})
        llm_client = self.get_llm_client()

        async def _score_all():
            async def _score_one(i: int):
                student_prompt_ids = prompt_rows[i].tolist()
                student_prompt_len = student_prompt_lens[i]
                response_ids = response_rows[i].tolist()
                response_len = len(response_ids)
                decision = decisions[i]
                if not decision.has_context:
                    neutral_ids, neutral_lps = build_neutral_teacher_outputs(
                        student_prompt_len=student_prompt_len,
                        response_len=response_len,
                        topk=width,
                        pad_token_id=pad_token_id,
                    )
                    return neutral_ids, neutral_lps, None

                sample_raw_prompt = (
                    raw_prompts[i]
                    if raw_prompts is not None
                    else self.tokenizer.decode(student_prompt_ids, skip_special_tokens=False)
                )
                sample_mm_data = await self._resolve_self_teacher_multimodal_data(
                    sample_raw_prompt,
                    multi_modal_datas[i] if multi_modal_datas is not None else None,
                )
                sample_mm_kwargs = (
                    dict(mm_processor_kwargs_list[i])
                    if mm_processor_kwargs_list is not None and isinstance(mm_processor_kwargs_list[i], dict)
                    else dict(default_mm_kwargs)
                )
                sample_mm_inputs = multi_modal_inputs_list[i] if multi_modal_inputs_list is not None else None

                reprompted = build_multimodal_teacher_prompt(
                    raw_prompt=sample_raw_prompt,
                    student_prompt_ids=student_prompt_ids,
                    privileged_text=decision.privileged_text,
                    tokenizer=self.tokenizer,
                    processor=getattr(self, "processor", None),
                    multi_modal_data=sample_mm_data,
                    mm_processor_kwargs=sample_mm_kwargs,
                    existing_multi_modal_inputs=sample_mm_inputs,
                    config=self_cfg,
                )
                teacher_seq_ids = reprompted.teacher_prompt_ids + response_ids
                teacher_output = await llm_client.generate(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=teacher_seq_ids,
                    sampling_params=sampling_params,
                    image_data=reprompted.multi_modal_data.get("images"),
                    video_data=reprompted.multi_modal_data.get("videos"),
                    audio_data=reprompted.multi_modal_data.get("audios"),
                    mm_processor_kwargs=reprompted.mm_processor_kwargs or None,
                )
                raw_ids = torch.as_tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
                raw_lps = torch.as_tensor(teacher_output.extra_fields["prompt_logprobs"], dtype=torch.float32)
                aligned_ids, aligned_lps = align_teacher_sequence_outputs(
                    teacher_ids=raw_ids,
                    teacher_logprobs=raw_lps,
                    teacher_prompt_len=len(reprompted.teacher_prompt_ids),
                    student_prompt_len=student_prompt_len,
                    response_len=response_len,
                    pad_token_id=pad_token_id,
                )
                return aligned_ids, aligned_lps, reprompted.added_prompt_tokens

            return await asyncio.gather(*[_score_one(i) for i in range(batch_size)])

        results = _run_coro_sync(_score_all())
        teacher_ids_list = [row[0] for row in results]
        teacher_logprobs_list = [row[1] for row in results]
        added_prompt_tokens = [row[2] for row in results if row[2] is not None]

        write_back = TensorDict(
            {
                "teacher_ids": torch.nested.as_nested_tensor(teacher_ids_list, layout=torch.jagged),
                "teacher_logprobs": torch.nested.as_nested_tensor(teacher_logprobs_list, layout=torch.jagged),
                "self_distill_mask": torch.tensor([d.has_context for d in decisions], dtype=torch.bool),
            },
            batch_size=batch_size,
        )
        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=write_back)
        metrics.update(
            summarize_self_distillation_metrics(
                decisions,
                added_prompt_tokens=added_prompt_tokens,
                fallback_to_pg=self_cfg.fallback_to_pg,
            )
        )
        return batch

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int):
        """Run one synchronous V1 step, inserting self-teacher scoring while rollout servers are awake."""
        if not is_self_distillation_enabled(getattr(self, "distillation_config", None)):
            return super()._step_once(metrics, timing_raw, sample_batch_size)

        # 1. Sample rollouts from replay buffer.
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            # 2. Score reprompted self-teacher sequences using the live student rollout servers.
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                # Colocated RM requires sleeping rollout replicas first, then waking them to score.
                self.on_sample_end()
                with marked_timer("reward", timing_raw, color="yellow"):
                    batch = self._compute_reward_colocate(batch, metrics=metrics)
                with marked_timer("self_teacher_wake", timing_raw, color="magenta"):
                    self.checkpoint_manager.update_weights(self.global_steps)
                with marked_timer("self_teacher", timing_raw, color="magenta"):
                    batch = self._compute_self_teacher_log_probs(batch, metrics=metrics)
                self.on_sample_end()
            else:
                # Standard V1 reward loop already populated rm_scores; rollout servers are still awake.
                with marked_timer("self_teacher", timing_raw, color="magenta"):
                    batch = self._compute_self_teacher_log_probs(batch, metrics=metrics)
                self.on_sample_end()

        # 3. Balance batch across data parallel groups.
        batch = self._balance_batch(batch, metrics=metrics)

        # 4. Compute old_log_prob.
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 5. [OPTIONAL] Compute ref_log_prob.
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] Compute critic values.
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 7. Compute advantage and return (used for task rewards and PG fallback groups).
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 8. [OPTIONAL] Update critic.
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 9. Update actor.
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    def _update_actor(self, batch, metrics):
        """Mark the logits processor as needed when using hidden-state distillation.

        Verl keys its logits-processor dispatch off ``distillation_use_topk``
        (== ``loss_settings.use_topk``). Hidden-state (nitrobrew) modes set that
        flag False but still need the full ``student_logits`` tensor in the
        processor, so expand the trigger to ``use_topk or use_hidden_states``.
        """
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = False
        distillation_only = False
        if is_distillation_enabled(self.config.get("distillation")):
            loss_settings = self.distillation_config.distillation_loss.loss_settings
            # verl's engine gates the distillation logits processor on this flag, so
            # hidden-state (nitrobrew) modes must set it too; getattr because topk/
            # estimator settings come from verl's registry and lack the field.
            distillation_use_topk = bool(loss_settings.use_topk or getattr(loss_settings, "use_hidden_states", False))
            distillation_loss_cfg = self.distillation_config.distillation_loss
            needs_pg_fallback = bool(
                is_self_distillation_enabled(self.distillation_config)
                and self.distillation_config.self_teacher.fallback_to_pg
            )
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
                and not needs_pg_fallback
            )

            if distillation_loss_cfg.loss_mode in ("nitrobrew", "nitrobrew_reverse_kl"):
                self._ensure_teacher_unembeds()

        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "distillation_only": distillation_only,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)
        return batch

    # The rollout server resumes admission after every successful wake; this
    # bridge remains a safety net for holds not preceded by a wake (init).
    # TODO (long): check and fix the resume bridge on the rollout side.
    def on_init_end(self):
        super().on_init_end()
        self.checkpoint_manager.resume_generation_replicas()

    def on_step_end(self):
        super().on_step_end()
        self.checkpoint_manager.resume_generation_replicas()


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
        self._has_old_adapter = "old" in resolve_lora_config(config.actor_rollout_ref.model).adapters
        if self._has_old_adapter:
            raise NotImplementedError("OmniDirectPreferenceRayTrainer does not support old-policy adapters yet.")
        self._loss_fn = get_omni_loss_fn(loss_mode)
        self.global_batch_size = self.config.data.train_batch_size
        self.ref_in_actor = self._infer_ref_in_actor()
        self.use_rm = False

        if train_dataset is not None or self.config.data.get("train_files", None) is not None:
            self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler, val_sampler)

    def _infer_ref_in_actor(self) -> bool:
        return resolve_lora_config(self.config.actor_rollout_ref.model).enabled

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
        except (KeyError, TypeError, AttributeError, OmegaConf.errors.OmegaConfBaseException) as exc:
            raise RuntimeError("Failed to propagate trainer.total_training_steps to actor optimizer config.") from exc

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

        pools = [(pool, class_dict) for pool, class_dict in self.resource_pool_to_cls.items() if class_dict]
        master_port_range = OmegaConf.select(self.config.trainer, "ray_master_port_range")
        port_ranges = worker_group_port_ranges(master_port_range, len(pools))
        for (resource_pool, class_dict), port_range in zip(pools, port_ranges, strict=True):
            if port_range is not None:
                wg_kwargs["master_port_range"] = port_range
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
        else:
            raise ValueError(
                f"Unknown trainer.resume_mode={self.config.trainer.resume_mode!r}. "
                "Available options: ['disable', 'auto', 'resume_path']."
            )

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
