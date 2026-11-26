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

``PPOTrainer._setup`` exposes no rendezvous-port hook; ``_setup`` below is a faithful
copy of the pinned verl body with the port-range wiring DIFF-marked — re-diff it
against the verl pin whenever the pin moves.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid
from functools import partial
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from tensordict import NonTensorData, NonTensorStack, TensorDict
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.config import CriticConfig, DistillationConfig, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.losses import value_loss

from verl_omni.trainer.distillation.self_distillation import (
    TeacherContextDecision,
    align_teacher_sequence_outputs,
    build_multimodal_teacher_prompt,
    build_neutral_teacher_outputs,
    extract_sequence_rewards,
    select_teacher_contexts,
    summarize_self_distillation_metrics,
)
from verl_omni.utils.config import resolve_lora_config
from verl_omni.utils.net_utils import worker_group_port_ranges
from verl_omni.workers.config.omni.distillation import (
    is_self_distillation_enabled,
    validate_self_teacher_rollout_topk,
)

__all__ = ["OmniPPOTrainer"]

logger = logging.getLogger(__name__)


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


def _extract_media_payload(item: dict[str, Any], primary_key: str, url_key: str) -> Any:
    """Extract an inline media object or URL/path from a chat content block."""
    value = item.get(primary_key)
    if value is None:
        value = item.get(url_key)
    if isinstance(value, dict) and "url" in value:
        return value["url"]
    return value


def _extract_inline_multimodal_data(raw_prompt: Any) -> dict[str, list[Any]]:
    """Collect already-decoded media payloads or URLs directly from structured chat messages."""
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
            if item_type in ("image", "image_url"):
                payload = _extract_media_payload(item, "image", "image_url")
                if payload is not None:
                    images.append(payload)
            elif item_type in ("video", "video_url"):
                payload = _extract_media_payload(item, "video", "video_url")
                if payload is not None:
                    videos.append(payload)
            elif item_type in ("audio", "audio_url"):
                payload = _extract_media_payload(item, "audio", "audio_url")
                if payload is not None:
                    audios.append(payload)
    result: dict[str, list[Any]] = {}
    if images:
        result["images"] = images
    if videos:
        result["videos"] = videos
    if audios:
        result["audios"] = audios
    return result


def _raw_prompt_has_media_blocks(raw_prompt: Any) -> bool:
    """Return True if a structured message list contains any image, video, or audio block."""
    if not isinstance(raw_prompt, list):
        return False
    for msg in raw_prompt:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in (
                "image",
                "image_url",
                "video",
                "video_url",
                "audio",
                "audio_url",
            ):
                return True
    return False


class OmniPPOTrainer(PPOTrainer):
    """``PPOTrainer`` whose worker groups get rendezvous port sub-ranges.

    Must be the first base so this ``_setup`` precedes ``PPOTrainer._setup`` in the MRO.
    """

    def _init_resource_pool_mgr(self) -> None:
        """Initialize resource pools, skipping external ``Role.TeacherModel`` for self-distillation."""
        distillation_cfg = self.config.get("distillation")
        if is_self_distillation_enabled(distillation_cfg):
            # Self-distillation reuses the student's own rollout servers, so no
            # separate teacher pool is allocated and nnodes=0 / n_gpus_per_node=0 is valid.
            with open_dict(self.config):
                self.config.distillation.enabled = False
            try:
                super()._init_resource_pool_mgr()
            finally:
                with open_dict(self.config):
                    self.config.distillation.enabled = True
        else:
            super()._init_resource_pool_mgr()

    def get_teacher_client(self):
        """Return external teacher server clients, or ``None`` for self-distillation."""
        if getattr(self, "teacher_model_manager", None) is None:
            return None
        return super().get_teacher_client()

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
        # DIFF vs upstream: self-distillation (SDPO/OPSD) scores against the student's
        # own rollout servers rather than launching an external MultiTeacherModelManager.
        if is_self_distillation_enabled(self.config.get("distillation")):
            self.use_teacher_policy = False
            self.teacher_model_manager = None
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
            validate_self_teacher_rollout_topk(
                self.config.actor_rollout_ref.rollout,
                self.distillation_config,
            )
        elif self.use_teacher_policy:
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

    def _ensure_teacher_unembeds(self) -> None:
        """Load teacher lm_head unembedding matrices onto actor workers once for nitrobrew."""
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
        has_media_blocks = _raw_prompt_has_media_blocks(raw_prompt)
        needs_loading = has_media_blocks and (
            not inline_mm or any(isinstance(val, str) for bucket in inline_mm.values() for val in bucket)
        )
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
                logger.warning(
                    "Falling back to inline multimodal payloads during self-distillation reprompting: %s",
                    exc,
                )
        return inline_mm

    def _compute_self_teacher_log_probs(self, batch, metrics: dict):
        """Build reprompted multimodal teacher sequences and score them on the student's rollout server."""
        self_cfg = self.distillation_config.self_teacher
        is_separate_async = getattr(self, "standalone_server_manager", None) is not None
        if self_cfg.teacher_source == "ema" and is_separate_async:
            raise NotImplementedError(
                "distillation.self_teacher.teacher_source='ema' is not supported with omni_separate_async "
                "because standalone rollout replicas serve live policy rollouts concurrently. "
                "Use teacher_source='policy' for omni_separate_async or switch to omni_sync for EMA."
            )

        desired_fields = [
            "prompts",
            "responses",
            "response_mask",
            "uid",
            "rm_scores",
            "raw_prompt",
            "extra_info",
            "extra_fields",
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
        extra_fields_list = _extract_non_tensor_list(data, "extra_fields", batch_size)
        top_reward_extra_infos = _extract_non_tensor_list(data, "reward_extra_info", batch_size)
        reward_extra_infos: list[Any] | None = None
        if top_reward_extra_infos is not None or extra_fields_list is not None:
            merged_reward_extra: list[Any] = []
            for i in range(batch_size):
                merged_entry: dict[str, Any] = {}
                if extra_fields_list is not None and isinstance(extra_fields_list[i], dict):
                    nested_rei = extra_fields_list[i].get("reward_extra_info")
                    if isinstance(nested_rei, dict):
                        merged_entry.update(nested_rei)
                if top_reward_extra_infos is not None and isinstance(top_reward_extra_infos[i], dict):
                    merged_entry.update(top_reward_extra_infos[i])
                merged_reward_extra.append(merged_entry if merged_entry else None)
            if any(entry is not None for entry in merged_reward_extra):
                reward_extra_infos = merged_reward_extra

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
        rollout_cfg = OmegaConf.select(self.config, "actor_rollout_ref.rollout", default=None)
        effective_rollout_limit: int | None = None
        if rollout_cfg is not None:
            max_model_len = OmegaConf.select(rollout_cfg, "max_model_len", default=None)
            if max_model_len is not None:
                effective_rollout_limit = int(max_model_len)
            else:
                r_prompt = OmegaConf.select(rollout_cfg, "prompt_length", default=None)
                r_resp = OmegaConf.select(rollout_cfg, "response_length", default=None)
                if r_prompt is not None and r_resp is not None:
                    effective_rollout_limit = int(r_prompt) + int(r_resp)
        if effective_rollout_limit is None:
            d_prompt = OmegaConf.select(self.config, "data.max_prompt_length", default=None)
            d_resp = OmegaConf.select(self.config, "data.max_response_length", default=None)
            if d_prompt is not None and d_resp is not None:
                effective_rollout_limit = int(d_prompt) + int(d_resp)

        llm_client = self.get_llm_client()
        has_any_teacher_query = any(d.has_context for d in decisions)

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
                    return neutral_ids, neutral_lps, None, False

                per_sample_max_prompt = (
                    effective_rollout_limit - response_len - 1 if effective_rollout_limit is not None else None
                )
                if per_sample_max_prompt is not None and per_sample_max_prompt <= 0:
                    neutral_ids, neutral_lps = build_neutral_teacher_outputs(
                        student_prompt_len=student_prompt_len,
                        response_len=response_len,
                        topk=width,
                        pad_token_id=pad_token_id,
                    )
                    return neutral_ids, neutral_lps, None, False

                sample_extra_fields = (
                    extra_fields_list[i]
                    if extra_fields_list is not None and isinstance(extra_fields_list[i], dict)
                    else {}
                )
                turn_messages = sample_extra_fields.get("messages") or sample_extra_fields.get("turn_messages")
                turn_index = sample_extra_fields.get("turn_idx", sample_extra_fields.get("turn_index", 0))
                if turn_messages is not None:
                    sample_raw_prompt = turn_messages
                elif raw_prompts is not None:
                    if isinstance(turn_index, int) and turn_index > 0:
                        # Subsequent multi-turn step without turn-specific messages: keep the actual
                        # decoded student prefix for text-only turns, or fall back when media is present.
                        existing_mm = multi_modal_datas[i] if multi_modal_datas is not None else None
                        if _raw_prompt_has_media_blocks(raw_prompts[i]) or (
                            isinstance(existing_mm, dict)
                            and any(existing_mm.get(k) for k in ("images", "videos", "audios"))
                        ):
                            neutral_ids, neutral_lps = build_neutral_teacher_outputs(
                                student_prompt_len=student_prompt_len,
                                response_len=response_len,
                                topk=width,
                                pad_token_id=pad_token_id,
                            )
                            return neutral_ids, neutral_lps, None, False
                        sample_raw_prompt = self.tokenizer.decode(student_prompt_ids, skip_special_tokens=False)
                    else:
                        sample_raw_prompt = raw_prompts[i]
                else:
                    sample_raw_prompt = self.tokenizer.decode(student_prompt_ids, skip_special_tokens=False)

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
                    max_prompt_tokens=per_sample_max_prompt,
                )
                if not reprompted.context_retained:
                    neutral_ids, neutral_lps = build_neutral_teacher_outputs(
                        student_prompt_len=student_prompt_len,
                        response_len=response_len,
                        topk=width,
                        pad_token_id=pad_token_id,
                    )
                    return neutral_ids, neutral_lps, None, False

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
                return aligned_ids, aligned_lps, reprompted.added_prompt_tokens, True

            return await asyncio.gather(*[_score_one(i) for i in range(batch_size)])

        ema_swapped = False
        if self_cfg.teacher_source == "ema" and has_any_teacher_query:
            self.actor_rollout_wg.swap_ema_weights(use_ema=True)
            ema_swapped = True
            try:
                self.checkpoint_manager.update_weights(getattr(self, "global_steps", 0))
                resume_fn = getattr(self.checkpoint_manager, "resume_generation_replicas", None)
                if callable(resume_fn):
                    resume_fn()
                results = _run_coro_sync(_score_all())
            finally:
                if ema_swapped:
                    self.actor_rollout_wg.swap_ema_weights(use_ema=False)
        else:
            results = _run_coro_sync(_score_all())

        teacher_ids_list = [row[0] for row in results]
        teacher_logprobs_list = [row[1] for row in results]
        added_prompt_tokens = [row[2] for row in results if row[2] is not None]
        effective_mask = [bool(row[3]) for row in results]
        effective_decisions = [
            decisions[i] if effective_mask[i] else TeacherContextDecision(has_context=False, source="none")
            for i in range(batch_size)
        ]

        write_back = TensorDict(
            {
                "teacher_ids": torch.nested.as_nested_tensor(teacher_ids_list, layout=torch.jagged),
                "teacher_logprobs": torch.nested.as_nested_tensor(teacher_logprobs_list, layout=torch.jagged),
                "self_distill_mask": torch.tensor(effective_mask, dtype=torch.bool),
            },
            batch_size=batch_size,
        )
        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=write_back)
        metrics.update(
            summarize_self_distillation_metrics(
                effective_decisions,
                added_prompt_tokens=added_prompt_tokens,
                fallback_to_pg=self_cfg.fallback_to_pg,
            )
        )
        return batch

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int):
        """Run one V1 step, inserting self-teacher scoring while rollout servers are awake."""
        if not is_self_distillation_enabled(getattr(self, "distillation_config", None)):
            return super()._step_once(metrics, timing_raw, sample_batch_size)

        is_separate_async = getattr(self, "standalone_server_manager", None) is not None

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

            # 2. Score reprompted self-teacher sequences using the student rollout servers.
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                if is_separate_async:
                    self.on_sample_end()
                    with marked_timer("reward", timing_raw, color="yellow"):
                        batch = self._compute_reward_colocate(batch, metrics=metrics)
                    with marked_timer("self_teacher", timing_raw, color="magenta"):
                        batch = self._compute_self_teacher_log_probs(batch, metrics=metrics)
                else:
                    # Colocated RM requires sleeping hybrid rollout replicas first, then waking them to score.
                    self.on_sample_end()
                    with marked_timer("reward", timing_raw, color="yellow"):
                        batch = self._compute_reward_colocate(batch, metrics=metrics)
                    with marked_timer("self_teacher_wake", timing_raw, color="magenta"):
                        self.checkpoint_manager.update_weights(self.global_steps)
                        resume_fn = getattr(self.checkpoint_manager, "resume_generation_replicas", None)
                        if callable(resume_fn):
                            resume_fn()
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
        """Mark the logits processor as needed for hidden-state and self-distillation paths.

        Verl keys its logits-processor dispatch off ``distillation_use_topk``
        (== ``loss_settings.use_topk``). Hidden-state (nitrobrew) modes set that
        flag False but still need the full ``student_logits`` tensor in the
        processor, so expand the trigger to ``use_topk or use_hidden_states``.
        """
        if not is_distillation_enabled(self.config.get("distillation")):
            return super()._update_actor(batch, metrics)

        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        loss_settings = self.distillation_config.distillation_loss.loss_settings
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
        score_centering = OmegaConf.select(
            self.config, "actor_rollout_ref.actor.policy_loss.score_centering", default=None
        )
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
        if score_centering is not None:
            extra_info["score_centering"] = bool(score_centering)
        batch.extra_info.update(extra_info)

        output = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)
        return batch
