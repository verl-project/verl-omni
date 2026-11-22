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
"""Colocated asynchronous FlowGRPO with whole-sample diffusion retries."""

import logging
import math
import time
from pathlib import Path
from typing import TypedDict

import msgspec
import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.utils.debug import marked_timer
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID

from verl_omni.agent_loop.diffusion_agent_loop import _config_to_sampling_dict
from verl_omni.trainer.diffusion.v1.tq_utils import _parse_tq_key
from verl_omni.trainer.diffusion.v1.trainer_base import register_diffusion_trainer
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.utils.config import resolve_lora_config
from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager
from verl_omni.workers.rollout.diffusion_llm_server import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DiffusionWholeSampleRetryLLMServerClient,
)

logger = logging.getLogger(__name__)

_STEP_SNAPSHOT_BUDGET = 64 << 20


class _RolloutHandoff(TypedDict):
    policy_ref: str
    model_fingerprint: str
    requests: dict[str, str]
    snapshots: list[dict[str, bytes]]
    adapters: list[bytes]


class _RolloutCheckpoint(TypedDict):
    version: int
    step: int
    handoff: _RolloutHandoff | None


class _LoRAIdentity(TypedDict):
    adapter_id: int


class _ColocateReplayBuffer(ReplayBufferAsync):
    sampling_deadline: float | None = None

    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:
        if self.sampling_deadline is not None:
            remaining = self.sampling_deadline - time.monotonic()
            if remaining <= self.poll_interval:
                if remaining > 0:
                    time.sleep(remaining)
                raise TimeoutError("colocate_async timed out waiting for complete rollout groups")
        return super()._wait_for_next_poll(partition_id, last_debug_time)


class _DeterministicColocateReplayBuffer(_ColocateReplayBuffer):
    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size):
        if partition_id != "train":
            return super()._select_prompt_uids(partition_id, sampleable_keys, batch_size)

        versions = dict(self.prompt_global_steps[partition_id])
        rows = dict(self.partitions[partition_id])
        generation_order = {
            _parse_tq_key(key)[0]: (tag["gen_batch_seq"], tag["prompt_index"])
            for key, tag in rows.items()
            if "gen_batch_seq" in tag and "prompt_index" in tag
        }
        selected = sorted(
            sampleable_keys,
            key=lambda uid: (versions.get(uid, 0), generation_order.get(uid, (math.inf, math.inf)), uid),
        )
        return selected[:batch_size], rows, versions


@register_diffusion_trainer("colocate_async")
class PolicyGradientDiffusionTrainerV1ColocateAsync(PolicyGradientDiffusionTrainerV1Sync):
    """Collect complete groups, reclaim rollout GPUs, train, then commit weights.

    Pending requests wait outside the rollout servers while the actor owns the
    GPUs. Aborted requests restart from the original prompt and noise seed.
    Actor PPO mini-batches run within one trainer step and one weight sync.
    """

    _resume_denoising = False
    _next_lora_adapter_id: int | None = None
    _checkpoint_handoff: _RolloutHandoff | None = None
    _checkpoint_outputs: dict | None = None

    def __init__(self, config):
        rollout = config.actor_rollout_ref.rollout
        if rollout.nnodes != 0 or rollout.checkpoint_engine.backend != "naive":
            raise ValueError("colocate_async requires colocated rollout (nnodes=0) and checkpoint_engine.backend=naive")
        if not rollout.free_cache_engine and rollout.step_execution:
            raise ValueError("resident colocate_async requires request-level execution (step_execution=false)")
        self._resume_denoising = config.trainer.v1.colocate_async.get("resume_denoising", False)
        if not isinstance(self._resume_denoising, bool):
            raise ValueError("colocate_async.resume_denoising must be a boolean")
        if self._resume_denoising:
            lora = resolve_lora_config(config.actor_rollout_ref.model)
            if not rollout.step_execution or not lora.enabled or lora.merge:
                raise ValueError("resume_denoising requires step execution and runtime LoRA")
            capacity = (rollout.engine_kwargs.get("vllm_omni") or {}).get("max_cpu_loras", 1)
            if capacity != 2:
                raise ValueError("resume_denoising requires max_cpu_loras=2")
            self._checkpoint_manager_cls = OmniCheckpointEngineManager
        if config.trainer.v1.colocate_async.get("parameter_sync_step", 1) != 1:
            raise ValueError("colocate_async supports one parameter sync per step (parameter_sync_step=1)")
        warmup = config.trainer.v1.colocate_async.num_warmup_batches
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("colocate_async.num_warmup_batches must be a nonnegative integer")
        if (
            config.algorithm.trainer_type != "policy_gradient"
            or config.actor_rollout_ref.model.algorithm != "flow_grpo"
        ):
            raise ValueError("colocate_async currently supports FlowGRPO policy-gradient training")
        train_batch_size = config.data.train_batch_size
        mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in (train_batch_size, mini_batch_size)
        ):
            raise ValueError("colocate_async requires positive integer train_batch_size and ppo_mini_batch_size")
        if train_batch_size % mini_batch_size:
            raise ValueError("colocate_async requires train_batch_size divisible by ppo_mini_batch_size")
        super().__init__(config)

    def _build_replay_buffer(self):
        deterministic = OmegaConf.select(self.config, "actor_rollout_ref.rollout.full_determinism", default=False)
        buffer_cls = _DeterministicColocateReplayBuffer if deterministic else _ColocateReplayBuffer
        return super()._build_replay_buffer(async_replay_buffer_cls=buffer_cls)

    def get_llm_client(self):
        """Retry whole samples while the hybrid pool is temporarily unavailable."""
        return self.llm_server_manager.get_client(
            client_cls=DiffusionWholeSampleRetryLLMServerClient,
            only_hybrid=True,
            request_timeout_s=self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S),
        )

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        outputs, self._checkpoint_outputs = self._checkpoint_outputs, None
        return super()._reissue_inflight_prompts(
            partition_id,
            preserve_complete_groups=self._resume_denoising,
            checkpoint_outputs=(outputs or {}) if self._resume_denoising else None,
        )

    def _load_rollout_checkpoint(self, checkpoint_folder: str) -> None:
        if not self._resume_denoising or self.config.trainer.get("val_only", False):
            return
        path = Path(checkpoint_folder) / "diffusion_rollout.msgpack"
        try:
            record = msgspec.msgpack.decode(path.read_bytes(), type=_RolloutCheckpoint)
        except FileNotFoundError:
            return
        except (OSError, msgspec.DecodeError):
            logger.warning("Ignoring unreadable diffusion checkpoint attachment", exc_info=True)
            return
        handoff = record["handoff"]
        if handoff is None:
            return
        request_ids = set(handoff["requests"].values())
        if (
            record["version"] != 2
            or record["step"] != self.global_steps
            or handoff["policy_ref"] != f"rollout-step-{self.global_steps - 1}"
            or len(handoff["snapshots"]) != 1
            or len(handoff["adapters"]) != 1
            or not request_ids
            or len(request_ids) != len(handoff["requests"])
            or request_ids != handoff["snapshots"][0].keys()
        ):
            logger.warning("Ignoring incompatible diffusion checkpoint attachment")
            return
        self._checkpoint_handoff = handoff

    def on_init_end(self):
        handoff, self._checkpoint_handoff = self._checkpoint_handoff, None
        rollout = self.config.actor_rollout_ref.rollout
        if handoff is None or "dummy" in rollout.get("load_format", ""):
            return super().on_init_end()
        try:
            adapter_id = msgspec.msgpack.decode(handoff["adapters"][0], type=_LoRAIdentity)["adapter_id"]
        except msgspec.DecodeError:
            logger.warning("Ignoring unreadable checkpoint LoRA identity")
            return super().on_init_end()
        if adapter_id not in {VLLM_LORA_INT_ID, VLLM_LORA_INT_ID + 1}:
            logger.warning("Ignoring incompatible checkpoint LoRA slot %s", adapter_id)
            return super().on_init_end()

        items = (tq.kv_list("train") or {}).get("train", {})
        requests = {}
        for key, inner in handoff["requests"].items():
            uid, session, turn = _parse_tq_key(key)
            prompt = items.get(uid, {})
            if (
                0 <= session < rollout.n
                and turn == 0
                and key == f"{uid}_{session}_0"
                and prompt.get("is_prompt")
                and prompt["status"] in ("pending", "running")
                and key not in items
            ):
                requests[key] = inner
        threshold = self.config.trainer.v1.sampler.max_off_policy_threshold
        if not requests or (threshold is not None and threshold < 2):
            return super().on_init_end()
        handoff = {**handoff, "requests": requests}
        manager = self.llm_server_manager
        (server,) = manager.server_handles
        timeout = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
        ray.get(manager.global_load_balancer.remove_servers.remote(manager.server_addresses))
        self.checkpoint_manager.abort_replicas()
        next_slot = VLLM_LORA_INT_ID + int(adapter_id == VLLM_LORA_INT_ID)
        self.checkpoint_manager.update_weights(self.global_steps, lora_adapter_id=next_slot)
        sampling = {
            **_config_to_sampling_dict(rollout.pipeline),
            **_config_to_sampling_dict(rollout.algo),
            "logprobs": rollout.calculate_log_probs,
        }
        self._checkpoint_outputs = ray.get(
            server.recover_diffusion_checkpoint.remote(handoff, adapter_id, sampling, timeout, _STEP_SNAPSHOT_BUDGET),
            timeout=4 * timeout,
        )
        logger.info("Recovered %d diffusion checkpoint sessions", len(self._checkpoint_outputs))
        self.checkpoint_manager.resume_generation_replicas()
        ray.get(
            manager.global_load_balancer.add_servers.remote(
                dict(zip(manager.server_addresses, manager.server_handles, strict=True))
            )
        )

    def _save_rollout_checkpoint(self, checkpoint_folder: str) -> None:
        if not self._resume_denoising:
            return
        items = (tq.kv_list("train") or {}).get("train", {})
        session_keys = {
            f"{uid}_{session}_0"
            for uid, tag in items.items()
            if tag.get("is_prompt") and tag["status"] in ("pending", "running")
            for session in range(self.config.actor_rollout_ref.rollout.n)
        }
        handoff = None
        if session_keys:
            timeout = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
            server = self.llm_server_manager.server_handles[0]
            handoff = ray.get(
                server.export_diffusion_checkpoint.remote(session_keys, timeout, _STEP_SNAPSHOT_BUDGET), timeout=timeout
            )
        path = Path(checkpoint_folder) / "diffusion_rollout.msgpack"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(msgspec.msgpack.encode({"version": 2, "step": self.global_steps, "handoff": handoff}))
        temporary.replace(path)

    def on_train_begin(self):
        """Top up warmup without duplicating restored prompt groups."""
        # Restored terminal groups count toward warmup as well as in-flight groups.
        items = (tq.kv_list("train") or {}).get("train", {})
        queued = sum(tag.get("is_prompt", False) for tag in items.values())
        target = self.config.trainer.v1.colocate_async.num_warmup_batches * self.config.data.train_batch_size
        if target > queued:
            self._add_prompts_to_generate(target - queued)

    def on_sample_end(self):
        """Close routing and reclaim rollout resources before actor computation."""
        self._quiesce_rollout(self.timing_raw)
        if self.curr_step_profile:
            self._stop_rollout_profiling()

    def _quiesce_rollout(self, timing_raw):
        # Remove routing before aborting: retries wait until committed weights are ready.
        manager = self.llm_server_manager
        snapshot = self._resume_denoising
        if snapshot:
            if len(manager.server_handles) != 1:
                raise ValueError("resume_denoising requires one rollout replica")
            sampler = self.config.trainer.v1.sampler
            # A continued request uses the previous policy. The next sample
            # step sees a span of two, so drop/1 would immediately evict it.
            snapshot = sampler.max_off_policy_strategy != "drop" or sampler.max_off_policy_threshold != 1
        with marked_timer("colocate_handoff_to_actor", timing_raw):
            with marked_timer("colocate_route_close", timing_raw):
                ray.get(manager.global_load_balancer.remove_servers.remote(manager.server_addresses))
            if snapshot:
                with marked_timer("colocate_snapshot", timing_raw):
                    timeout = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
                    self._next_lora_adapter_id = ray.get(
                        manager.server_handles[0].suspend_diffusion_steps.remote(timeout, _STEP_SNAPSHOT_BUDGET),
                        timeout=timeout,
                    )
            else:
                with marked_timer("colocate_abort", timing_raw):
                    self.checkpoint_manager.abort_replicas()
            with marked_timer("colocate_sleep", timing_raw):
                self.checkpoint_manager.sleep_replicas()

    def on_step_end(self):
        """Reopen routing only after committing weights and resuming generation."""
        self._resume_rollout(self.timing_raw)

    def close_rollout(self):
        manager = getattr(self, "llm_server_manager", None)
        if manager is not None and self.config.actor_rollout_ref.rollout.name == "vllm_omni":
            ray.get([server.close_diffusion_engine.remote() for server in manager.server_handles], timeout=30)

    def _resume_rollout(self, timing_raw):
        manager = self.llm_server_manager
        adapter_id = self._next_lora_adapter_id
        versioned = {"lora_adapter_id": adapter_id} if adapter_id is not None else {}
        with marked_timer("update_weights", timing_raw, color="red"):
            with marked_timer("colocate_weight_sync", timing_raw):
                self.checkpoint_manager.update_weights(self.global_steps, **versioned)
            with marked_timer("colocate_resume", timing_raw):
                if adapter_id is not None:
                    timeout = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
                    ray.get(
                        manager.server_handles[0].restore_diffusion_steps.remote(timeout, _STEP_SNAPSHOT_BUDGET),
                        timeout=timeout,
                    )
                    self._next_lora_adapter_id = None
                self.checkpoint_manager.resume_generation_replicas()
            servers = dict(zip(manager.server_addresses, manager.server_handles, strict=True))
            with marked_timer("colocate_route_open", timing_raw):
                ray.get(manager.global_load_balancer.add_servers.remote(servers))

    def prepare_step(self) -> dict:
        """Feed normally and observe group counts from the existing replay snapshot."""
        metrics = super().prepare_step()
        replay = self.replay_buffer
        metrics["rollout/queue/sampleable_groups"] = replay.get_sampleable_count(self.global_steps, "train")
        for name in ("pending", "running", "finished", "failure"):
            metrics[f"rollout/queue/{name}_groups"] = len(getattr(replay, f"{name}_keys")["train"])
        return metrics

    def _sample_training_batch(self, batch_size):
        replay = self.replay_buffer
        refill = replay.refill_fn
        timeout_s = self.config.trainer.v1.colocate_async.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
        deadline = time.monotonic() + timeout_s

        def refill_before_deadline(num_prompts):
            if time.monotonic() >= deadline:
                raise TimeoutError("colocate_async sampling exhausted its rollout refill deadline")
            return refill(num_prompts)

        replay.refill_fn = refill_before_deadline
        replay.sampling_deadline = deadline
        try:
            batch, metrics = super()._sample_training_batch(batch_size)
        finally:
            replay.refill_fn = refill
            replay.sampling_deadline = None
        tags = [tag for tag in batch.tags if not tag.get("is_padding", False)]
        metrics["rollout/trained_trajectories/retries"] = sum(tag.get("rollout_retries", 0) for tag in tags)
        metrics["rollout/trained_trajectories/retry_wait_s"] = sum(tag.get("rollout_retry_wait_s", 0) for tag in tags)
        return batch, metrics

    def _compute_metrics(self, batch_meta, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch_meta, metrics, timing_raw, global_steps, epoch)
        metrics["perf/colocate_gpu_handoff_s"] = timing_raw["colocate_handoff_to_actor"] + timing_raw["update_weights"]
        metrics["perf/trained_trajectories_per_s"] = (
            sum(not tag.get("is_padding", False) for tag in batch_meta.tags) / timing_raw["step"]
        )
