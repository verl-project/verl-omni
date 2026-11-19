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
"""Bagel UND+GEN Co-RL (Joint-Training) trainer: one post-gather composite update + weight publish.

The class is the orchestrator only. Config retargeting, the dual-lane batch gather, the GEN
lane, step metrics, the UND AR replica and the reward loop each live in a ``bagel_corl_*``
sibling module and are called from the hooks below.
"""

from __future__ import annotations

import logging
import os

from omegaconf import open_dict
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer

from verl_omni.trainer.omni import bagel_corl_batch as batch_gather
from verl_omni.trainer.omni import bagel_corl_gen_lane as gen_lane
from verl_omni.trainer.omni import bagel_corl_metrics as step_metrics
from verl_omni.trainer.omni import bagel_corl_und_ar as und_ar
from verl_omni.trainer.omni import bagel_corl_und_ar_runtime as und_ar_runtime
from verl_omni.trainer.omni.bagel_corl_config import expected_s, gen_adv_estimator, rewrite_bagel_corl_configs
from verl_omni.trainer.omni.bagel_corl_diff_v1 import DiffusionV1GenLane
from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto
from verl_omni.trainer.omni.bagel_corl_replay_buffer import install_auxiliary_row_filter
from verl_omni.trainer.omni.bagel_corl_reward_loop import (
    bagel_rm_enabled,
    compute_reward_colocate,
    named_reward_models_enabled,
)
from verl_omni.trainer.omni.bagel_corl_und_adv import GRPO_STD_FLOOR  # noqa: F401  (registers the estimator)
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)


@register_trainer("bagel_corl_sync")
class OmniBagelCoRLTrainerSync(OmniPPOTrainerSync):
    """Synchronous Bagel Co-RL (Joint-Training): serial J-episode gather, then one UND+GEN optimizer step."""

    METRIC_EMA_BETA = step_metrics.METRIC_EMA_BETA
    METRIC_EMA_KEYS = step_metrics.METRIC_EMA_KEYS
    DIVERGENCE_KL_LIMIT = step_metrics.DIVERGENCE_KL_LIMIT
    DIVERGENCE_GRAD_NORM_LIMIT = step_metrics.DIVERGENCE_GRAD_NORM_LIMIT
    UND_HF_SUBCONFIG = und_ar.UND_HF_SUBCONFIG
    UND_AR_LOAD_FORMAT = und_ar.UND_AR_LOAD_FORMAT
    UND_AR_POLICY_STAGE_ID = und_ar.UND_AR_POLICY_STAGE_ID
    UND_AR_SYNC_CHECK_PARAM = und_ar.UND_AR_SYNC_CHECK_PARAM
    UND_AR_SYNC_CHECK_PARAMS = und_ar.UND_AR_SYNC_CHECK_PARAMS

    def _rewrite_bagel_corl_configs(self) -> None:
        """Retarget model/rollout to diffusion configs; inject actor ``diffusion_loss``."""
        rewrite_bagel_corl_configs(self.config)

    def _init_tokenizer(self):
        self._rewrite_bagel_corl_configs()
        model_config: DiffusionModelConfig = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.model, DiffusionModelConfig
        )
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _build_replay_buffer(self):
        """Build the sync ``ReplayBuffer`` under a temporary ``trainer_mode`` and filter GEN seed rows out."""
        original_mode = self.trainer_mode
        self.trainer_mode = "sync"
        try:
            buffer = super()._build_replay_buffer()
        finally:
            self.trainer_mode = original_mode
        return install_auxiliary_row_filter(buffer)

    def _init_resource_pool_mgr(self):
        """Use VeRL-Omni ``ActorRolloutRefWorker`` and honour ``trainer.max_colocate_count`` (reward engines)."""
        import ray

        super()._init_resource_pool_mgr()
        raw = self.config.trainer.get("max_colocate_count", None)
        max_colocate_count = 3 if raw is None else int(raw)
        if max_colocate_count < 1:
            raise ValueError("config.trainer.max_colocate_count must be >= 1")
        self.resource_pool_manager.max_colocate_count = max_colocate_count
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker as OmniActorRolloutRefWorker

        remote_cls = ray.remote(OmniActorRolloutRefWorker)
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = remote_cls

    def _expected_s(self) -> int:
        """Seeds per ``generate_image`` call (RFC §5 knob; no silent default)."""
        return expected_s(self.config)

    def _gen_adv_estimator(self) -> str:
        """GEN FlowGRPO estimator, never ``algorithm.adv_estimator``."""
        return gen_adv_estimator(self.config)

    @staticmethod
    def _extra_info(batch) -> dict:
        return batch_gather.extra_info(batch)

    def _diffusion_v1_gen_lane(self) -> DiffusionV1GenLane:
        """GEN hooks from ``PolicyGradientDiffusionTrainerV1`` on this job's workers, cached per worker group."""
        lane = getattr(self, "_diff_v1_gen_lane", None)
        wg = getattr(self, "actor_rollout_wg", None)
        gen_estimator = self._gen_adv_estimator()
        if (
            lane is None
            or getattr(lane, "actor_rollout_wg", None) is not wg
            or getattr(lane, "gen_adv_estimator", None) != gen_estimator
        ):
            self._diff_v1_gen_lane = DiffusionV1GenLane(self, gen_adv_estimator=gen_estimator)
        return self._diff_v1_gen_lane

    def _gen_dispatch_divisor(self) -> int:
        return batch_gather.gen_dispatch_divisor(getattr(self, "actor_rollout_wg", None), self.config)

    def _gen_batch_from_step(self, batch) -> list:
        return batch_gather.gen_batch_from_step(batch, self._extra_info(batch))

    def _compute_old_log_prob(self, batch, metrics: dict):
        """UND: PPO V1 token old-logprob. GEN: diffusion V1 ``infer_actor_batch`` (required when K>0)."""
        extra = self._extra_info(batch)
        gen_rows = self._gen_batch_from_step(batch)
        logger.info(
            "bagel_corl old_log_prob batch_rows=%d roles=%s gen_rows=%d",
            len(batch),
            batch_gather.bagel_role_histogram(getattr(batch, "tags", None)),
            len(gen_rows),
        )
        proto = build_gen_flowgrpo_proto(gen_rows)
        if proto is None:
            return super()._compute_old_log_prob(batch, metrics)
        if "all_latents" not in proto.batch.keys():
            raise RuntimeError(
                "bagel_corl GEN old_log_prob requires all_latents; refuse rollout_log_probs-only recompute skip."
            )
        if getattr(self, "actor_rollout_wg", None) is None:
            metrics["gen/old_log_prob_recomputed"] = 0.0
            return super()._compute_old_log_prob(batch, metrics)
        old = gen_lane.gen_old_log_prob(proto, self._diffusion_v1_gen_lane(), self._gen_dispatch_divisor())
        extra["bagel_corl_gen_old"] = old.batch
        metrics["gen/old_log_prob_recomputed"] = 1.0
        return super()._compute_old_log_prob(batch, metrics)

    def _compute_advantage(self, batch, metrics: dict):
        """Same step, separate tensors: GEN FlowGRPO -> ``extra['bagel_corl_gen']``; UND token GRPO via ``super``."""
        extra = self._extra_info(batch)
        algo = self.config.algorithm
        gen_estimator = self._gen_adv_estimator()
        und_estimator = str(algo.get("adv_estimator", "grpo"))
        gen_lane.check_estimators_disjoint(gen_estimator, und_estimator)
        gen_rows = self._gen_batch_from_step(batch)
        lane = self._diffusion_v1_gen_lane() if getattr(self, "actor_rollout_wg", None) is not None else None
        gen_proto, gen_metrics = gen_lane.gen_advantage(
            build_gen_flowgrpo_proto(gen_rows), gen_rows, extra=extra, algo=algo, gen_estimator=gen_estimator, lane=lane
        )
        if gen_proto is not None:
            gen_lane.gen_proto_metrics(gen_proto, metrics)
        has_complete = gen_lane.stamp_gen_lane(extra, metrics, gen_proto, gen_metrics)
        logger.info(
            "bagel_corl_sync advantage und=%s gen=%s skip_gen=%s J=%s K=%s K0=%s",
            und_estimator,
            gen_estimator,
            extra["skip_gen"],
            metrics.get("episode/J"),
            metrics.get("episode/K"),
            metrics.get("episode/K_zero"),
        )
        result = super()._compute_advantage(batch, metrics)
        gen_lane.carry_gen_lane(extra, self._extra_info(result), has_complete)
        return result

    def _update_actor(self, batch, metrics: dict):
        """Single composite ``update_actor`` after the N-sibling gather. Skip GEN when no complete S-groups."""
        extra = self._extra_info(batch)
        gen_lane.validate_batch_ownership(batch, extra)
        gen_lane.attach_gen_flags(batch, extra, metrics)
        logger.info(
            "bagel_corl_sync update_actor skip_gen=%s policy_version=%s J=%s K=%s K0=%s",
            extra.get("skip_gen"),
            getattr(self, "global_steps", None),
            metrics.get("episode/J"),
            metrics.get("episode/K"),
            metrics.get("episode/K_zero"),
        )
        updated = super()._update_actor(batch, metrics)
        step_metrics.publish_lane_metrics(metrics)
        self._divergence_tripwire(metrics)
        return updated

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
        state = self.__dict__.setdefault("_metric_ema", {})
        step_metrics.update_metric_ema(state, metrics, keys=self.METRIC_EMA_KEYS, beta=self.METRIC_EMA_BETA)

    def _divergence_tripwire(self, metrics: dict) -> None:
        """Warn once per excursion (or abort with ``BAGEL_CORL_DIVERGENCE_ABORT=1``) on runaway KL / grad norm."""
        triggers = step_metrics.divergence_triggers(
            metrics, kl_limit=self.DIVERGENCE_KL_LIMIT, grad_norm_limit=self.DIVERGENCE_GRAD_NORM_LIMIT
        )
        metrics["bagel_corl/divergence_warning"] = 1.0 if triggers else 0.0
        if not triggers:
            self._divergence_warned = False
            return
        if getattr(self, "_divergence_warned", False):
            return
        self._divergence_warned = True
        message = step_metrics.divergence_message(getattr(self, "global_steps", None), metrics, triggers)
        if os.environ.get("BAGEL_CORL_DIVERGENCE_ABORT", "0") == "1":
            raise RuntimeError(message)
        logger.warning("%s Set BAGEL_CORL_DIVERGENCE_ABORT=1 to fail instead of warn.", message)

    def _und_deploy_config_path(self) -> str | None:
        return und_ar.und_deploy_config_path(self.config)

    def _build_und_ar_entrypoint_config(self):
        """Clone entry config for a standalone UND AR ``LLMServerManager`` (Omni model + ``output_mode=ar``)."""
        return und_ar.build_und_ar_entrypoint_config(
            self.config, load_format=self.UND_AR_LOAD_FORMAT, policy_stage_id=self.UND_AR_POLICY_STAGE_ID
        )

    def _und_actor_pool_size(self) -> int | None:
        """Width of the actor pool the UND AR replica is carved out of, or ``None`` when unresolved."""
        try:
            actor_pool = self.resource_pool_manager.get_resource_pool(self._actor_role())
        except Exception:  # noqa: BLE001 - a missing pool is reported by _ensure_dual_role_rollout
            return None
        world_size = getattr(actor_pool, "world_size", None)
        return int(world_size) if world_size else None

    def _actor_role(self) -> Role:
        return Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout

    def _ensure_dual_role_rollout(self) -> None:
        """Start UND AR colocated on the actor GPU pool; wrap GEN+UND behind one client."""
        if getattr(self, "_bagel_dual_role_ready", False):
            return
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        if not bool(agent.get("und_ar_serving_ready")):
            raise RuntimeError(
                "bagel_corl_sync dual-role init requires agent.und_ar_serving_ready=True "
                "(verify Hermes generate_image calls on the UND AR replica first)"
            )
        und_cfg = self._build_und_ar_entrypoint_config()
        und_rollout = und_cfg.actor_rollout_ref.rollout
        und_load_format = str(und_rollout.get("load_format") or "").strip().lower()
        if und_load_format in ("", "dummy"):
            raise RuntimeError(
                "bagel_corl_sync: the UND AR replica would be launched with "
                f"load_format={und_load_format or '<unset>'}, but it is a COLOCATED replica that "
                "the naive checkpoint backend never publishes to -- it would decode from random "
                "weights for the whole run. Pin actor_rollout_ref.rollout.load_format to "
                f"{self.UND_AR_LOAD_FORMAT!r} in _build_und_ar_entrypoint_config."
            )
        und_n_gpus = int(agent.get("und_n_gpus") or 1)
        if und_n_gpus < 1:
            raise ValueError(f"agent.und_n_gpus must be >= 1, got {und_n_gpus}")
        und_deploy = self._und_deploy_config_path()
        und_ar.validate_und_ar_pool(und_rollout, und_n_gpus, und_deploy, self._und_actor_pool_size())
        gen_manager = self.llm_server_manager
        und_replica = und_ar_runtime.launch_und_ar_replica(
            und_cfg,
            actor_pool=self.resource_pool_manager.get_resource_pool(self._actor_role()),
            replica_rank=len(gen_manager.rollout_replicas),
            und_n_gpus=und_n_gpus,
            und_deploy=und_deploy,
            actor_model=self.config.actor_rollout_ref.model,
        )
        self.und_rollout_replicas = [und_replica]
        self.checkpoint_manager.add_replicas(self.und_rollout_replicas)
        logger.info(
            "bagel_corl_sync UND AR replica registered for sleep/wake replica=%s und_n_gpus=%s "
            "(weights published by _publish_und_ar_weights)",
            getattr(und_replica, "replica_rank", None),
            und_n_gpus,
        )
        self._bagel_dual_client = und_ar_runtime.build_dual_role_client(self.config, und_cfg, und_replica, gen_manager)
        self._bagel_dual_role_ready = True

    def get_llm_client(self):
        """Return the dual-role client (UND AR + GEN diffusion)."""
        self._ensure_dual_role_rollout()
        return self._bagel_dual_client

    def _gen_rollout_replicas(self) -> list:
        """Every registered replica except the UND AR pool (identity comparison)."""
        registered = getattr(getattr(self, "checkpoint_manager", None), "replicas", None) or []
        und_ids = {id(r) for r in (getattr(self, "und_rollout_replicas", None) or [])}
        return [r for r in registered if id(r) not in und_ids]

    def _wake_und_rollout_replicas(self) -> None:
        und_ar_runtime.wake_replicas(getattr(self, "und_rollout_replicas", None) or [], label="UND AR")

    def _wake_rollout_replicas(self) -> None:
        """Wake *both* colocated pools after a sleep; the parent's tagless wake leaves ``kv_cache`` set."""
        und = getattr(self, "und_rollout_replicas", None) or []
        gen = self._gen_rollout_replicas()
        logger.info(
            "bagel_corl_sync step=%s wake topology: und_ar=%d gen=%d (both woken with ['weights', 'kv_cache'])",
            getattr(self, "global_steps", None),
            len(und),
            len(gen),
        )
        und_ar_runtime.wake_replicas(und, label="UND AR")
        und_ar_runtime.wake_replicas(gen, label="GEN")

    def _publish_und_ar_weights(self) -> None:
        """Push the merged actor policy into the UND AR Thinker, verify it landed, record sync metrics."""
        replicas = getattr(self, "und_rollout_replicas", None) or []
        if not replicas:
            return
        receiver_ranks = und_ar.und_stage_devices(self._und_deploy_config_path()).get(self.UND_AR_POLICY_STAGE_ID)
        if not receiver_ranks:
            raise RuntimeError(
                f"bagel_corl_sync: UND AR deploy config declares no devices for stage "
                f"{self.UND_AR_POLICY_STAGE_ID}; cannot route the policy push"
            )
        num_tensors, fingerprint, seconds = und_ar_runtime.publish_und_ar_weights(
            replicas,
            actor_rollout_wg=self.actor_rollout_wg,
            receiver_ranks=receiver_ranks,
            check_names=self.UND_AR_SYNC_CHECK_PARAMS,
            global_steps=self.global_steps,
        )
        previous = getattr(self, "_und_ar_last_fingerprint", None)
        self._und_ar_last_fingerprint = fingerprint
        sync_metrics = {
            "bagel_corl/und_sync/num_tensors": float(num_tensors),
            "bagel_corl/und_sync/seconds": seconds,
            "bagel_corl/und_sync/fingerprint_delta": 0.0 if previous is None else abs(fingerprint - previous),
        }
        pending = getattr(self, "_pending_sync_metrics", None) or {}
        pending.update(sync_metrics)
        self._pending_sync_metrics = pending
        logger.info(
            "bagel_corl_sync step=%s published policy to %d UND AR replica(s): %d tensors, %.1fs, "
            "%s fingerprint verified (delta since last publish %s)",
            self.global_steps,
            len(replicas),
            num_tensors,
            seconds,
            self.UND_AR_SYNC_CHECK_PARAM,
            sync_metrics["bagel_corl/und_sync/fingerprint_delta"],
        )

    def _bagel_rm_enabled(self) -> bool:
        return bagel_rm_enabled(self.config)

    def _setup(self) -> None:
        """Parent V1 setup with the named-model ``OmniRewardLoopManager`` in place of upstream's manager."""
        if not named_reward_models_enabled(self.config):
            super()._setup()
            return
        reward_cfg = self.config.reward
        num_workers = reward_cfg.num_workers
        try:
            with open_dict(reward_cfg):
                reward_cfg.num_workers = 0
            super()._setup()
        finally:
            with open_dict(reward_cfg):
                reward_cfg.num_workers = num_workers
        self.reward_loop_manager = None
        self._ensure_reward_loop_manager()

    def _ensure_reward_loop_manager(self):
        """Create the colocated ``OmniRewardLoopManager`` (RFC §4.2) serving both reward consumers."""
        if getattr(self, "reward_loop_manager", None) is not None:
            return self.reward_loop_manager
        from verl_omni.reward_loop import OmniRewardLoopManager

        resource_pool = None
        if self._bagel_rm_enabled() and getattr(self, "resource_pool_manager", None) is not None:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        self.reward_loop_manager = OmniRewardLoopManager(config=self.config, rm_resource_pool=resource_pool)
        return self.reward_loop_manager

    def get_reward_handles(self):
        """Handles the pinned ``TaskRunnerV1`` forwards into the agent-loop manager."""
        manager = getattr(self, "reward_loop_manager", None)
        workers = getattr(manager, "reward_loop_workers", None) if manager is not None else None
        if workers:
            return workers
        getter = getattr(super(), "get_reward_handles", None)
        handles = getter() if callable(getter) else None
        return list(handles) if handles else None

    def _compute_reward_colocate(self, batch, metrics: dict | None = None):
        """Score a batch with the colocated RM, handing it the row fields the scorer needs."""
        return compute_reward_colocate(
            batch,
            reward_loop_manager=self.reward_loop_manager,
            tokenizer=self.tokenizer,
            lengths_to_mask=self._lengths_to_mask,
        )

    def on_init_end(self):
        """Build UND AR and publish step-0 weights to both pools before the parent hook."""
        self._ensure_dual_role_rollout()
        if self._bagel_rm_enabled():
            self._ensure_reward_loop_manager()
        self._bagel_corl_published_weight_version = None
        self._publish_und_ar_weights()
        super().on_init_end()
        self._wake_rollout_replicas()
        self._bagel_corl_published_weight_version = self.global_steps

    def on_step_end(self):
        """UND push first (replicas are asleep), then the parent's GEN publish, then wake both pools."""
        self._bagel_corl_published_weight_version = None
        with marked_timer("update_und_weights", self.timing_raw, color="red"):
            self._publish_und_ar_weights()
        super().on_step_end()
        self._wake_rollout_replicas()
        self._bagel_corl_published_weight_version = self.global_steps

    def on_validate_end(self):
        """Re-wake the AR replica after ``_validate``'s colocated-reward sleep."""
        super().on_validate_end()
        self._wake_rollout_replicas()

    def _validate(self):
        """V1 validation, plus the RFC §8.2 evidence record."""
        manager = getattr(self, "checkpoint_manager", None)
        publish = getattr(manager, "update_weights", None)
        if publish is None:
            val_metrics = super()._validate()
        else:

            def _publish_then_wake(*args, **kwargs):
                result = publish(*args, **kwargs)
                self._wake_rollout_replicas()
                return result

            try:
                manager.update_weights = _publish_then_wake
                val_metrics = super()._validate()
            finally:
                manager.update_weights = publish
        evidence = step_metrics.rfc82_evidence(val_metrics)
        val_metrics = dict(val_metrics or {})
        val_metrics["val/rfc82_evidence_keys"] = float(len(evidence))
        if "episode/K" in evidence and evidence.get("episode/J") == 0 and evidence.get("episode/K") == 0:
            logger.warning(
                "bagel_corl_sync §8.2: validation produced no dual-lane episode evidence "
                "(J=0, K=0); the validation batch did not run the bagel_multiturn_agent loop."
            )
        logger.info("bagel_corl_sync §8.2 validation evidence: %s", evidence or "<none>")
        return val_metrics
