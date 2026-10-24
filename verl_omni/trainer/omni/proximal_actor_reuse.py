# Copyright 2026 verl-omni contributors
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

"""Optional single-update actor reuse for the Omni separate-async trainer."""

import logging

import numpy as np
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils import tensordict_utils as tu
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.utils.padding import response_from_nested, response_to_nested

from verl_omni.workers.utils.training_metrics import get_training_output_metrics

logger = logging.getLogger(__name__)


class ProximalActorReuseMixin:
    def _compute_old_log_prob(self, batch, metrics):
        if getattr(self, "_omni_prepared_actor", None) is not None:
            raise RuntimeError("A previous actor cohort has not finished")
        prepared = self._select_proximal_actor(batch, metrics)
        self._omni_prepared_actor = prepared
        if prepared is None:
            return super()._compute_old_log_prob(batch, metrics)
        return batch

    def _compute_advantage(self, batch, metrics):
        prepared = getattr(self, "_omni_prepared_actor", None)
        if prepared is None:
            return super()._compute_advantage(batch, metrics)
        self._check_proximal_algorithm(batch)
        if prepared.request.partition_id != batch.partition_id or prepared.request.binding.keys != tuple(batch.keys):
            raise ValueError("Advantage cohort changed after actor preparation")
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["uid", "response_mask", "rm_scores"]
        )
        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.batch["token_level_rewards"] = data.batch["token_level_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
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
        output = TensorDict(
            {key: response_to_nested(data.batch[key], response_mask) for key in ("advantages", "returns")},
            batch_size=len(batch),
        )
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    def _update_actor(self, batch, metrics):
        prepared = getattr(self, "_omni_prepared_actor", None)
        if prepared is None:
            return super()._update_actor(batch, metrics)
        # Do not clear on failure: a partly applied cohort must never retry or fall back.
        result = self._update_proximal_actor(batch, metrics, prepared)
        self._omni_prepared_actor = None
        return result

    def _select_proximal_actor(self, batch, metrics):
        """Opt in only before model work; unsupported execution keeps native inference."""
        enabled = getattr(self.config.trainer, "v1", {}).get("reuse_proximal_actor_forward", False)
        if type(enabled) is not bool:
            raise ValueError("reuse_proximal_actor_forward must be a boolean")
        if not enabled:
            return None
        metrics["training/proximal_anchor_reuse"] = 0
        if self.config.trainer.critic_warmup > self.global_steps:
            return None
        try:
            self._check_proximal_algorithm(batch)
        except ValueError as exc:
            logger.info("Using independent old-policy inference: %s", exc)
            return None
        reports = self.actor_rollout_wg.qualify_proximal_actor()
        mapping = self.actor_rollout_wg._query_dispatch_info("actor")
        if not isinstance(reports, list) or len(reports) != len(mapping):
            raise ValueError("Missing actual actor-rank eligibility reports")
        for report in reports:
            if not isinstance(report, dict) or type(report.get("supported")) is not bool:
                raise ValueError("Malformed actor-rank eligibility report")
        if not all(report["supported"] for report in reports):
            logger.info("Using independent old-policy inference: %s", reports)
            return None
        # Never catch errors from preparation or a partially started update.
        # Workers independently recheck execution while binding actual inputs.
        prepared = self._prepare_proximal_actor(batch)
        metrics["training/proximal_anchor_reuse"] = 1
        return prepared

    def _check_proximal_algorithm(self, batch):
        """Reject old-dependent math before any actor reservation or update."""
        algorithm = self.config.algorithm
        actor = self.config.actor_rollout_ref.actor
        correction = algorithm.get("rollout_correction", None) or {}
        hybrid = self.config.trainer.v1.get("separate_async", {}).get("hybrid_rollout", {})
        if hybrid.get("enable_switch", False):
            raise ValueError("Hybrid rollout switching requires independent qualification")
        if (
            self.parameter_sync_step != 1
            or self.use_critic
            or self.use_teacher_policy
            or algorithm.adv_estimator != "grpo"
            or algorithm.use_kl_in_reward
        ):
            raise ValueError("Unqualified proximal advantage/reward or parameter-sync policy")
        if correction.get("rollout_is") or correction.get("rollout_rs") or correction.get("bypass_mode", False):
            raise ValueError("Old-dependent rollout correction requires the ordinary path")
        if (
            actor.ppo_epochs != 1
            or actor.shuffle
            or actor.use_prefix_grouper
            or actor.calculate_sum_pi_squared
            or (actor.policy_loss.get("rollout_correction", None) or {}).get("score_centering", False)
            or actor.policy_loss.get("loss_mode", "vanilla") not in {"vanilla", "gspo"}
        ):
            raise ValueError("Proximal reuse requires one qualified full optimizer update")
        if len(batch) != actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n:
            raise ValueError("Proximal cohort differs from the full optimizer batch")

    def _prepare_proximal_actor(self, batch: KVBatchMeta):
        """Prepare native actor shards after the caller has established eligibility.

        The ordinary loop selects this only through the explicit opt-in and
        actual worker capability query. Worker receipts recheck execution.
        """
        import uuid

        from verl_omni.workers.utils.cohort_data import CohortData
        from verl_omni.workers.utils.proximal_rpc import REQUEST, PrepareRequest, actor_execution

        self._check_proximal_algorithm(batch)
        execution = actor_execution(self.config.actor_rollout_ref)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        binding = CohortData.prepare(data, batch.keys, execution)
        mapping = tuple(self.actor_rollout_wg._query_dispatch_info("actor"))
        request = PrepareRequest(uuid.uuid4().hex, batch.partition_id, binding, mapping)
        # Fetch every row field in both RPCs; an old select_fields view must
        # not hide changed media or advantages from the actual worker check.
        call = KVBatchMeta(
            keys=batch.keys,
            tags=batch.tags,
            partition_id=batch.partition_id,
            fields=None,
            extra_info={**batch.extra_info, REQUEST: request},
        )
        output = self.actor_rollout_wg.prepare_proximal_actor(call)
        prepared = tu.get_non_tensor_data(output, "proximal_preparation", default=None)
        if prepared.request != request:
            raise ValueError("Actor preparation returned a different cohort")
        return prepared

    def _update_proximal_actor(self, batch: KVBatchMeta, metrics: dict, prepared) -> KVBatchMeta:
        """Bind actual advantages and run the checked native atomic update once."""
        from verl_omni.workers.utils.proximal_rpc import REQUEST, actor_execution

        self._check_proximal_algorithm(batch)
        execution = actor_execution(self.config.actor_rollout_ref)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        request = prepared.bind(data, batch.keys, execution)
        controls = {
            key: execution[key]
            for key in (
                "temperature",
                "global_batch_size",
                "mini_batch_size",
                "epochs",
                "seed",
                "dataloader_kwargs",
                "calculate_entropy",
                "return_model_output",
            )
        }
        call = KVBatchMeta(
            keys=batch.keys,
            tags=batch.tags,
            partition_id=batch.partition_id,
            fields=None,
            extra_info={**batch.extra_info, **controls, REQUEST: request},
        )
        # A failure after any rank enters the update cannot fall back and
        # perform a second optimizer step. Let the original failure propagate.
        output = self.actor_rollout_wg.update_proximal_actor(call)
        actor_metrics = rename_dict(get_training_output_metrics(output, batch), "actor/")
        actor_metrics["perf/mfu/actor"] = actor_metrics.pop("actor/mfu")
        metrics.update(reduce_metrics(actor_metrics))
        result = self._record_old_log_prob_output(batch, metrics)
        correction = self.config.algorithm.get("rollout_correction", None)
        if correction is not None and self.config.actor_rollout_ref.rollout.calculate_log_probs:
            if correction.get("rollout_is") or correction.get("rollout_rs") or correction.get("bypass_mode", False):
                raise ValueError("Old-dependent rollout correction was not rejected before preparation")
            actual = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["old_log_probs", "rollout_log_probs", "response_mask"],
            )
            _, diagnostics = compute_rollout_correction_and_add_to_batch(
                DataProto(batch=actual.to_padded_tensor()), correction
            )
            metrics.update(diagnostics)
        return result

    def _record_old_log_prob_output(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Consume full-cohort pre-update outputs without another model call."""
        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch
