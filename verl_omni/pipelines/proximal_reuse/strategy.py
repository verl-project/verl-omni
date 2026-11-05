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
from functools import wraps
from threading import Lock

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.utils.padding import response_from_nested, response_to_nested

from verl_omni.pipelines.proximal_reuse import calls as worker_calls
from verl_omni.pipelines.step_extension import register_step_extension
from verl_omni.workers.utils.training_metrics import get_training_output_metrics

logger = logging.getLogger(__name__)


def validate_deferred_rollout_inputs(data, *, require_nonempty):
    """Reject known invalid rollout diagnostics before preparing any actor.

    The old probabilities become available only inside the actor forward.
    Check the independently available rollout probabilities and mask now so
    the native helper's known input errors cannot occur after an update.
    """
    values = data.select("response_mask", "rollout_log_probs", "responses").to_padded_tensor()
    mask, rollout, responses = values["response_mask"], values["rollout_log_probs"], values["responses"]
    if mask.ndim != 2 or responses.ndim != 2:
        raise ValueError("Rollout diagnostics require two-dimensional response tensors")
    if not mask.any():
        if require_nonempty:
            raise ValueError("response_mask must contain at least one valid token (1).")
        return
    if rollout.ndim != 2 or rollout.shape != mask.shape:
        raise ValueError("rollout_log_probs and response_mask must have the same shape")
    if responses.shape != mask.shape:
        raise ValueError("responses and response_mask must have the same shape")


def validate_proximal_output_collection(worker_group, plan):
    """Check that native source ranks cover DP shards in row order before commit."""
    count = len(plan.versions)
    plan.validate(count)
    mapping = worker_group._query_dispatch_info("actor")
    mask = worker_group._query_collect_info("actor")
    if (
        not isinstance(mapping, list | tuple)
        or not isinstance(mask, list | tuple)
        or len(mapping) != count
        or len(mask) != count
        or any(type(dp) is not int or not 0 <= dp < len(plan.ranks_by_dp) for dp in mapping)
        or any(type(selected) is not bool for selected in mask)
    ):
        raise ValueError("Invalid native actor output collection")
    expected = tuple(tuple(rank for rank, dp in enumerate(mapping) if dp == i) for i in range(len(plan.ranks_by_dp)))
    selected_dp = tuple(mapping[rank] for rank, selected in enumerate(mask) if selected)
    if expected != plan.ranks_by_dp or selected_dp != tuple(range(len(plan.ranks_by_dp))):
        raise ValueError("Actor output sources do not match the prepared shard order")
    return tuple(mask)


def publish_proximal_outputs(worker_group, batch, outputs, collect_mask):
    """Publish selected model outputs while preserving native scalar metrics."""
    from verl.single_controller.base.decorator import collect_nd_compute_dataproto

    if not isinstance(outputs, list) or len(outputs) != len(collect_mask):
        raise ValueError("Expected an output from every actor commit RPC")
    selected = collect_nd_compute_dataproto(list(collect_mask), worker_group, outputs)
    if not isinstance(selected, TensorDict) or selected.batch_size != (len(batch),):
        raise ValueError("Collected actor outputs do not match the training batch")
    if any(not isinstance(value, torch.Tensor | NonTensorStack | NonTensorData) for value in selected.values()):
        raise ValueError("Unsupported actor output type")
    fields = [key for key, value in selected.items() if isinstance(value, torch.Tensor | NonTensorStack)]
    if not {"log_probs", "entropy"}.issubset(fields):
        raise ValueError("Collected actor outputs are missing log probabilities or entropy")
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=selected.select(*fields))
    return selected


def _settle_actor_refs(refs, count, phase):
    """Surface an observed failure without assuming the other actors stopped."""
    if not isinstance(refs, list) or len(refs) != count or len(set(refs)) != count:
        raise ValueError(f"Expected a distinct {phase} request for every actor")
    pending = list(refs)
    results = {}
    while pending:
        ready, remaining = ray.wait(pending, num_returns=1, fetch_local=True)
        if (
            len(ready) != 1
            or ready[0] not in pending
            or len(remaining) != len(pending) - 1
            or ready[0] in remaining
            or set(ready + remaining) != set(pending)
        ):
            raise RuntimeError(f"Invalid actor {phase} wait result")
        # An error does not prove its actor or peers stopped. The caller must
        # not clean up reservations or retry on an uncertain RPC outcome.
        results[ready[0]] = ray.get(ready[0], timeout=0)
        pending = remaining
    return [results[ref] for ref in refs]


def coordinate_staged_proximal_update(worker_group, call, plan):
    """Validate every actor before commit, without claiming distributed atomicity.

    A failed or cancelled RPC can have an unknown outcome. In that case, do
    not release reservations, retry, or issue a second update automatically.
    """
    from verl_omni.workers.utils.cohort_lease import validate_observations

    count = len(plan.versions)
    plan.validate(count)
    _settle_actor_refs(worker_calls.stage(worker_group, "update", call), count, "stage")
    observations = worker_calls.control(worker_group, "get_update")
    try:
        if (
            not isinstance(observations, list)
            or len(observations) != count
            or not all(isinstance(observation, dict) for observation in observations)
        ):
            raise ValueError("Missing or malformed staged actor observations")
        validate_observations(plan, observations)
    except (TypeError, ValueError, RuntimeError) as exc:
        discarded = worker_calls.control(worker_group, "discard_update", plan.cohort_id)
        if not isinstance(discarded, list) or len(discarded) != count or not all(x is True for x in discarded):
            raise RuntimeError("Could not discard every rejected actor update") from exc
        raise
    # This is not an atomic transaction: after sending the first commit,
    # errors or cancellation may mean some actors have already updated.
    # Propagate failure and stop the job; never release or retry on this path.
    return _settle_actor_refs(worker_calls.commit(worker_group, "update", plan.cohort_id), count, "commit")


def _fail_stop(method):
    """Fence this instance after an in-flight reuse outcome becomes unknown."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        if not self._omni_call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent actor reuse operations are not supported")
        try:
            if self._omni_failed:
                raise RuntimeError("A previous actor reuse operation failed; replace the worker group")
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                if self._omni_inflight:
                    self._omni_failed = True
                raise
        finally:
            self._omni_call_lock.release()

    return guarded


class ProximalActorReuse:
    def __init__(self, trainer):
        self.trainer = trainer
        self._omni_inflight = False
        self._omni_failed = False
        self._omni_call_lock = Lock()

    @_fail_stop
    def _compute_old_log_prob(self, batch, metrics, native):
        if getattr(self, "_omni_prepared_actor", None) is not None:
            raise RuntimeError("A previous actor cohort has not finished")
        prepared = self._select_proximal_actor(batch, metrics)
        self._omni_prepared_actor = prepared
        if prepared is None:
            return native(batch, metrics)
        return batch

    @_fail_stop
    def _compute_advantage(self, batch, metrics, native):
        prepared = getattr(self, "_omni_prepared_actor", None)
        if prepared is None:
            return native(batch, metrics)
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
            adv_estimator=self.trainer.config.algorithm.adv_estimator,
            gamma=self.trainer.config.algorithm.gamma,
            lam=self.trainer.config.algorithm.lam,
            num_repeat=self.trainer.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.trainer.config.algorithm,
        )
        output = TensorDict(
            {key: response_to_nested(data.batch[key], response_mask) for key in ("advantages", "returns")},
            batch_size=len(batch),
        )
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    @_fail_stop
    def _update_actor(self, batch, metrics, native):
        prepared = getattr(self, "_omni_prepared_actor", None)
        if prepared is None:
            return native(batch, metrics)
        # Do not clear on failure: a partly applied cohort must never retry or fall back.
        result = self._update_proximal_actor(batch, metrics, prepared)
        self._omni_prepared_actor = None
        self._omni_inflight = False
        return result

    def _select_proximal_actor(self, batch, metrics):
        """Opt in only before model work; unsupported execution keeps native inference."""
        enabled = _enabled(self.trainer.config)
        if type(enabled) is not bool:
            raise ValueError("reuse_proximal_actor_forward must be a boolean")
        if not enabled:
            return None
        metrics["training/proximal_anchor_reuse"] = 0
        if self.trainer.config.trainer.critic_warmup > self.trainer.global_steps:
            return None
        try:
            self._check_proximal_algorithm(batch)
        except ValueError as exc:
            logger.info("Using independent old-policy inference: %s", exc)
            return None
        reports = worker_calls.control(self.trainer.actor_rollout_wg, "qualify")
        mapping = self.trainer.actor_rollout_wg._query_dispatch_info("actor")
        if not isinstance(reports, list) or len(reports) != len(mapping):
            raise ValueError("Missing actual actor-rank eligibility reports")
        for report in reports:
            if not isinstance(report, dict) or type(report.get("supported")) is not bool:
                raise ValueError("Malformed actor-rank eligibility report")
        if not all(report["supported"] for report in reports):
            logger.info("Using independent old-policy inference: %s", reports)
            return None
        # Conservatively limit the opt-in to the previously qualified layout.
        # These read-only rank checks do not establish group identity; staged
        # coordination does not invoke a collective on WORLD.
        if any(
            type(report.get("world_size")) is not int
            or report["world_size"] != len(mapping)
            or type(report.get("world_rank")) is not int
            or report["world_rank"] != rank
            for rank, report in enumerate(reports)
        ):
            logger.info("Using independent old-policy inference: actor and WORLD ranks differ")
            return None
        # Never catch errors from preparation or a partially started update.
        # Workers independently recheck execution while binding actual inputs.
        prepared = self._prepare_proximal_actor(batch)
        if prepared is None:
            return None
        metrics["training/proximal_anchor_reuse"] = 1
        return prepared

    def _check_proximal_algorithm(self, batch):
        """Reject old-dependent math before any actor reservation or update."""
        from verl_omni.pipelines.proximal_reuse.rpc import validate_actor_batch_controls

        controls = getattr(batch, "extra_info", {})
        if not isinstance(controls, dict):
            raise ValueError("Unqualified actor batch controls")
        validate_actor_batch_controls(controls.get)
        algorithm = self.trainer.config.algorithm
        actor = self.trainer.config.actor_rollout_ref.actor
        correction = algorithm.get("rollout_correction", None) or {}
        hybrid = self.trainer.config.trainer.v1.get("separate_async", {}).get("hybrid_rollout", {})
        if hybrid.get("enable_switch", False):
            raise ValueError("Hybrid rollout switching requires independent qualification")
        if (
            self.trainer.parameter_sync_step != 1
            or self.trainer.use_critic
            or self.trainer.use_teacher_policy
            or algorithm.adv_estimator != "grpo"
            or algorithm.use_kl_in_reward
        ):
            raise ValueError("Unqualified proximal advantage/reward or parameter-sync policy")
        if (
            correction.get("rollout_is") is not None
            or correction.get("rollout_rs") is not None
            or correction.get("bypass_mode", False)
        ):
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
        if len(batch) != actor.ppo_mini_batch_size * self.trainer.config.actor_rollout_ref.rollout.n:
            raise ValueError("Proximal cohort differs from the full optimizer batch")

    def _prepare_proximal_actor(self, batch: KVBatchMeta):
        """Prepare native actor shards after the caller has established eligibility.

        The ordinary loop selects this only through the explicit opt-in and
        actual worker capability query. Worker receipts recheck execution.
        """
        import uuid

        from verl_omni.pipelines.proximal_reuse.rpc import (
            REQUEST,
            PrepareRequest,
            UnqualifiedActorBatchControl,
            actor_execution,
            validate_materialized_actor_batch_controls,
            validate_preparation_observations,
        )
        from verl_omni.workers.utils.cohort_data import CohortData

        self._check_proximal_algorithm(batch)
        execution = actor_execution(self.trainer.config.actor_rollout_ref)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        if REQUEST in batch.extra_info or REQUEST in data:
            raise ValueError("Caller-supplied actor reuse request is forbidden")
        try:
            validate_materialized_actor_batch_controls(data)
        except UnqualifiedActorBatchControl as exc:
            logger.info("Using independent old-policy inference: %s", exc)
            return None
        correction = getattr(self.trainer.config, "algorithm", {}).get("rollout_correction", None)
        if self.trainer.config.actor_rollout_ref.rollout.calculate_log_probs:
            validate_deferred_rollout_inputs(data, require_nonempty=correction is not None)
        binding = CohortData.prepare(data, batch.keys, execution)
        mapping = tuple(self.trainer.actor_rollout_wg._query_dispatch_info("actor"))
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
        # Mesh dispatch reaches every worker but collects only selected model
        # ranks. Fetch a second control-plane result from every actual actor.
        # Once dispatch begins, an exception or cancellation cannot establish
        # that all peers stopped or that no rank changed local state.
        self._omni_inflight = True
        _settle_actor_refs(
            worker_calls.stage(self.trainer.actor_rollout_wg, "prepare", call), len(mapping), "preparation"
        )
        observations = worker_calls.control(self.trainer.actor_rollout_wg, "get_preparation")
        try:
            prepared = validate_preparation_observations(request, observations)
        except (TypeError, ValueError) as exc:
            discarded = worker_calls.control(self.trainer.actor_rollout_wg, "discard_preparation", request.cohort_id)
            if (
                not isinstance(discarded, list)
                or len(discarded) != len(mapping)
                or not all(x is True for x in discarded)
            ):
                raise RuntimeError("Could not discard every rejected actor preparation") from exc
            raise
        installed = worker_calls.control(self.trainer.actor_rollout_wg, "install", prepared)
        if (
            not isinstance(installed, list)
            or len(installed) != len(mapping)
            or not all(item is True for item in installed)
        ):
            raise ValueError("Failed to install the prepared cohort on every actor")
        return prepared

    def _update_proximal_actor(self, batch: KVBatchMeta, metrics: dict, prepared) -> KVBatchMeta:
        """Bind actual advantages and run one checked staged actor update."""
        from verl_omni.pipelines.proximal_reuse.rpc import REQUEST, actor_execution

        self._check_proximal_algorithm(batch)
        execution = actor_execution(self.trainer.config.actor_rollout_ref)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        if REQUEST in batch.extra_info or REQUEST in data:
            raise ValueError("Caller-supplied actor reuse request is forbidden")
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
        # Check output collection before any actor reserves or updates. Once
        # commits are dispatched, any failure can leave a partial update;
        # callers must stop rather than retry or fall back to another path.
        collection = validate_proximal_output_collection(self.trainer.actor_rollout_wg, request.plan)
        outputs = coordinate_staged_proximal_update(self.trainer.actor_rollout_wg, call, request.plan)
        output = publish_proximal_outputs(self.trainer.actor_rollout_wg, batch, outputs, collection)
        actor_metrics = rename_dict(get_training_output_metrics(output, batch), "actor/")
        actor_metrics["perf/mfu/actor"] = actor_metrics.pop("actor/mfu")
        metrics.update(reduce_metrics(actor_metrics))
        result = self._record_old_log_prob_output(batch, metrics)
        correction = self.trainer.config.algorithm.get("rollout_correction", None)
        if correction is not None and self.trainer.config.actor_rollout_ref.rollout.calculate_log_probs:
            if (
                correction.get("rollout_is") is not None
                or correction.get("rollout_rs") is not None
                or correction.get("bypass_mode", False)
            ):
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
        if self.trainer.config.actor_rollout_ref.rollout.calculate_log_probs:
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
        actor_config = self.trainer.config.actor_rollout_ref.actor
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
        if self.trainer.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch


def _enabled(config):
    settings = getattr(config.trainer, "v1", {})
    legacy = settings.get("reuse_proximal_actor_forward", False)
    if type(legacy) is not bool:
        raise ValueError("reuse_proximal_actor_forward must be a boolean")
    return legacy or settings.get("step_extension") == "proximal_actor_reuse"


register_step_extension(
    "proximal_actor_reuse", _enabled, ProximalActorReuse, trainer_modes=frozenset({"omni_separate_async"})
)
