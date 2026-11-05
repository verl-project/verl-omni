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

"""Native-dispatch preparation and staged update for a qualified cohort.

These RPCs do not establish algorithm/model eligibility. Callers must qualify
that separately before preparation; normal compute_log_prob/update_actor are
unchanged. Each receipt uses the actual TQ shard and actual engine config.
An error after commit begins may leave some actors updated; callers must fail
the training run rather than retrying the update.
"""

from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from types import SimpleNamespace

import torch.distributed as dist
from omegaconf import OmegaConf
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.actor import PolicyLossConfig

from verl_omni.pipelines.proximal_reuse.anchor import proximal_worker_loss_scope
from verl_omni.workers.utils.cohort_data import CohortData, freeze_cohort_data, value_digest
from verl_omni.workers.utils.cohort_lease import CohortPlan
from verl_omni.workers.utils.parameter_revision import ParameterVersion

REQUEST = "_proximal_cohort_request"
TRANSPORT = "_step_transport_identity"


class UnqualifiedActorBatchControl(ValueError):
    """A known batch control requires the ordinary actor path."""


def validate_actor_batch_controls(get_value):
    """Reject per-batch forward controls outside the qualified execution path."""
    for key in (
        "enable_routing_replay",
        "distillation_use_topk",
        "distillation_only",
        "calculate_sum_pi_squared",
        "score_centering",
        "no_lora_adapter",
    ):
        if get_value(key, False) is not False:
            raise UnqualifiedActorBatchControl(f"Unqualified actor batch control: {key}")
    for key in ("forced_max_seqlen", "local_cp_size"):
        if get_value(key, None) is not None:
            raise UnqualifiedActorBatchControl(f"Unqualified actor batch control: {key}")
    group_size = get_value("force_group_size", 1)
    if type(group_size) is not int or group_size != 1:
        raise UnqualifiedActorBatchControl("Unqualified actor batch control: force_group_size")
    if get_value("pad_mode", DatasetPadMode.NO_PADDING) != DatasetPadMode.NO_PADDING:
        raise UnqualifiedActorBatchControl("Unqualified actor batch control: pad_mode")


def validate_materialized_actor_batch_controls(data):
    validate_actor_batch_controls(lambda key, default: tu.get_non_tensor_data(data, key, default=default))


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError("Require an actual resolved configuration")


def actor_execution(config):
    """Independently read the native training controls on coordinator and worker."""
    actor, rollout = config.actor, config.rollout
    return {
        "temperature": rollout.temperature,
        "global_batch_size": actor.ppo_mini_batch_size * rollout.n,
        "mini_batch_size": actor.ppo_mini_batch_size * rollout.n,
        "epochs": actor.ppo_epochs,
        "seed": actor.data_loader_seed,
        "dataloader_kwargs": {"shuffle": actor.shuffle},
        "calculate_entropy": True,  # Preserve independent-old diagnostics.
        "return_model_output": True,
        "policy_loss": _policy_loss(actor.policy_loss),
        "entropy_coeff": actor.entropy_coeff,
        "actor_calculate_entropy": actor.calculate_entropy,
        "use_kl_loss": actor.use_kl_loss,
        "kl_loss_coef": actor.kl_loss_coef,
        "kl_loss_type": actor.kl_loss_type,
        "loss_agg_mode": actor.loss_agg_mode,
        "loss_scale_factor": actor.loss_scale_factor,
        "clip_ratio": actor.clip_ratio,
        "clip_ratio_low": actor.clip_ratio_low,
        "clip_ratio_high": actor.clip_ratio_high,
        "clip_ratio_c": actor.clip_ratio_c,
    }


def _policy_loss(config):
    # The coordinator holds YAML while the actual native loss holds a Hydra
    # dataclass with populated defaults. Compare their effective settings,
    # excluding construction metadata, without dropping unknown math fields.
    value = omega_conf_to_dataclass(_plain(config), dataclass_type=PolicyLossConfig)

    def without_target(value):
        if isinstance(value, dict):
            return {key: without_target(item) for key, item in value.items() if key != "_target_"}
        return value

    return without_target(asdict(value))


def _engine_digest(engine):
    # Include the actual worker-resolved engine settings, not merely the
    # coordinator's launch configuration. Unknown objects fail before forward.
    from verl_omni.pipelines.proximal_reuse.execution import megatron_proximal_execution
    from verl_omni.workers.engine.megatron.omni_impl import OmniMegatronEngine

    if type(engine) is not OmniMegatronEngine:
        raise ValueError("Engine does not support qualified actor reuse")
    return value_digest({"engine": _plain(engine.engine_config), "proximal": megatron_proximal_execution(engine)})


def _check_actual_loss(worker):
    from verl.workers.utils.losses import ppo_loss

    loss = worker.actor.loss_fn
    if not isinstance(loss, partial) or loss.func is not ppo_loss or loss.args or set(loss.keywords) != {"config"}:
        raise ValueError("Custom, score-centered or distilled loss requires independent old inference")
    actual = actor_execution(SimpleNamespace(actor=loss.keywords["config"], rollout=worker.config.rollout))
    if actual != actor_execution(worker.config):
        raise ValueError("Actual loss configuration differs from the prepared actor settings")


@dataclass(frozen=True)
class PrepareRequest:
    cohort_id: str
    partition_id: str
    binding: CohortData
    dp_mapping: tuple[int, ...]


@dataclass(frozen=True)
class RankReceipt:
    rank: int
    dp_rank: int
    version: ParameterVersion
    binding: CohortData
    engine_digest: str


@dataclass(frozen=True)
class PreparedCohort:
    request: PrepareRequest
    ranks: tuple[RankReceipt, ...]

    def bind(self, data, keys, execution):
        added = tuple(sorted(set(data.keys()) - set(self.request.binding.raw_fields)))
        # Coordinator storage contains row fields only, no execution metadata.
        bound = self.request.binding.bind_derived(data, keys, execution, added)
        lookup = {key: i for i, key in enumerate(bound.keys)}
        dp_count = max(self.request.dp_mapping) + 1
        shards, ranks_by_dp = [], []
        for dp in range(dp_count):
            replicas = tuple(receipt for receipt in self.ranks if receipt.dp_rank == dp)
            indices = [lookup[key] for key in replicas[0].binding.keys]
            local = bound.shard(indices)
            # Raw rows are independently established by actual dispatch, not
            # guessed contiguous offsets or a whole-batch hash on every rank.
            if local.raw_rows != replicas[0].binding.raw_rows:
                raise ValueError("Prepared raw rows changed before update")
            shards.append(local.identity())
            ranks_by_dp.append(tuple(receipt.rank for receipt in replicas))
        plan = CohortPlan(
            self.request.cohort_id,
            self.request.partition_id,
            tuple(ranks_by_dp),
            tuple(shards),
            tuple(r.version for r in self.ranks),
        )
        plan.validate(len(self.ranks))
        return UpdateRequest(self, bound, plan)


@dataclass(frozen=True)
class UpdateRequest:
    prepared: PreparedCohort
    binding: CohortData
    plan: CohortPlan


def _actual_transport(data):
    identity = tu.get_non_tensor_data(data, TRANSPORT, default=None)
    keys, partition = identity["keys"], identity["partition_id"]
    if len(keys) != len(data) or not partition:
        raise ValueError("Invalid native transport identity")
    return tuple(keys), partition


def _select(binding, keys):
    lookup = {key: i for i, key in enumerate(binding.keys)}
    return binding.shard([lookup[key] for key in keys])


def qualify_actor(worker):
    """Read-only capability query before reserving any cohort or model work."""
    try:
        engine = worker.actor.engine
        _check_actual_loss(worker)
        digest = _engine_digest(engine)
        if worker.enable_routing_replay:
            raise ValueError("Routing replay requires independent old inference")
        if not dist.is_initialized():
            raise ValueError("Actor process group is not initialized")
        return {
            "supported": True,
            "execution_digest": digest,
            "world_size": dist.get_world_size(),
            "world_rank": dist.get_rank(),
        }
    except (ValueError, TypeError, AttributeError) as exc:
        return {"supported": False, "reason": str(exc)}


def validate_preparation_observations(request, observations):
    """Validate every actor's preparation before binding a cohort.

    The caller is responsible for obtaining one settled observation per
    actual actor, in native dispatch order. This function neither launches
    RPCs nor proves the process-group membership of those actors.
    """
    if not isinstance(observations, list | tuple):
        raise TypeError("Malformed actor preparation observations")
    if any(not isinstance(item, dict) for item in observations):
        raise TypeError("Malformed actor preparation observation")
    if any(item.get("error") for item in observations):
        raise ValueError(f"Actor preparation rejected before forward: {observations}")
    if not isinstance(request, PrepareRequest) or not isinstance(request.binding, CohortData):
        raise TypeError("Expected a coordinator preparation request")
    if (
        not isinstance(request.cohort_id, str)
        or not request.cohort_id
        or not isinstance(request.partition_id, str)
        or not request.partition_id
        or request.binding.derived_fields
        or not request.dp_mapping
        or any(type(dp) is not int or dp < 0 for dp in request.dp_mapping)
        or sorted(set(request.dp_mapping)) != list(range(len(set(request.dp_mapping))))
    ):
        raise ValueError("Invalid preparation request or DP mapping")
    size = len(request.dp_mapping)
    if len(observations) != size:
        raise ValueError("Expected one preparation observation for every actor")
    if any(set(item) != {"request", "receipt"} or item["request"] != request for item in observations):
        raise ValueError("Ranks received different or malformed coordinator requests")
    receipts = tuple(item["receipt"] for item in observations)
    if any(not isinstance(receipt, RankReceipt) for receipt in receipts):
        raise TypeError("Malformed actor preparation receipt")
    if any(
        type(r.rank) is not int or type(r.dp_rank) is not int or r.rank != i or r.dp_rank != request.dp_mapping[i]
        for i, r in enumerate(receipts)
    ):
        raise ValueError("Actual training ranks differ from native dispatch")
    if any(
        not isinstance(r.version, ParameterVersion)
        or not isinstance(r.version.incarnation, str)
        or not r.version.incarnation
        or type(r.version.revision) is not int
        or r.version.revision < 0
        for r in receipts
    ):
        raise ValueError("Invalid actor version")
    if len({r.version.incarnation for r in receipts}) != size or len({r.version.revision for r in receipts}) != 1:
        raise ValueError("Training engine identities or writer histories differ")
    all_keys: list[str] = []
    for dp in sorted(set(request.dp_mapping)):
        replicas = [r for r in receipts if r.dp_rank == dp]
        if not isinstance(replicas[0].binding, CohortData):
            raise TypeError("Malformed actor cohort binding")
        if any(r.binding != replicas[0].binding or r.engine_digest != replicas[0].engine_digest for r in replicas):
            raise ValueError("Model-parallel replicas received different data or execution settings")
        try:
            expected = _select(request.binding, replicas[0].binding.keys)
        except (KeyError, ValueError) as exc:
            raise ValueError("Actor shard does not belong to the requested cohort") from exc
        if replicas[0].binding != expected:
            raise ValueError("Actor shard differs from the requested cohort")
        all_keys.extend(replicas[0].binding.keys)
    if len(all_keys) != len(set(all_keys)) or set(all_keys) != set(request.binding.keys):
        raise ValueError("Native DP dispatch lost or duplicated rows")
    return PreparedCohort(request, receipts)


def observe_actor_preparation(worker, data, *, rank, size):
    """Build one actor's observation without entering a distributed collective."""
    engine = worker.actor.engine
    try:
        if type(rank) is not int or type(size) is not int or size <= 0 or not 0 <= rank < size:
            raise ValueError("Invalid native actor rank or worker count")
        request = tu.get_non_tensor_data(data, REQUEST, default=None)
        if not isinstance(request, PrepareRequest):
            raise TypeError("Expected coordinator prepare request")
        if len(request.dp_mapping) != size or sorted(set(request.dp_mapping)) != list(
            range(engine.get_data_parallel_size())
        ):
            raise ValueError("Actor process group differs from native dispatch")
        dp_rank = engine.get_data_parallel_rank()
        if request.dp_mapping[rank] != dp_rank:
            raise ValueError("Actual rank and native dispatch mapping differ")
        if not request.cohort_id or request.binding.derived_fields:
            raise ValueError("Invalid fresh cohort request")
        validate_materialized_actor_batch_controls(data)
        keys, partition = _actual_transport(data)
        if partition != request.partition_id:
            raise ValueError("Actual transport partition changed")
        execution = actor_execution(worker.config)
        cfg = engine.engine_config
        if cfg.use_dynamic_bsz or cfg.micro_batch_size_per_gpu != cfg.infer_micro_batch_size_per_gpu:
            raise ValueError("Unqualified old/actor microbatch partition")
        if cfg.micro_batch_size_per_gpu is None or cfg.micro_batch_size_per_gpu <= 0:
            raise ValueError("Missing actual engine microbatch size")
        if execution["epochs"] != 1 or execution["dataloader_kwargs"]["shuffle"]:
            raise ValueError("Require one unshuffled optimizer batch")
        if len(keys) * engine.get_data_parallel_size() != execution["mini_batch_size"]:
            raise ValueError("Native DP shard is not one complete optimizer batch")
        if len(keys) % cfg.micro_batch_size_per_gpu:
            raise ValueError("Partial actor microbatch is not qualified")
        if any(field in data for field in ("old_log_probs", "rollout_is_weights", "advantages", "returns")):
            raise ValueError("Existing old-policy fields cannot be prepared for reuse")
        expected = _select(request.binding, keys)
        _check_actual_loss(worker)
        tracker = engine.enable_parameter_tracking()
        version = tracker.version()
        actual = CohortData.prepare(data, keys, execution)
        if actual != expected:
            raise ValueError("Actual worker inputs differ from coordinator inputs")
        engine_digest = _engine_digest(engine)
        if tracker.version() != version:
            raise ValueError("Parameters changed while preparing inputs")
        receipt = RankReceipt(rank, dp_rank, version, actual, engine_digest)
        observation = {"request": request, "receipt": receipt}
    except Exception as exc:
        observation = {"error": f"{type(exc).__name__}: {exc}"}
    return observation


def prepare_actor_locally(worker, data):
    """Stage a native-dispatched shard for all-worker coordinator validation."""
    if getattr(worker, "_prepared_proximal_cohort", None) is not None:
        raise RuntimeError("A previous actor cohort has not finished")
    if getattr(worker, "_local_proximal_preparation", None) is not None:
        raise RuntimeError("A previous actor preparation is still staged")
    request = tu.get_non_tensor_data(data, REQUEST, default=None)
    cohort_id = request.cohort_id if isinstance(request, PrepareRequest) else None
    observation = observe_actor_preparation(worker, data, rank=worker.rank, size=worker.world_size)
    worker._local_proximal_preparation = (cohort_id, observation)
    return tu.get_tensordict({}, non_tensor_dict={"proximal_local_staged": True})


def get_actor_preparation(worker):
    """Return this actor's last local observation to the coordinator."""
    staged = getattr(worker, "_local_proximal_preparation", None)
    if not isinstance(staged, tuple) or len(staged) != 2 or not isinstance(staged[1], dict):
        raise RuntimeError("Actor has no staged preparation")
    return staged[1]


def discard_actor_preparation(worker, cohort_id):
    """Discard a matching pre-commit observation after every stage has settled."""
    staged = getattr(worker, "_local_proximal_preparation", None)
    if staged is None:
        return True
    if not isinstance(staged, tuple) or len(staged) != 2 or staged[0] != cohort_id:
        raise ValueError("Staged actor cohort differs from the requested discard")
    if getattr(worker, "_prepared_proximal_cohort", None) is not None:
        raise RuntimeError("Cannot discard an installed actor cohort")
    worker._local_proximal_preparation = None
    return True


def install_actor_preparation(worker, prepared):
    """Install a coordinator-validated cohort only on the matching actor."""
    if not isinstance(prepared, PreparedCohort):
        raise TypeError("Expected a validated preparation")
    observation = get_actor_preparation(worker)
    rank = worker.rank
    if type(rank) is not int or not 0 <= rank < len(prepared.ranks):
        raise ValueError("Invalid native actor rank")
    if observation != {"request": prepared.request, "receipt": prepared.ranks[rank]}:
        raise ValueError("Validated preparation differs from the actor observation")
    tracker = worker.actor.engine.enable_parameter_tracking()
    with tracker.preflight_update(prepared.ranks[rank].version, prepared.request.cohort_id):
        worker._prepared_proximal_cohort = prepared
        worker._local_proximal_preparation = None
    return True


def observe_actor_update(worker, data, request, prepared, rank):
    """Verify the actual worker shard and controls before actor model work."""
    validate_materialized_actor_batch_controls(data)
    engine = worker.actor.engine
    _check_actual_loss(worker)
    receipt = prepared.ranks[rank]
    if _engine_digest(engine) != receipt.engine_digest:
        raise ValueError("Resolved engine execution changed after preparation")
    keys, partition = _actual_transport(data)
    if partition != prepared.request.partition_id or keys != receipt.binding.keys:
        raise ValueError("Native update shard changed")
    if not {"advantages", "returns"}.issubset(data.keys()) or any(
        field in data for field in ("old_log_probs", "rollout_is_weights")
    ):
        raise ValueError("Missing derived data or unqualified old-dependent fields")
    binding = _select(request.binding, keys)
    if binding.raw_fields != receipt.binding.raw_fields or binding.raw_rows != receipt.binding.raw_rows:
        raise ValueError("Raw preparation was replaced at update")
    execution = actor_execution(worker.config)
    # The actual native train wrapper consumes these metadata values.
    for key in (
        "temperature",
        "global_batch_size",
        "mini_batch_size",
        "epochs",
        "seed",
        "dataloader_kwargs",
        "calculate_entropy",
        "return_model_output",
    ):
        if tu.get_non_tensor_data(data, key, default=None) != execution[key]:
            raise ValueError(f"Actual training metadata changed: {key}")
    for key in ("use_dynamic_bsz", "max_token_len_per_gpu", "micro_batch_size_per_gpu", "use_fused_kernels"):
        if key in data and tu.get_non_tensor_data(data, key, default=None) != getattr(engine.engine_config, key):
            raise ValueError(f"Actual engine metadata overrides prepared configuration: {key}")
    return binding.observe(data, keys, execution)


@dataclass
class _StagedActorUpdate:
    cohort_id: str
    observation: dict
    request: UpdateRequest | None = None
    data: object = None


def stage_actor_update(worker, data):
    """Validate and reserve one frozen shard before coordinator approval."""
    if getattr(worker, "_staged_proximal_update", None) is not None:
        raise RuntimeError("A previous actor update is still staged")
    prepared = getattr(worker, "_prepared_proximal_cohort", None)
    if not isinstance(prepared, PreparedCohort):
        raise RuntimeError("No successfully prepared actor cohort")
    cohort_id = prepared.request.cohort_id
    try:
        request = tu.get_non_tensor_data(data, REQUEST, default=None)
        if not isinstance(request, UpdateRequest) or request.prepared != prepared:
            raise ValueError("Update does not match the locally prepared cohort")
        rank, size = worker.rank, worker.world_size
        if type(rank) is not int or type(size) is not int or size <= 0 or not 0 <= rank < size:
            raise ValueError("Invalid native actor rank or worker count")
        plan = request.plan
        plan.validate(size)
        if (
            plan.versions != tuple(r.version for r in prepared.ranks)
            or plan.cohort_id != cohort_id
            or plan.partition_id != prepared.request.partition_id
            or plan.ranks_by_dp
            != tuple(
                tuple(i for i, mapped_dp in enumerate(prepared.request.dp_mapping) if mapped_dp == dp)
                for dp in range(len(plan.ranks_by_dp))
            )
        ):
            raise ValueError("Update attempted to replace the prepared versions or topology")
        frozen = freeze_cohort_data(data)
        identity = observe_actor_update(worker, frozen, request, prepared, rank)
        dp_rank = worker.actor.engine.get_data_parallel_rank()
        if type(dp_rank) is not int or not 0 <= dp_rank < len(plan.shards) or plan.shards[dp_rank] != identity:
            raise ValueError("Actual actor update shard differs from the approved plan")
        version = plan.versions[rank]
        observation = {
            "cohort_id": cohort_id,
            "partition_id": plan.partition_id,
            "rank": rank,
            "dp_rank": dp_rank,
            "version": version,
            "shard": identity,
        }
        staged = _StagedActorUpdate(cohort_id, observation, request, frozen)
        tracker = worker.actor.engine.enable_parameter_tracking()
        tracker.reserve_update(version, cohort_id)
        try:
            worker._staged_proximal_update = staged
        except BaseException:
            tracker.abort_reserved_update(version, cohort_id)
            raise
    except Exception as exc:
        worker._staged_proximal_update = _StagedActorUpdate(cohort_id, {"error": f"{type(exc).__name__}: {exc}"})
    return tu.get_tensordict({}, non_tensor_dict={"proximal_update_staged": True})


def get_staged_actor_update(worker):
    staged = getattr(worker, "_staged_proximal_update", None)
    if not isinstance(staged, _StagedActorUpdate):
        raise RuntimeError("Actor has no staged update")
    return staged.observation


def discard_staged_actor_update(worker, cohort_id):
    """Abort only matching pre-commit reservations after all staging has settled."""
    staged = getattr(worker, "_staged_proximal_update", None)
    if staged is None:
        return True
    if not isinstance(staged, _StagedActorUpdate) or staged.cohort_id != cohort_id:
        raise ValueError("Staged actor update differs from the requested discard")
    if "error" not in staged.observation:
        tracker = worker.actor.engine.enable_parameter_tracking()
        tracker.abort_reserved_update(staged.observation["version"], cohort_id)
    worker._staged_proximal_update = None
    return True


def commit_staged_actor_update(worker, cohort_id):
    """Consume a coordinator-approved reservation once, using its frozen shard.

    Once commit starts, a failed optimizer call can have partially changed
    model state. The revision guard poisons that worker; callers must stop the
    whole training job rather than retry or fall back to another update.
    """
    staged = getattr(worker, "_staged_proximal_update", None)
    if not isinstance(staged, _StagedActorUpdate) or staged.cohort_id != cohort_id:
        raise ValueError("No matching staged actor update")
    if "error" in staged.observation or staged.request is None or staged.data is None:
        raise ValueError("A rejected actor update cannot be committed")
    tracker = worker.actor.engine.enable_parameter_tracking()
    with tracker.commit_reserved_update(staged.observation["version"], cohort_id):
        worker._staged_proximal_update = None
        worker._prepared_proximal_cohort = None
        actor_config = worker.config.actor
        diagnostic_entropy = actor_config.entropy_coeff == 0 and not actor_config.calculate_entropy
        with proximal_worker_loss_scope(worker.actor, tracker, diagnostic_entropy=diagnostic_entropy):
            output = worker.actor.train_mini_batch(data=staged.data)
        # A failed output transfer follows an already-applied optimizer update.
        # Keep it within the fail-stop guard so this actor cannot be reused.
        output = output.cpu() if output is not None else None
    return output
