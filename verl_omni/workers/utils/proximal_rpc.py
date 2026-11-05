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

"""Native-dispatch preparation and atomic update for an explicitly qualified cohort.

These RPCs do not establish algorithm/model eligibility. Callers must qualify
that separately before preparation; normal compute_log_prob/update_actor are
unchanged. Each receipt uses the actual TQ shard and actual engine config.
"""

from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from types import SimpleNamespace

import torch.distributed as dist
from omegaconf import OmegaConf
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.actor import PolicyLossConfig

from verl_omni.workers.utils.cohort_data import CohortData, value_digest
from verl_omni.workers.utils.cohort_lease import CohortPlan, collective_update
from verl_omni.workers.utils.parameter_revision import ParameterVersion
from verl_omni.workers.utils.proximal_anchor import proximal_anchor_loss_scope

REQUEST = "_proximal_cohort_request"
TRANSPORT = "_proximal_transport_identity"


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value):
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
    return value_digest({"engine": _plain(engine.engine_config), "proximal": engine.proximal_execution()})


def _check_actual_loss(worker):
    from verl_omni.workers.utils.proximal_anchor import proximal_ppo_loss as ppo_loss

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


def _gather(value, group):
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def qualify_actor(worker):
    """Read-only capability query before reserving any cohort or model work."""
    try:
        engine = worker.actor.engine
        if not callable(getattr(engine, "proximal_execution", None)):
            raise ValueError("Engine does not support qualified actor reuse")
        _check_actual_loss(worker)
        digest = _engine_digest(engine)
        if worker.enable_routing_replay:
            raise ValueError("Routing replay requires independent old inference")
        return {"supported": True, "execution_digest": digest}
    except (ValueError, TypeError, AttributeError) as exc:
        return {"supported": False, "reason": str(exc)}


def prepare_actor(worker, data, *, group):
    """Prepare on every native actor rank before advantage calculation.

    A rank-local failure is exchanged before any model work. No optimizer,
    forward, device movement or tensor parameter write occurs in this method.
    """
    engine = worker.actor.engine
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    worker._prepared_proximal_cohort = None
    try:
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
    observations = _gather(observation, group)
    if any(item.get("error") for item in observations):
        raise ValueError(f"Actor preparation rejected before forward: {observations}")
    if any(item["request"] != request for item in observations):
        raise ValueError("Ranks received different coordinator requests")
    receipts = tuple(item["receipt"] for item in observations)
    if any(r.rank != i or r.dp_rank != request.dp_mapping[i] for i, r in enumerate(receipts)):
        raise ValueError("Actual training ranks differ from native dispatch")
    if len({r.version.incarnation for r in receipts}) != size or len({r.version.revision for r in receipts}) != 1:
        raise ValueError("Training engine identities or writer histories differ")
    all_keys = []
    for dp in sorted(set(request.dp_mapping)):
        replicas = [r for r in receipts if r.dp_rank == dp]
        if any(r.binding != replicas[0].binding or r.engine_digest != replicas[0].engine_digest for r in replicas):
            raise ValueError("Model-parallel replicas received different data or execution settings")
        all_keys.extend(replicas[0].binding.keys)
    if len(all_keys) != len(set(all_keys)) or set(all_keys) != set(request.binding.keys):
        raise ValueError("Native DP dispatch lost or duplicated rows")
    prepared = PreparedCohort(request, receipts)
    worker._prepared_proximal_cohort = prepared
    return tu.get_tensordict({}, non_tensor_dict={"proximal_preparation": prepared})


def update_actor(worker, data, *, group):
    """Enter the collective lease OUTSIDE the protected native training call."""
    engine = worker.actor.engine
    # A malformed request may be rank-local; build rejection inside the
    # collective callback using the locally stored, already-agreed plan.
    prepared = getattr(worker, "_prepared_proximal_cohort", None)
    request_error = None
    try:
        request = tu.get_non_tensor_data(data, REQUEST, default=None)
        if prepared is None:
            raise RuntimeError("No successfully prepared actor cohort")
        if not isinstance(request, UpdateRequest) or request.prepared != prepared:
            raise ValueError("Update does not match the locally prepared cohort")
        plan = request.plan
        if (
            plan.versions != tuple(r.version for r in prepared.ranks)
            or plan.cohort_id != prepared.request.cohort_id
            or plan.partition_id != prepared.request.partition_id
        ):
            raise ValueError("Update attempted to replace the reserved parameter versions")
    except Exception as exc:
        request_error = exc
        plan = None
    # Exchange the update plan before entering parameter guards, so malformed
    # metadata on one rank cannot strand peers in the model's collectives.
    plans = _gather((plan, str(request_error) if request_error else None), group)
    worker._prepared_proximal_cohort = None  # Consume once, including failures.
    if any(error for _, error in plans) or any(p != plans[0][0] for p, _ in plans):
        raise ValueError("Actor update plans differ or contain a rank-local error")
    rank = dist.get_rank(group)
    tracker = engine.enable_parameter_tracking()

    def observe():
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
        if tu.get_non_tensor_data(data, "no_lora_adapter", False):
            raise ValueError("Adapter mode differs from prepared actor")
        if tu.get_non_tensor_data(data, "distillation_use_topk", False) or tu.get_non_tensor_data(
            data, "score_centering", False
        ):
            raise ValueError("Unsupported alternate training objective")
        return binding.observe(data, keys, execution)

    with collective_update(tracker, plan, observe, engine.get_data_parallel_rank(), group=group):
        actor_config = worker.config.actor
        diagnostic_entropy = actor_config.entropy_coeff == 0 and not actor_config.calculate_entropy
        with proximal_anchor_loss_scope(tracker, diagnostic_entropy=diagnostic_entropy):
            output = worker.actor.train_mini_batch(data=data)
    return output.cpu() if output is not None else None
