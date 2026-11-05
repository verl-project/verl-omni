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

"""Collectively bind one update to each DP shard and each actual actor version.

Callers must compute identities from the actual prepared data/config, not from
trainer step counters. This module does not compute those fingerprints or
establish that an algorithm is eligible to omit its old-policy forward.
"""

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch.distributed as dist

from verl_omni.workers.utils.parameter_revision import ParameterRevision, ParameterVersion


@dataclass(frozen=True)
class ShardIdentity:
    keys: tuple[str, ...]
    raw_digest: str
    derived_digest: str
    execution_digest: str


@dataclass(frozen=True)
class CohortPlan:
    cohort_id: str
    partition_id: str
    # Group-relative ranks, not global process-group ranks.
    ranks_by_dp: tuple[tuple[int, ...], ...]
    shards: tuple[ShardIdentity, ...]
    versions: tuple[ParameterVersion, ...]

    def validate(self, group_size):
        if not self.cohort_id or not self.partition_id:
            raise ValueError("Missing cohort or partition")
        if not self.shards or len(self.shards) != len(self.ranks_by_dp):
            raise ValueError("DP shard and topology counts differ")
        if any(not ranks for ranks in self.ranks_by_dp):
            raise ValueError("Empty model-parallel replica group")
        ranks = [rank for group in self.ranks_by_dp for rank in group]
        if sorted(ranks) != list(range(group_size)) or len(self.versions) != group_size:
            raise ValueError("The plan must cover every training rank exactly once")
        if any(
            not isinstance(version, ParameterVersion) or not version.incarnation or version.revision < 0
            for version in self.versions
        ):
            raise ValueError("Invalid engine incarnation or parameter revision")
        if len({version.incarnation for version in self.versions}) != group_size:
            raise ValueError("A worker incarnation is reused by multiple training ranks")
        if len({version.revision for version in self.versions}) != 1:
            raise ValueError("Training ranks have different parameter-writing histories")
        keys = []
        for shard in self.shards:
            if not shard.keys or any(not isinstance(key, str) or not key for key in shard.keys):
                raise ValueError("Empty shard or sample key")
            for digest in (shard.raw_digest, shard.derived_digest, shard.execution_digest):
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError("Expected a complete SHA256 data/config identity")
            keys.extend(shard.keys)
        if len(keys) != len(set(keys)):
            raise ValueError("Repeated sample key inside or across DP shards")


def validate_observations(plan: CohortPlan, observations: list[dict]):
    """All MP replicas agree within a DP shard; different DP shards stay distinct."""
    plan.validate(len(observations))
    for rank, observation in enumerate(observations):
        if observation.get("error"):
            raise RuntimeError(f"Rank {rank} rejected the cohort: {observation['error']}")
        dp_rank = next(i for i, ranks in enumerate(plan.ranks_by_dp) if rank in ranks)
        expected = {
            "rank": rank,
            "dp_rank": dp_rank,
            "cohort_id": plan.cohort_id,
            "partition_id": plan.partition_id,
            "version": plan.versions[rank],
            "shard": plan.shards[dp_rank],
        }
        if observation != expected:
            raise ValueError(f"Rank {rank} actor version, topology or shard identity differs")


def _gather(value, group):
    observations = [None] * dist.get_world_size(group)
    dist.all_gather_object(observations, value, group=group)
    return observations


@contextmanager
def collective_update(
    tracker: ParameterRevision,
    plan: CohortPlan,
    actual_shard: ShardIdentity | Callable[[], ShardIdentity],
    dp_rank: int,
    *,
    group,
):
    """Reject every rank before model work if any local version/shard check fails.

    The supplied group must contain exactly the training workers. It must not
    include colocated rollout workers. A failed/partial update poisons all
    participating trackers; it is never automatically retried. Broken model
    collectives still depend on the framework/process-group failure timeout.
    """
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    with ExitStack() as stack:
        try:
            plan.validate(size)
            actual_shard = actual_shard() if callable(actual_shard) else actual_shard
            version = tracker.version()
            stack.enter_context(tracker.update(plan.versions[rank], plan.cohort_id))
            observation = {
                "rank": rank,
                "dp_rank": dp_rank,
                "cohort_id": plan.cohort_id,
                "partition_id": plan.partition_id,
                "version": version,
                "shard": actual_shard,
            }
        except Exception as exc:
            observation = {"error": f"{type(exc).__name__}: {exc}"}
        # Local rejection must still enter this collective, before any engine
        # context, forward or gradient communication on another rank.
        validate_observations(plan, _gather(observation, group))
        failure = None
        try:
            yield
            tracker.assert_update_complete()
        except BaseException as exc:
            failure = exc
        statuses = _gather(None if failure is None else f"{type(failure).__name__}: {failure}", group)
        if any(status is not None for status in statuses):
            raise RuntimeError(f"Atomic cohort failed on training ranks: {statuses}") from failure
