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

"""Cohort validation before the actor-only control plane enters training."""

from dataclasses import replace

import pytest

from verl_omni.workers.utils.cohort_lease import CohortPlan, ShardIdentity, validate_observations
from verl_omni.workers.utils.parameter_revision import ParameterVersion


@pytest.fixture
def cohort():
    shards = (
        ShardIdentity(("a", "b"), "a" * 64, "b" * 64, "c" * 64),
        ShardIdentity(("c", "d"), "d" * 64, "e" * 64, "f" * 64),
    )
    versions = tuple(ParameterVersion(f"actor-{rank}", 3) for rank in range(4))
    plan = CohortPlan("cohort", "train", ((0, 2), (1, 3)), shards, versions)
    observations = [
        {
            "cohort_id": "cohort",
            "partition_id": "train",
            "rank": rank,
            "dp_rank": rank % 2,
            "version": versions[rank],
            "shard": shards[rank % 2],
        }
        for rank in range(4)
    ]
    return plan, observations


def test_all_actor_shards_and_versions_are_accepted(cohort):
    plan, observations = cohort
    validate_observations(plan, observations)


@pytest.mark.parametrize("field", ["cohort_id", "partition_id", "rank", "dp_rank", "version", "shard"])
def test_hidden_replica_cannot_change_any_approved_field(cohort, field):
    plan, observations = cohort
    observations[2] = {**observations[2], field: object()}
    with pytest.raises(ValueError, match="differs"):
        validate_observations(plan, observations)


def test_actor_rejection_aborts_cohort_before_training(cohort):
    plan, observations = cohort
    observations[2] = {"error": "stale actor"}
    with pytest.raises(RuntimeError, match="Rank 2.*stale"):
        validate_observations(plan, observations)


def test_repeated_keys_and_worker_identity_are_rejected(cohort):
    plan, _ = cohort
    duplicate = replace(plan.shards[1], keys=("a", "d"))
    with pytest.raises(ValueError, match="Repeated sample key"):
        replace(plan, shards=(plan.shards[0], duplicate)).validate(4)
    with pytest.raises(ValueError, match="incarnation is reused"):
        replace(plan, versions=(plan.versions[0], plan.versions[0], *plan.versions[2:])).validate(4)


def test_incomplete_or_overlapping_worker_groups_are_rejected(cohort):
    plan, _ = cohort
    for groups in (((0, 2), (1,)), ((0, 2), (1, 2))):
        with pytest.raises(ValueError, match="cover every training rank"):
            replace(plan, ranks_by_dp=groups).validate(4)
