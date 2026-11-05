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

"""Preparation receipts only; no Ray, TQ or distributed training."""

from dataclasses import replace

import pytest

from verl_omni.pipelines.proximal_reuse.rpc import (
    PreparedCohort,
    PrepareRequest,
    RankReceipt,
    validate_preparation_observations,
)
from verl_omni.workers.utils.cohort_data import CohortData
from verl_omni.workers.utils.parameter_revision import ParameterVersion


def observations():
    cohort = CohortData(("key0", "key1"), ("input_ids",), ("a" * 64, "b" * 64), "c" * 64)
    request = PrepareRequest("cohort", "train", cohort, (0, 1, 0, 1))
    ranks = [
        RankReceipt(rank, dp, ParameterVersion(f"worker-{rank}", 7), cohort.shard([dp]), "d" * 64)
        for rank, dp in enumerate(request.dp_mapping)
    ]
    return request, [{"request": request, "receipt": rank} for rank in ranks]


def test_all_rank_receipts_preserve_native_interleaved_dispatch():
    request, replies = observations()
    result = validate_preparation_observations(request, replies)
    assert result == PreparedCohort(request, tuple(reply["receipt"] for reply in replies))


def test_hidden_replica_error_and_missing_reply_fail_before_preparation():
    request, replies = observations()
    replies[2] = {"error": "stale version"}
    with pytest.raises(ValueError, match="stale version"):
        validate_preparation_observations(request, replies)
    with pytest.raises(ValueError, match="every actor"):
        validate_preparation_observations(request, observations()[1][:-1])
    with pytest.raises(ValueError, match="stale version"):
        validate_preparation_observations(None, replies)


@pytest.mark.parametrize("fault", ["rank", "bool-rank", "dp", "incarnation", "revision", "replica"])
def test_bad_rank_or_replica_receipt_is_rejected(fault):
    request, replies = observations()
    index = 1 if fault == "bool-rank" else 2
    target = replies[index]["receipt"]
    if fault == "rank":
        target = replace(target, rank=0)
    elif fault == "bool-rank":
        target = replace(target, rank=True)
    elif fault == "dp":
        target = replace(target, dp_rank=1)
    elif fault == "incarnation":
        target = replace(target, version=replies[0]["receipt"].version)
    elif fault == "revision":
        target = replace(target, version=replace(target.version, revision=8))
    else:
        target = replace(target, engine_digest="e" * 64)
    replies[index] = {"request": request, "receipt": target}
    with pytest.raises(ValueError):
        validate_preparation_observations(request, replies)


def test_agreeing_replicas_cannot_bless_changed_shard():
    request, replies = observations()
    for index in (0, 2):
        original = replies[index]["receipt"]
        changed = replace(original.binding, raw_rows=("f" * 64,))
        replies[index] = {"request": request, "receipt": replace(original, binding=changed)}
    with pytest.raises(ValueError, match="differs from the requested cohort"):
        validate_preparation_observations(request, replies)


def test_bad_request_and_malformed_observation_are_rejected():
    request, replies = observations()
    with pytest.raises(ValueError, match="DP mapping"):
        validate_preparation_observations(replace(request, dp_mapping=(0, 2, 0, 2)), replies)
    with pytest.raises(ValueError, match="DP mapping"):
        validate_preparation_observations(replace(request, dp_mapping=(0, 10**12, 0, 1)), replies)
    replies[0] = {"request": request}
    with pytest.raises(ValueError, match="malformed"):
        validate_preparation_observations(request, replies)
