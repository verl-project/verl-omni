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
"""Serving metadata validation needs no episode collector or trajectory state."""

import sys

import pytest

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import serving_rollout_provenance


def test_observed_serving_identity_is_preserved_without_loading_the_collector():
    loaded = set(sys.modules)
    result = serving_rollout_provenance(
        {"rollout_policy_version": 7, "rollout_replica_id": "replica-1", "serving_provenance_verified": True},
        required=True,
    )
    assert result == {
        "rollout_policy_version": 7,
        "rollout_replica_id": "replica-1",
        "serving_provenance_verified": True,
    }
    assert "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode" not in set(sys.modules) - loaded
    assert serving_rollout_provenance.__module__ == "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving"


@pytest.mark.parametrize(
    "field,value", [("rollout_policy_version", True), ("rollout_policy_version", -1), ("rollout_replica_id", "")]
)
def test_malformed_observed_identity_is_rejected(field, value):
    with pytest.raises(ValueError, match="Invalid observed serving"):
        serving_rollout_provenance({field: value})


def test_a_request_step_cannot_stand_in_for_observed_serving_identity():
    assert not serving_rollout_provenance({"global_steps": 7})["serving_provenance_verified"]
    with pytest.raises(ValueError, match="observed serving version"):
        serving_rollout_provenance({"global_steps": 7}, required=True)
