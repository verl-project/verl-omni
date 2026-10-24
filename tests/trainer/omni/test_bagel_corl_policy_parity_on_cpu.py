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

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location("parity", ROOT / "scripts/bagel_corl/check_policy_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows():
    return [
        dict(
            lane=lane,
            policy_version="v2",
            replica_id=replica,
            episode_id="episode",
            gen_row_id=0,
            timestep=1,
            context_hash="ctx",
            kernel_config_hash="kernel",
            rollout_logp=-1.0,
            actor_old_logp=-1.00001,
            actor_repeat_logp=-1.00001,
        )
        for lane in ("und", "gen")
        for replica in (0, 1)
    ]


def config():
    return dict(
        policy_version="v2",
        expected_replicas=[0, 1],
        actor_repeat_tolerance=1e-5,
        gen_ratio_p95_tolerance=0.005,
        und_mean_gap_baseline=0.0,
        und_mean_gap_tolerance=0.01,
        logprob_convention="raw_policy",
        calibration_id="fixture",
    )


def test_reports_distributions_and_gates():
    result = load().check_policy_parity(rows(), config())
    assert result["passed"]
    assert result["lanes"]["gen"]["ratio_error"]["max"] > 0
    bad = rows()
    bad[0]["actor_repeat_logp"] += 0.1
    assert not load().check_policy_parity(bad, config())["passed"]


@pytest.mark.parametrize("mutation", ["version", "nan", "hash", "duplicate", "replica"])
def test_invalid_artifacts_fail_closed(mutation):
    data = rows()
    if mutation == "version":
        data[0]["policy_version"] = "v1"
    elif mutation == "nan":
        data[0]["rollout_logp"] = float("nan")
    elif mutation == "hash":
        data[0]["context_hash"] = ""
    elif mutation == "duplicate":
        data.append(dict(data[0]))
    else:
        data = [r for r in data if r["replica_id"] == 0]
    with pytest.raises(ValueError):
        load().check_policy_parity(data, config())


def test_every_captured_context_requires_all_replicas():
    data = rows()
    data[0]["episode_id"] = "different-episode"
    with pytest.raises(ValueError, match="coverage"):
        load().check_policy_parity(data, config())


def test_lane_specific_replica_topologies_and_missing_coverage():
    calibrated = dict(config(), expected_replicas={"und": [0, 1], "gen": [10]})
    data = [r for r in rows() if r["lane"] == "und"]
    data.append(dict(rows()[-1], replica_id=10))
    tool = load()
    assert tool.check_policy_parity(data, calibrated)["passed"]
    # Both UND replicas occur somewhere, but this context is incomplete.
    incomplete = [dict(row) for row in data]
    incomplete[0]["episode_id"] = "different-episode"
    with pytest.raises(ValueError, match="coverage"):
        tool.check_policy_parity(incomplete, calibrated)
    with pytest.raises(ValueError, match="Incomplete replicas"):
        tool.check_policy_parity(data[:-1], calibrated)


@pytest.mark.parametrize("expected", [{"und": [0]}, {"und": [0], "gen": []}, {"und": [0, 0], "gen": [1]}])
def test_lane_replica_configuration_fails_closed(expected):
    with pytest.raises(ValueError):
        load().check_policy_parity(rows(), dict(config(), expected_replicas=expected))
