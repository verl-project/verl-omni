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

"""Validate captured frozen-policy replay rows and emit calibrated parity gates.

This consumes host-generated artifacts; it does not launch models or infer the
installed vLLM probability convention. --config must predeclare calibration and
the exact logprob convention. Run capture before and after each publication.
"""

import argparse
import json
import math
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def distribution(values):
    return {"p50": quantile(values, 0.5), "p95": quantile(values, 0.95), "max": max(values)}


def check_policy_parity(rows: list[dict], config: dict) -> dict:
    required = (
        "policy_version",
        "expected_replicas",
        "actor_repeat_tolerance",
        "gen_ratio_p95_tolerance",
        "und_mean_gap_baseline",
        "und_mean_gap_tolerance",
        "logprob_convention",
        "calibration_id",
    )
    for key in required:
        if key not in config or config[key] is None:
            raise ValueError(f"Parity config must predeclare {key}")
    if config["logprob_convention"] not in {"raw_policy", "exact_behavior_transforms"} or not config["calibration_id"]:
        raise ValueError("Require verified logprob convention and calibration_id")
    for key in ("actor_repeat_tolerance", "gen_ratio_p95_tolerance", "und_mean_gap_tolerance"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid tolerance: {key}")
    if not math.isfinite(config["und_mean_gap_baseline"]):
        raise ValueError("Nonfinite UND baseline")
    declared = config["expected_replicas"]
    if isinstance(declared, list):
        declared = {"und": declared, "gen": declared}
    if not isinstance(declared, dict) or set(declared) != {"und", "gen"}:
        raise ValueError("Expected replicas must list both UND and GEN lane topologies")
    expected = {}
    for lane, ids in declared.items():
        if not isinstance(ids, list) or not ids or any(not isinstance(replica, str | int) for replica in ids):
            raise ValueError(f"Expected replicas for {lane} must be a nonempty list of IDs")
        expected[lane] = set(ids)
        if len(expected[lane]) != len(ids):
            raise ValueError(f"Expected replicas for {lane} must be unique")
    seen, groups, contexts, coverage = set(), {"und": [], "gen": []}, {}, {}
    replicas = {"und": set(), "gen": set()}
    kernels = {"und": set(), "gen": set()}
    for row in rows:
        for key in (
            "lane",
            "policy_version",
            "replica_id",
            "episode_id",
            "gen_row_id",
            "timestep",
            "context_hash",
            "kernel_config_hash",
            "rollout_logp",
            "actor_old_logp",
            "actor_repeat_logp",
        ):
            if key not in row:
                raise ValueError(f"Missing parity field: {key}")
        lane = row["lane"]
        if lane not in groups or row["replica_id"] not in expected[lane]:
            raise ValueError("Unknown lane or replica")
        if row["policy_version"] != config["policy_version"]:
            raise ValueError("Stale policy version")
        if not row["context_hash"] or not row["kernel_config_hash"]:
            raise ValueError("Missing context/kernel hash")
        key = (lane, row["policy_version"], row["replica_id"], row["episode_id"], row["gen_row_id"], row["timestep"])
        if key in seen:
            raise ValueError("Duplicate parity row")
        seen.add(key)
        context_key = (lane, row["episode_id"], row["gen_row_id"], row["timestep"])
        if context_key in contexts and contexts[context_key] != row["context_hash"]:
            raise ValueError("Context differs across replicas")
        contexts[context_key] = row["context_hash"]
        coverage.setdefault(context_key, set()).add(row["replica_id"])
        if any(
            not isinstance(row[k], int | float) or not math.isfinite(row[k])
            for k in ("rollout_logp", "actor_old_logp", "actor_repeat_logp")
        ):
            raise ValueError("Nonfinite parity log-probability")
        replicas[lane].add(row["replica_id"])
        kernels[lane].add(row["kernel_config_hash"])
        groups[lane].append(row)
    if any(replicas != expected[key[0]] for key, replicas in coverage.items()):
        raise ValueError("Every captured context requires complete replica coverage")
    result = {
        "schema_version": 1,
        "policy_version": config["policy_version"],
        "calibration_id": config["calibration_id"],
        "logprob_convention": config["logprob_convention"],
        "lanes": {},
    }
    for lane, values in groups.items():
        if not values or replicas[lane] != expected[lane] or len(kernels[lane]) != 1:
            raise ValueError(f"Incomplete replicas or inconsistent kernel for {lane}")
        errors = [r["actor_old_logp"] - r["rollout_logp"] for r in values]
        repeat = [abs(r["actor_old_logp"] - r["actor_repeat_logp"]) for r in values]
        ratios = [abs(math.expm1(value)) if value < 700 else float("inf") for value in errors]
        # Saturate overflow for strict JSON artifacts while preserving a failed gate.
        ratios = [min(value, 1e300) for value in ratios]
        stats = {
            "rows": len(values),
            "log_ratio_error": distribution([abs(v) for v in errors]),
            "ratio_error": distribution(ratios),
            "actor_repeat_error": distribution(repeat),
            "mean_signed_logp_gap": sum(errors) / len(errors),
            "repeat_passed": max(repeat) < config["actor_repeat_tolerance"],
        }
        stats["cross_engine_passed"] = (
            stats["ratio_error"]["p95"] < config["gen_ratio_p95_tolerance"]
            if lane == "gen"
            else abs(stats["mean_signed_logp_gap"] - config["und_mean_gap_baseline"]) < config["und_mean_gap_tolerance"]
        )
        stats["passed"] = stats["repeat_passed"] and stats["cross_engine_passed"]
        result["lanes"][lane] = stats
    result["passed"] = all(stats["passed"] for stats in result["lanes"].values())
    result["scope"] = "Captured rows only; publication coverage and exact token/mask capture require host verification"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON with calibration and parity_artifact path")
    parser.add_argument("--artifact", type=Path, help="Override config parity_artifact JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Artifact directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = args.artifact or Path(config["parity_artifact"])
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = check_policy_parity(rows, config)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "policy_parity.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    raise SystemExit(0 if result["passed"] else 2)
