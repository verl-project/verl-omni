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

"""Paired prompt-cluster evaluation, stratified by task and training seed.

JSONL rows require task, training_seed, prompt_id, render_seed, kernel,
kernel_config_hash, scorer_version, eval_config_hash, prompt_hash and all metrics
below. Run separately for training SDE and deployment ODE. A missing pair is an
error, never silently dropped. NumPy is the only non-stdlib dependency.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

METRICS = ("image_quality", "protocol_valid", "autonomous_done", "assisted_reflection", "num_gen_calls", "rewrite_gain")
IDENTITY = ("task", "training_seed", "prompt_id", "render_seed")
CONSTANT = ("kernel", "kernel_config_hash", "scorer_version", "eval_config_hash")


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation records are empty")
    result = {}
    for row in rows:
        for field in (*IDENTITY, *CONSTANT, "prompt_hash", *METRICS):
            if field not in row or row[field] is None:
                raise ValueError(f"Missing required field: {field}")
        key = tuple(row[k] for k in IDENTITY)
        if key in result:
            raise ValueError(f"Duplicate evaluation pair: {key}")
        for field in METRICS:
            if not isinstance(row[field], float | int) or not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite metric {field}: {key}")
        for field in ("protocol_valid", "autonomous_done", "assisted_reflection"):
            if row[field] not in (0, 1):
                raise ValueError(f"{field} must be 0/1")
        if row["num_gen_calls"] < 0:
            raise ValueError("num_gen_calls must be nonnegative")
        result[key] = row
    return result


def summarize(effect, samples):
    return {
        name: {"effect": float(effect[i]), "ci95": [float(v) for v in np.quantile(samples[:, i], [0.025, 0.975])]}
        for i, name in enumerate(METRICS)
    }


def compare_checkpoints(
    baseline_records: Path, candidate_records: Path, *, bootstrap_seed: int = 42, n_bootstrap: int = 10000
) -> dict:
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be >= 2")
    baseline, candidate = read_records(baseline_records), read_records(candidate_records)
    if baseline.keys() != candidate.keys():
        raise ValueError("Missing sample pair between baseline and candidate")
    for field in CONSTANT:
        values = {row[field] for row in [*baseline.values(), *candidate.values()]}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"Inconsistent evaluation {field}; compare kernels separately")
    groups = {}
    prompt_metadata = {}
    seed_sets = {}
    for key in sorted(baseline, key=lambda value: tuple(map(str, value))):
        old, new = baseline[key], candidate[key]
        if old["prompt_hash"] != new["prompt_hash"]:
            raise ValueError(f"Prompt identity mismatch: {key}")
        task, seed, prompt_id, render_seed = key
        prompt_key = (task, prompt_id)
        if prompt_key in prompt_metadata and prompt_metadata[prompt_key] != old["prompt_hash"]:
            raise ValueError(f"Prompt contents changed across seeds: {prompt_key}")
        prompt_metadata[prompt_key] = old["prompt_hash"]
        group = groups.setdefault((task, seed), {})
        group.setdefault(prompt_id, []).append([new[m] - old[m] for m in METRICS])
        seed_sets.setdefault((task, seed, prompt_id), set()).add(render_seed)
    if len({frozenset(value) for value in seed_sets.values()}) != 1:
        raise ValueError("Every prompt must have the same exact render seed list")
    tasks = sorted({key[0] for key in groups})
    seeds = sorted({key[1] for key in groups}, key=str)
    if len(tasks) != 2 or any((task, seed) not in groups for task in tasks for seed in seeds):
        raise ValueError("Require exactly two task strata for every training seed")
    for task in tasks:
        if len({frozenset(groups[(task, seed)]) for seed in seeds}) != 1:
            raise ValueError("Training seeds must evaluate the same prompt panel")
    rng = np.random.default_rng(bootstrap_seed)
    by_seed, bootstrap, effects = {}, {}, {}
    passes, sufficient = [], []
    for seed in seeds:
        by_seed[str(seed)] = {}
        for task in tasks:
            prompts = groups[(task, seed)]
            # Average correlated images before sampling independent prompt clusters.
            values = np.array([np.mean(prompts[prompt], axis=0) for prompt in sorted(prompts, key=str)])
            draws = np.empty((n_bootstrap, len(METRICS)))
            for start in range(0, n_bootstrap, 256):
                count = min(256, n_bootstrap - start)
                index = rng.integers(0, len(values), size=(count, len(values)))
                draws[start : start + count] = values[index].mean(axis=1)
            effect = values.mean(axis=0)
            stats = summarize(effect, draws)
            stats["prompt_clusters"] = len(values)
            stats["rendered_pairs"] = sum(len(rows) for rows in prompts.values())
            stats["quality_gate"] = stats["image_quality"]["ci95"][0] > 0
            stats["protocol_gate"] = stats["protocol_valid"]["effect"] >= -0.02
            stats["sufficient_panel"] = len(values) >= 200
            sufficient.append(stats["sufficient_panel"])
            passes.append(stats["quality_gate"] and stats["protocol_gate"])
            by_seed[str(seed)][task] = stats
            bootstrap[(task, seed)], effects[(task, seed)] = draws, effect
    # Resample training seeds, then independent paired prompt-cluster draws.
    hierarchy = np.zeros((n_bootstrap, len(METRICS)))
    for i in range(n_bootstrap):
        for selected in rng.integers(0, len(seeds), size=len(seeds)):
            seed = seeds[selected]
            for task in tasks:
                hierarchy[i] += bootstrap[(task, seed)][rng.integers(n_bootstrap)] / (2 * len(seeds))
    aggregate = summarize(np.mean(list(effects.values()), axis=0), hierarchy)
    aggregate.update(
        task_weights={task: 0.5 for task in tasks},
        method="hierarchical training-seed then prompt-cluster bootstrap",
        training_seeds=len(seeds),
        uncertainty_note="Limited training-seed uncertainty, especially with only three seeds",
    )
    return {
        "schema_version": 1,
        "bootstrap_seed": bootstrap_seed,
        "n_bootstrap": n_bootstrap,
        "evaluation": {field: next(iter(baseline.values()))[field] for field in CONSTANT},
        "by_training_seed": by_seed,
        "aggregate": aggregate,
        "quality_protocol_gate_passed": all(passes) and all(sufficient) and len(seeds) >= 3,
        "generalization_claim_ready": False,
        "pending": ["Independent frozen scorer or blinded paired review", "Useful rewrite gain at reported image cost"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    result = compare_checkpoints(
        args.baseline, args.candidate, bootstrap_seed=args.bootstrap_seed, n_bootstrap=args.n_bootstrap
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
