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
"""Check that a replay pass reproduced its record pass.

Reads the two ``trainer.logger=file`` metric files and exits 1 unless every
training metric matches exactly at every step and each replayed stage reports
all its steps as skipped. Timing and throughput metrics are not compared.
"""

import argparse
import json
from pathlib import Path

COMPARED_PREFIXES = ("actor/", "critic/")


def load_metrics(path: Path) -> dict[int, dict]:
    """Merge the metric rows of a ``file`` logger output per step."""
    metrics: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        metrics.setdefault(row["step"], {}).update(row["data"])
    return metrics


def training_metrics(metrics: dict[int, dict]) -> dict[tuple[int, str], float]:
    return {
        (step, key): value
        for step, data in metrics.items()
        for key, value in data.items()
        if key.startswith(COMPARED_PREFIXES)
    }


def find_problems(record: dict[int, dict], replay: dict[int, dict], stages: list[str], num_steps: int) -> list[str]:
    """Return one line per reason the replay does not reproduce the record pass."""
    expected, actual = training_metrics(record), training_metrics(replay)
    if not expected:
        return ["no comparable training metrics in the record pass"]
    problems = [
        f"step {step} {key}: record={expected.get((step, key))} replay={actual.get((step, key))}"
        for step, key in sorted(expected.keys() | actual.keys())
        if expected.get((step, key), "missing") != actual.get((step, key), "missing")
    ]
    replay_totals = {key: value for data in replay.values() for key, value in data.items()}
    for stage in stages:
        skipped_steps = replay_totals.get(f"skip/{stage}/steps")
        if skipped_steps != num_steps:
            problems.append(f"skip/{stage}/steps: expected {num_steps} replayed steps, got {skipped_steps}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", type=Path, help="Metric file of the record pass.")
    parser.add_argument("replay", type=Path, help="Metric file of the replay pass.")
    parser.add_argument("--num-steps", type=int, required=True, help="Training steps each pass ran.")
    parser.add_argument("--replayed-stages", nargs="*", default=[], help="omni_skip roles expected to have replayed.")
    args = parser.parse_args()

    record, replay = load_metrics(args.record), load_metrics(args.replay)
    problems = find_problems(record, replay, args.replayed_stages, args.num_steps)
    if problems:
        print(f"REPLAY MISMATCH ({len(problems)} problems):")
        print("\n".join(problems))
        return 1
    compared = len(training_metrics(record))
    print(f"REPLAY OK: {compared} training metric values match exactly over {args.num_steps} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
