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
"""Assert CRD training metrics and checkpoint files; GPU state restoration has its own regression."""

import argparse
import math
import re
from pathlib import Path


def check_run(log_path: Path, output_dir: Path, steps: list[int]):
    """Reject missing/nonfinite updates and incomplete per-rank checkpoints."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text())
    if "skipping this optimizer step" in text:
        raise ValueError("An optimizer update was skipped.")
    seen = set()
    for line in text.splitlines():
        if not re.search(r"(?:actor/)+crd_loss:", line):
            continue
        step_match = re.search(r"\bstep:(\d+)\b", line)
        if step_match is None:
            continue
        seen.add(int(step_match[1]))
        for key in ("crd_loss", "crd_policy_loss", "crd_kl_loss", "grad_norm"):
            match = re.search(rf"(?:actor/)+{key}:([^\s]+)", line)
            if match is None or not math.isfinite(float(match[1])):
                raise ValueError(f"Missing or nonfinite {key} at step {step_match[1]}.")
            if key == "grad_norm" and float(match[1]) <= 0:
                raise ValueError(f"Zero CRD gradient at step {step_match[1]}.")
    if seen != set(steps):
        raise ValueError(f"Expected training steps {steps}, found {sorted(seen)}.")
    for step in steps:
        actor = output_dir / f"global_step_{step}" / "actor"
        ranks = []
        for prefix in ("model", "optim", "extra_state"):
            files = list(actor.glob(f"{prefix}_world_size_*_rank_*.pt"))
            if not files or any(p.stat().st_size == 0 for p in files):
                raise ValueError(f"Missing/empty {prefix} checkpoint at step {step}.")
            parsed = [re.fullmatch(rf"{prefix}_world_size_(\d+)_rank_(\d+)\.pt", p.name) for p in files]
            sizes = {int(m[1]) for m in parsed}
            rank_set = {int(m[2]) for m in parsed}
            if len(sizes) != 1 or rank_set != set(range(next(iter(sizes)))):
                raise ValueError(f"Incomplete {prefix} checkpoint ranks at step {step}.")
            ranks.append((sizes, rank_set))
        if ranks[1:] != ranks[:-1]:
            raise ValueError(f"Model/optimizer/extra rank sets differ at step {step}.")


def main():
    """Check one fresh or resumed console log and its saved checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    check_run(args.log, args.output_dir, args.steps)
    print(f"CRD finite metrics and complete checkpoints verified for steps {args.steps}.")


if __name__ == "__main__":
    main()
