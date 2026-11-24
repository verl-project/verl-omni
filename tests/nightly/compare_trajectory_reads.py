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
"""Report a matched L3 trajectory read A/B; gate only the payload reduction."""

import argparse
import json
from pathlib import Path
from statistics import median

_BYTES = "training/tq_driver_tensor_bytes"
_IMAGES = "perf/total_num_images"
_METRICS = (_BYTES, "timing_s/tq_driver_read", "perf/time_per_step", "perf/throughput")


def _read_metrics(path: Path, steps: int) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["metrics_for_compare"]
    expected_steps = list(range(payload["perf_skip_steps"] + 1, steps + 1))
    if not expected_steps or [record["step"] for record in records] != expected_steps:
        raise ValueError(f"{path}: expected one metric record per post-warmup step: {expected_steps}")
    if any(record["data"]["train/nonfinite_grad_steps"] for record in records):
        raise ValueError(f"{path}: found nonfinite gradients in the measured steps")
    return {key: [record["data"][key] for record in records] for key in (*_METRICS, _IMAGES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", type=Path)
    parser.add_argument("projected", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    full, projected = (_read_metrics(path, args.steps) for path in (args.full, args.projected))
    if full[_IMAGES] != projected[_IMAGES] or min(full[_IMAGES]) <= 0:
        raise ValueError("Full and projected runs must train on the same number of images each step.")
    passed = all(0 < after < before for before, after in zip(full[_BYTES], projected[_BYTES], strict=True))
    report = {
        "passed": passed,
        "post_warmup_steps": len(full[_BYTES]),
        "images_per_step": full[_IMAGES],
        "medians": {key: {"full": median(full[key]), "projected": median(projected[key])} for key in _METRICS},
        "driver_tensor_bytes_per_step": {"full": full[_BYTES], "projected": projected[_BYTES]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps(report, indent=2) + "\n"
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if not passed:
        raise SystemExit("Projected driver reads did not return fewer tensor bytes on every measured step.")


if __name__ == "__main__":
    main()
