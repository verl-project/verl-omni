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
"""Verify real V1 updates, validation media, and optimizer/checkpoint continuation."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image


def local(tensor):
    """Read a checkpoint tensor without starting another distributed job."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def main():
    """Reject successful process exits without completed, finite training updates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output / "checkpoints"
    assert int((root / "latest_checkpointed_iteration.txt").read_text()) == args.steps

    def actor(step):
        return root / f"global_step_{step}" / "actor"

    model = torch.load(actor(args.steps) / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    assert model and all(torch.isfinite(local(value)).all() for value in model.values())
    lora = {name: local(value) for name, value in model.items() if "lora_" in name}
    assert lora and any(torch.count_nonzero(value) for name, value in lora.items() if "lora_B" in name)
    del model
    previous = torch.load(
        actor(args.steps - 1) / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False
    )
    changed = sum(not torch.equal(value, local(previous[name])) for name, value in lora.items())
    assert changed > 0
    del previous

    def optimizer(step):
        state = torch.load(actor(step) / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
        active = [value for value in state["state"].values() if "step" in value]
        assert active and all(
            torch.isfinite(local(value[key])).all() for value in active for key in ("exp_avg", "exp_avg_sq")
        )
        return sorted({int(local(value["step"])) for value in active})

    current_steps, previous_steps = optimizer(args.steps), optimizer(args.steps - 1)
    assert min(current_steps) > max(previous_steps) > 0
    log = args.log.read_text()
    norms = [float(value) for value in re.findall(r"actor/grad_norm[^0-9+\-]*([0-9.eE+\-]+)", log)]
    assert len(norms) >= args.steps - args.resume_from
    assert all(math.isfinite(value) and value > 0 for value in norms)
    if args.resume_from:
        assert re.search(rf"Found checkpoint:.*global_step_{args.resume_from}\b", log)
    media = {}
    for step in range(1, args.steps + 1):
        for kind, expected in (("rollouts", 4), ("validation", 2)):
            rows = [json.loads(line) for line in (args.output / kind / f"{step}.jsonl").read_text().splitlines()]
            assert len(rows) == expected and all(math.isfinite(row["score"]) for row in rows)
            if kind == "rollouts":
                groups = Counter(row["uid"].rsplit("_", 2)[0] for row in rows)
                assert len(groups) == 2 and set(groups.values()) == {2}
            for row in rows:
                with Image.open(row["output"]) as image:
                    assert image.size == (args.image_size, args.image_size)
                    assert any(low < high for low, high in image.getextrema())
            media[f"{kind}/{step}"] = len(rows)
    result = {
        "passed": True,
        "step": args.steps,
        "resume_from": args.resume_from,
        "image_size": args.image_size,
        "changed_lora_tensors": changed,
        "optimizer_steps": current_steps,
        "previous_optimizer_steps": previous_steps,
        "gradient_norms": norms,
        "media": media,
    }
    (args.output / f"verification-step{args.steps}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
