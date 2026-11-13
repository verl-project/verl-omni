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

"""Create a caller-declared, byte-checked source inventory before model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

OWNERS = (
    "verl_omni/workers/engine/fsdp/diffusers_impl.py",
    "verl_omni/workers/utils/losses.py",
    "verl_omni/pipelines/sd3_flow_grpo/diffusers_training_adapter.py",
    "verl_omni/pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py",
    "verl_omni/pipelines/diffusion_rollout_output.py",
    "examples/flowgrpo_trainer/sd35/consistency/capture.py",
    "examples/flowgrpo_trainer/sd35/consistency/capture_scope.py",
    "examples/flowgrpo_trainer/sd35/consistency/model_contract.py",
    "examples/flowgrpo_trainer/sd35/consistency/source_manifest.py",
    "examples/flowgrpo_trainer/sd35/consistency/pin_producer.py",
    "examples/flowgrpo_trainer/sd35/consistency/run.sh",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head", help="explicit 40-character revision for deployed snapshots")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    head = args.head
    if head is None:
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        parser.error("head must be a lowercase 40-character Git revision")
    rows = []
    for name in OWNERS:
        candidate = source / name
        path = candidate.resolve(strict=True)
        if path != candidate or not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"missing or escaped owner: {name}")
        rows.append({"path": name, "sha256": sha(path.read_bytes()), "size": path.stat().st_size})
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("manifest output must be a fresh path with existing parent")
    data = (json.dumps({"head": head, "files": rows}, indent=2) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(data)
    print(sha(data))


if __name__ == "__main__":
    main()
