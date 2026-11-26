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

"""Write the eight original producer-byte pins outside their capture root."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.resolve().is_relative_to(root):
        raise ValueError("pin output must be fresh and outside capture root")
    names = ["rollout/manifest.json", "rollout/actor_batch.pt", "actor/comparison.json", "actor/comparison.pt"]
    for role in ("rollout", "actor"):
        records = list((root / role / "captures" / role).glob("*/record.json"))
        if len(records) != 1:
            raise ValueError(f"expected exactly one {role} record")
        names += [
            str(records[0].relative_to(root)),
            str((records[0].parent / "dump/step0/rank0/dump.json").relative_to(root)),
        ]
    lines = []
    for name in names:
        path = root / name
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError(f"invalid producer path: {name}")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n")
    data = "".join(lines).encode()
    with output.open("xb") as stream:
        stream.write(data)
    print(hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
