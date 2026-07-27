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
"""Small UniCoT-shaped image fixtures shared by PR1 and later PR2 tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def write_rgb_png(path: Path, color: tuple[int, int, int]) -> None:
    """Write a deterministic one-pixel RGB PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color).save(path, format="PNG")


def make_unicot_row(root: Path, *, edit_turns: int, data_id: str = "fixture") -> dict[str, Any]:
    """Create a valid zero-, one-, or two-edit row using the public field shape."""
    state_count = edit_turns + 1
    input_paths = [(Path("images") / "inputs" / f"state_{index}.png").as_posix() for index in range(state_count)]
    for index in range(state_count):
        write_rgb_png(root / input_paths[index], (index * 70 % 255, index * 40 % 255, index * 20 % 255))
    output_paths: list[str | None] = []
    for index in range(edit_turns):
        output_path = Path("images") / "outputs" / f"state_{index + 1}.png"
        (root / output_path).parent.mkdir(parents=True, exist_ok=True)
        (root / output_path).write_bytes((root / input_paths[index + 1]).read_bytes())
        output_paths.append(output_path.as_posix())
    output_paths.append(None)

    evaluations = [f"Detailed evaluation for state {index}." for index in range(state_count)]
    summaries = [f"Summary for state {index}." for index in range(state_count)]
    if state_count > 1:
        summaries[1] = ""
        evaluations[1] = "  Detailed\n evaluation   fallback for state 1.  "
    edits = [f"Apply delta edit {index}." for index in range(edit_turns)]
    edits.append("Everything is good. No editing needed.")
    return {
        "data_id": data_id,
        "prompt": "A precise visual prompt.",
        "eval": evaluations,
        "eval_summary": summaries,
        "edit": edits,
        "input_image": input_paths,
        "output_image": output_paths,
    }
