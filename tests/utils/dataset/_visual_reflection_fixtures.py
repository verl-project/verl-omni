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
"""Generic local-image fixtures for visual-reflection dataset tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from verl_omni.utils.dataset.visual_reflection.contracts import (
    PipelineVariant,
    VisualReflectionTrajectory,
    derive_trajectory_id,
    validate_trajectory,
)
from verl_omni.utils.dataset.visual_reflection.images import LocalImageResolver


def write_rgb_png(path: Path, color: tuple[int, int, int]) -> None:
    """Write a deterministic one-pixel RGB PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color).save(path, format="PNG")


def make_visual_reflection_trajectory(
    root: Path,
    *,
    edit_turns: int,
    source_record_id: str = "fixture",
    manifest_id: str = "manifest-test",
    source_dataset: str = "example/visual-reflection",
    pipeline_variant: PipelineVariant = "direct_parse",
) -> VisualReflectionTrajectory:
    """Create a valid generic trajectory backed by deterministic local images."""
    image_paths = [Path("images") / f"state_{index}.png" for index in range(edit_turns + 1)]
    for index, image_path in enumerate(image_paths):
        write_rgb_png(root / image_path, (index * 70 % 255, index * 40 % 255, index * 20 % 255))
    resolver = LocalImageResolver(root)
    images = [
        resolver(
            image_path.as_posix(),
            field="images",
            index=index,
            source_record_id=source_record_id,
        )
        for index, image_path in enumerate(image_paths)
    ]
    steps = [
        {
            "reflection": f"Reflection for state {index}.",
            "action": "continue",
            "edit": f"Apply delta edit {index}.",
        }
        for index in range(edit_turns)
    ]
    steps.append({"reflection": "The image is complete.", "action": "stop", "edit": ""})
    prompt = "A precise visual prompt."
    trajectory: VisualReflectionTrajectory = {
        "trajectory_id": derive_trajectory_id(
            manifest_id=manifest_id,
            source_dataset=source_dataset,
            source_record_id=source_record_id,
            pipeline_variant=pipeline_variant,
            prompt=prompt,
            images=images,
            steps=steps,
        ),
        "manifest_id": manifest_id,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "pipeline_variant": pipeline_variant,
        "prompt": prompt,
        "images": images,
        "steps": steps,
    }
    return validate_trajectory(trajectory)
