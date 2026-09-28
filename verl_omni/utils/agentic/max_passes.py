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

"""Shared ``agentic_image_gen.max_generate_image_passes`` reader (tool + agent loop)."""

from __future__ import annotations

from verl_omni.tools.trajectory.hydra_env import agentic_get


def max_generate_passes() -> int:
    """Return Hydra ``max_generate_image_passes``.

    Returns:
        Integer ``>= 1`` from the bound config or yaml default.

    Raises:
        RuntimeError: If ``agentic_image_gen`` is unbound.
        ValueError: If the bound value is not an integer ``>= 1``.
    """
    raw = agentic_get("max_generate_image_passes")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic_image_gen.max_generate_image_passes must be an integer >= 1, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"agentic_image_gen.max_generate_image_passes must be >= 1, got {value}")
    return value
