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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class StagePlacement:
    stage_id: Any
    num_replicas: int
    devices: str | None
    tensor_parallel_size: int | None


@dataclass(frozen=True)
class RolloutPlacement:
    visible_device_count: int | None
    stages: tuple[StagePlacement, ...]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer, got {value!r}") from exc


def _parse_devices(devices: Any) -> list[int]:
    if devices is None:
        return []
    raw_items = devices if isinstance(devices, list | tuple) else str(devices).split(",")
    try:
        return [int(str(item).strip()) for item in raw_items if str(item).strip()]
    except ValueError:
        return []


def load_stage_placements(config_path: str | Path | None) -> tuple[StagePlacement, ...]:
    if not config_path:
        return ()

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"vLLM-Omni rollout config does not exist: {path}")

    config = _as_dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    raw_stages = config.get("stage_args")
    legacy_format = raw_stages is not None
    if raw_stages is None:
        raw_stages = config.get("stages", [])
    if not isinstance(raw_stages, list):
        raise ValueError(f"vLLM-Omni rollout stages must be a list: {path}")

    stages = []
    for index, raw_stage in enumerate(raw_stages):
        stage = _as_dict(raw_stage)
        runtime = _as_dict(stage.get("runtime")) if legacy_format else stage
        engine_args = _as_dict(stage.get("engine_args")) if legacy_format else stage
        stages.append(
            StagePlacement(
                stage_id=stage.get("stage_id", index),
                num_replicas=_as_int(runtime.get("num_replicas"), 1) or 1,
                devices=None if runtime.get("devices") is None else str(runtime.get("devices")),
                tensor_parallel_size=_as_int(engine_args.get("tensor_parallel_size")),
            )
        )
    return tuple(stages)


def validate_vllm_omni_rollout_placement(
    *,
    config_path: str | Path | None,
    visible_device_count: int | None,
) -> RolloutPlacement:
    stages = load_stage_placements(config_path)
    placement = RolloutPlacement(visible_device_count=visible_device_count, stages=stages)

    replicated_stages = [stage for stage in stages if stage.num_replicas != 1]
    if replicated_stages:
        details = ", ".join(
            f"stage_id={stage.stage_id} num_replicas={stage.num_replicas}" for stage in replicated_stages
        )
        raise ValueError(
            "vLLM-Omni AR rollout uses verl-owned outer replicas and requires "
            f"exactly one inner stage replica ({details})"
        )

    if visible_device_count is not None and visible_device_count > 0:
        invalid_stages = []
        for stage in stages:
            devices = _parse_devices(stage.devices)
            if devices and (min(devices) < 0 or max(devices) >= visible_device_count):
                invalid_stages.append((stage.stage_id, devices))
        if invalid_stages:
            details = ", ".join(f"stage_id={stage_id} devices={devices}" for stage_id, devices in invalid_stages)
            raise ValueError(
                "vLLM-Omni stage devices must use Ray actor-local CUDA ids: "
                f"{details}, visible_device_count={visible_device_count}"
            )

    return placement
