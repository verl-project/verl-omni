from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class StagePlacementInfo:
    stage_id: Any
    stage_type: str | None
    num_replicas: int
    devices: str | None
    tensor_parallel_size: int | None


@dataclass(frozen=True)
class RolloutPlacementPreflight:
    outer_replicas: int
    visible_device_count: int | None
    stages: tuple[StagePlacementInfo, ...]

    @property
    def max_stage_replicas(self) -> int:
        return max((stage.num_replicas for stage in self.stages), default=1)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer value, got {value!r}") from exc


def _parse_stage_devices(devices: Any) -> list[int]:
    if devices is None:
        return []
    if isinstance(devices, (list, tuple)):
        raw_items = devices
    else:
        raw_items = str(devices).split(",")

    parsed: list[int] = []
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            # vLLM-Omni stage configs currently use plain comma-separated
            # GPU ids. Leave non-integer forms to the backend parser.
            return []
    return parsed


def load_stage_placement_info(stage_configs_path: str | Path | None) -> tuple[StagePlacementInfo, ...]:
    if not stage_configs_path:
        return ()

    path = Path(stage_configs_path)
    if not path.exists():
        raise FileNotFoundError(f"vLLM-Omni stage config does not exist: {path}")

    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config = _as_dict(raw_config)
    stage_args = config.get("stage_args") or []
    if not isinstance(stage_args, list):
        raise ValueError(f"vLLM-Omni stage config stage_args must be a list: {path}")

    stages: list[StagePlacementInfo] = []
    for index, raw_stage in enumerate(stage_args):
        stage = _as_dict(raw_stage)
        runtime = _as_dict(stage.get("runtime"))
        engine_args = _as_dict(stage.get("engine_args"))
        stages.append(
            StagePlacementInfo(
                stage_id=stage.get("stage_id", index),
                stage_type=stage.get("stage_type"),
                num_replicas=_as_int(runtime.get("num_replicas"), 1) or 1,
                devices=None if runtime.get("devices") is None else str(runtime.get("devices")),
                tensor_parallel_size=_as_int(engine_args.get("tensor_parallel_size"), None),
            )
        )
    return tuple(stages)


def estimate_outer_rollout_replicas(
    *,
    nnodes: int,
    gpus_per_node: int,
    tensor_model_parallel_size: int,
    data_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> int:
    total_gpus = max(int(nnodes), 1) * max(int(gpus_per_node), 1)
    rollout_world_size = (
        max(int(tensor_model_parallel_size), 1)
        * max(int(data_parallel_size), 1)
        * max(int(pipeline_model_parallel_size), 1)
    )
    return max(total_gpus // rollout_world_size, 1)


def validate_vllm_omni_rollout_placement(
    *,
    stage_configs_path: str | Path | None,
    outer_replicas: int,
    visible_device_count: int | None,
    allow_physical_stage_devices: bool = False,
) -> RolloutPlacementPreflight:
    stages = load_stage_placement_info(stage_configs_path)
    preflight = RolloutPlacementPreflight(
        outer_replicas=max(int(outer_replicas), 1),
        visible_device_count=visible_device_count,
        stages=stages,
    )

    duplicated_stage_dp = [stage for stage in stages if stage.num_replicas > 1]
    if preflight.outer_replicas > 1 and duplicated_stage_dp:
        details = ", ".join(
            f"stage_id={stage.stage_id} stage_type={stage.stage_type} num_replicas={stage.num_replicas}"
            for stage in duplicated_stage_dp
        )
        raise ValueError(
            "Invalid vLLM-Omni rollout placement: verl already created "
            f"{preflight.outer_replicas} outer rollout replicas, but the vLLM-Omni stage config also "
            f"requests inner runtime.num_replicas > 1 ({details}). Use exactly one DP owner: either "
            "verl outer rollout replicas or vLLM-Omni inner stage replicas, not both."
        )

    if visible_device_count is not None and visible_device_count > 0 and not allow_physical_stage_devices:
        invalid_device_stages = []
        for stage in stages:
            devices = _parse_stage_devices(stage.devices)
            if devices and (min(devices) < 0 or max(devices) >= visible_device_count):
                invalid_device_stages.append((stage, devices))
        if invalid_device_stages:
            details = ", ".join(
                f"stage_id={stage.stage_id} devices={devices} visible_device_count={visible_device_count}"
                for stage, devices in invalid_device_stages
            )
            raise ValueError(
                "Invalid vLLM-Omni stage device placement for this Ray actor: "
                f"{details}. Stage runtime.devices must be actor-local CUDA ids by default. "
                "Set VERL_OMNI_ALLOW_PHYSICAL_STAGE_DEVICES=1 only for a known physical-id launch path."
            )

    return preflight
