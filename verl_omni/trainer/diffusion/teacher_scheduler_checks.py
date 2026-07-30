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
"""Scheduler compatibility between the actor and a diffusion teacher (RFC #293).

The teacher replays states the student visited, so both sides must resolve the
same sigma grid. Checking that by enumerating config fields is not decidable --
the grid depends on ``shift``, ``num_train_timesteps``, ``use_dynamic_shifting``,
``base_shift``/``max_shift``, the image sequence length derived from pipeline
height/width, the Karras/exponential/beta sigma modes and more. The resolved
``timesteps``/``sigmas`` are decidable, so those are what get compared.

Two stages, because neither half is available at a single point in time: the
resolved grid needs both model configs complete (startup), while
``all_timesteps`` does not exist until rollout has run (per request).
"""

import torch
from diffusers import SchedulerMixin

from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config.diffusion import DiffusionModelConfig

__all__ = ["build_cpu_scheduler", "validate_scheduler_grids", "validate_request_timesteps"]


def build_cpu_scheduler(model_config: DiffusionModelConfig, adapter) -> SchedulerMixin:
    """Build a scheduler on CPU, deliberately not through ``adapter.build_scheduler``.

    That helper resolves the timesteps onto ``get_device_name()``, i.e. a CUDA
    device the driver may not have. The adapter's ``set_timesteps`` already takes
    a device, so validation composes the two steps itself.

    The class is SD3's, matching PR A's support matrix. An adapter-owned
    scheduler loader is what generalises this (Bagel constructs its scheduler
    bare, Wan through a module-level helper), and it is its own workstream.
    """
    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_config.local_path,
        subfolder="scheduler",
    )
    adapter.set_timesteps(scheduler, model_config, device="cpu")
    return scheduler


def _first_diff(actor_values: torch.Tensor, teacher_values: torch.Tensor) -> str:
    """Where the two grids first disagree, so a mismatch is diagnosable without a debugger."""
    shared = min(actor_values.numel(), teacher_values.numel())
    flat_actor, flat_teacher = actor_values.reshape(-1), teacher_values.reshape(-1)
    for index in range(shared):
        if flat_actor[index] != flat_teacher[index]:
            return f"index {index}: actor {flat_actor[index].item()} vs teacher {flat_teacher[index].item()}"
    return f"index {shared}: one grid is a prefix of the other"


def validate_scheduler_grids(
    actor_model_cfg: DiffusionModelConfig,
    teacher_model_cfg: DiffusionModelConfig,
    adapter,
    teacher_key: str,
) -> tuple[SchedulerMixin, SchedulerMixin]:
    """Stage 1, at startup: each side's own ``scheduler_config.json``, stepped with
    the same inherited pipeline parameters, must resolve to the same grid.

    Equality is exact, deliberately. ``index_for_timestep`` matches exactly, so a
    tolerance here would only admit grids that stage 2 rejects on the first
    request. Two schedulers built from identical configs through identical code
    produce bit-identical tensors; anything else is a real difference.

    Returns the constructed ``(actor_scheduler, teacher_scheduler)`` pair, so the
    caller can hand them straight to :func:`validate_request_timesteps`.
    """
    actor_scheduler = build_cpu_scheduler(actor_model_cfg, adapter)
    teacher_scheduler = build_cpu_scheduler(teacher_model_cfg, adapter)

    for name, actor_values, teacher_values in (
        ("timesteps", actor_scheduler.timesteps, teacher_scheduler.timesteps),
        ("sigmas", actor_scheduler.sigmas, teacher_scheduler.sigmas),
    ):
        if actor_values.shape != teacher_values.shape or not torch.equal(actor_values, teacher_values):
            raise ValueError(
                f"teacher {teacher_key!r}: resolved {name} differ from the actor's "
                f"(shapes {tuple(actor_values.shape)} vs {tuple(teacher_values.shape)}, "
                f"first mismatch at {_first_diff(actor_values, teacher_values)}); "
                f"checkpoint {teacher_model_cfg.path}"
            )

    return actor_scheduler, teacher_scheduler


def validate_request_timesteps(
    all_timesteps: torch.Tensor,
    actor_scheduler: SchedulerMixin,
    teacher_scheduler: SchedulerMixin,
    teacher_key: str,
) -> None:
    """Stage 2, per request: every timestep the rollout visited must land on the
    same sigma index for both sides.

    ``all_timesteps`` is ``[batch, steps]`` and rows may start at different
    windows, so the unique scalars are what matter, not the row layout.
    """
    for timestep in torch.unique(all_timesteps.detach().cpu().reshape(-1)):
        try:
            actor_index = actor_scheduler.index_for_timestep(timestep)
            teacher_index = teacher_scheduler.index_for_timestep(timestep)
        except (IndexError, RuntimeError) as exc:
            raise ValueError(
                f"teacher {teacher_key!r}: timestep {timestep.item()} is not on the shared scheduler grid"
            ) from exc
        if actor_index != teacher_index:
            raise ValueError(
                f"teacher {teacher_key!r}: timestep {timestep.item()} maps to sigma index "
                f"{actor_index} for the actor but {teacher_index} for the teacher"
            )
