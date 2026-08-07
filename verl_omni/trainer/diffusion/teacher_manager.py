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
"""Trainer-facing boundary for the diffusion teacher.

The trainer sees one call and must not branch on placement, checkpoint form or
backend:

    batch = batch.union(self.teacher_manager.compute_teacher_outputs(batch))

Scoring is batched once per step after rollout completes -- the teacher
evaluates states the student visited, and a diffusion request carries large
latent tensors plus scheduler conditioning, so there is no per-timestep RPC.

Config and internal state are teacher-keyed from day one; the MVP rejects more
than one teacher, so multi-teacher routing lands without a config migration.
"""

import torch
from diffusers import SchedulerMixin
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config.diffusion import DiffusionModelConfig, DiffusionTeacherConfig
from verl_omni.workers.config.diffusion.teacher import resolve_teacher_model_config
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

__all__ = [
    "DiffusionTeacherModelManager",
    "MVP_SUPPORT_MATRIX",
    "build_cpu_scheduler",
    "validate_scheduler_grids",
    "validate_request_timesteps",
]

# (architecture, algorithm) pairs this runtime serves. The request contract requires
# pooled_prompt_embeds and scheduler validation loads a checkpoint scheduler
# directory; both are SD3-specific, and generalising needs adapter-owned hooks.
MVP_SUPPORT_MATRIX = (("StableDiffusion3Pipeline", "flow_grpo"),)

REQUIRED_REQUEST_KEYS = ("all_latents", "all_timesteps")


# --- Scheduler-grid compatibility (actor vs teacher) ---------------------------
# The teacher replays states the student visited, so both sides must resolve the
# same sigma grid. That is not decidable by enumerating config fields (the grid
# depends on shift, num_train_timesteps, dynamic shifting, image sequence length,
# the sigma mode and more), but the resolved timesteps/sigmas are, so those get
# compared. Two stages, because neither half exists at one point in time: the
# resolved grid needs both model configs complete (startup), while all_timesteps
# only exists after rollout has run (per request).


def build_cpu_scheduler(model_config: DiffusionModelConfig, adapter) -> SchedulerMixin:
    """Build a scheduler on CPU, deliberately not through ``adapter.build_scheduler``.

    That helper resolves the timesteps onto ``get_device_name()``, i.e. a CUDA
    device the driver may not have. The adapter's ``set_timesteps`` already takes
    a device, so validation composes the two steps itself.
    """
    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_config.local_path,
        subfolder="scheduler",
    )
    adapter.set_timesteps(scheduler, model_config, device="cpu")
    return scheduler


def _first_diff(actor_values: torch.Tensor, teacher_values: torch.Tensor) -> str:
    """Where the two grids first disagree, so a mismatch is diagnosable without a debugger."""
    flat_actor, flat_teacher = actor_values.reshape(-1), teacher_values.reshape(-1)
    shared = min(flat_actor.numel(), flat_teacher.numel())
    diff = (flat_actor[:shared] != flat_teacher[:shared]).nonzero()
    if diff.numel() == 0:
        return f"index {shared}: one grid is a prefix of the other"
    index = int(diff[0].item())
    return f"index {index}: actor {flat_actor[index].item()} vs teacher {flat_teacher[index].item()}"


def validate_scheduler_grids(
    actor_model_cfg: DiffusionModelConfig,
    teacher_model_cfg: DiffusionModelConfig,
    adapter,
    teacher_key: str,
) -> tuple[SchedulerMixin, SchedulerMixin]:
    """Stage 1, at startup: each side's own ``scheduler_config.json``, stepped with
    the same inherited pipeline parameters, must resolve to the same grid.

    Equality is exact: two schedulers built from identical configs through
    identical code produce bit-identical tensors, and ``index_for_timestep``
    matches exactly, so a tolerance here would only admit grids that the
    per-request stage rejects anyway. Returns the ``(actor, teacher)`` pair for
    :func:`validate_request_timesteps`.
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
    shared scheduler grid.

    ``all_timesteps`` is ``[batch, steps]`` and rows may start at different
    windows, so the unique scalars are what matter, not the row layout. Stage 1
    already proved the two grids identical, so landing on both is the check.
    """
    for timestep in torch.unique(all_timesteps.detach().cpu().reshape(-1)):
        try:
            actor_scheduler.index_for_timestep(timestep)
            teacher_scheduler.index_for_timestep(timestep)
        except (IndexError, RuntimeError) as exc:
            raise ValueError(
                f"teacher {teacher_key!r}: timestep {timestep.item()} is not on the shared scheduler grid"
            ) from exc


class DiffusionTeacherModelManager:
    """Owns teacher worker-group access, request/response handling and validation.

    Construction runs the checks that need constructed configs: single teacher,
    a checkpoint-derived architecture inside the MVP matrix and compatible with
    the actor's, and the resolved scheduler grid.
    """

    def __init__(
        self,
        teacher_config: DiffusionTeacherConfig,
        teacher_wg,
        actor_model_config: DiffusionModelConfig,
        adapter,
    ):
        if len(teacher_config.models) != 1:
            raise NotImplementedError(
                f"The diffusion teacher runtime supports exactly one teacher, got {sorted(teacher_config.models)}."
            )
        self.teacher_key, entry = next(iter(teacher_config.models.items()))
        self._teacher_wg = teacher_wg
        self.actor_model_config = actor_model_config
        self.teacher_model_config = resolve_teacher_model_config(actor_model_config, entry)

        self._validate_support_matrix()
        self.actor_scheduler, self.teacher_scheduler = validate_scheduler_grids(
            actor_model_config, self.teacher_model_config, adapter, self.teacher_key
        )

    def _validate_support_matrix(self) -> None:
        if self.teacher_model_config.architecture != self.actor_model_config.architecture:
            raise ValueError(
                f"Teacher {self.teacher_key!r}: checkpoint architecture "
                f"{self.teacher_model_config.architecture!r} differs from the actor's "
                f"{self.actor_model_config.architecture!r}. The MVP requires teacher and student to share "
                "the pipeline adapter, latent layout and conditioning contract."
            )
        pair = (self.teacher_model_config.architecture, self.teacher_model_config.algorithm)
        if pair not in MVP_SUPPORT_MATRIX:
            raise ValueError(
                f"Teacher {self.teacher_key!r}: (architecture, algorithm) {pair} is outside the supported "
                f"matrix {list(MVP_SUPPORT_MATRIX)}. Other pipelines need an adapter-owned scheduler loader "
                "and teacher-request schema, which is a separate workstream."
            )

    def compute_teacher_outputs(self, batch: DataProto) -> DataProto:
        """Replay the batch under the teacher and return the teacher_* keys.

        Tensor-level validation (key presence, ``[batch, steps]`` prefix, dtype,
        device, finiteness) happens in the worker, where the failing shard is
        visible. What is only decidable here is the request contract, the shared
        scheduler grid, and that the assembled response covers the whole batch.
        """
        batch_td = batch.to_tensordict()
        missing = [key for key in REQUIRED_REQUEST_KEYS if key not in batch_td.keys()]
        if missing:
            raise ValueError(
                f"Teacher {self.teacher_key!r}: request is missing {missing}; the teacher replays the "
                "student's own trajectory, so rollout must supply the visited latents and timesteps."
            )

        validate_request_timesteps(
            tu.get(batch_td, "all_timesteps"), self.actor_scheduler, self.teacher_scheduler, self.teacher_key
        )

        batch_td = embeds_padding_2_no_padding(batch_td)
        tu.assign_non_tensor(
            batch_td,
            compute_loss=False,
            height=self.actor_model_config.pipeline.height,
            width=self.actor_model_config.pipeline.width,
            vae_scale_factor=self.actor_model_config.get("vae_scale_factor", 8),
        )

        output = self._teacher_wg.compute_teacher_outputs(batch_td)
        teacher_mean = tu.get(output, "teacher_prev_sample_mean")
        if teacher_mean.shape[0] != len(batch):
            raise ValueError(
                f"Teacher {self.teacher_key!r}: response covers {teacher_mean.shape[0]} rows but the "
                f"request carried {len(batch)}."
            )
        return DataProto.from_tensordict(tu.get_tensordict({"teacher_prev_sample_mean": teacher_mean}))

    def start_profile(self, step: int) -> None:
        self._teacher_wg.start_profile(profile_step=step)

    def stop_profile(self) -> None:
        self._teacher_wg.stop_profile()
