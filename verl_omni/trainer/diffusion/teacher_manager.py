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
"""Trainer-facing boundary for the diffusion teacher (RFC #293).

The trainer sees one call and must not branch on placement, checkpoint form or
backend:

    batch = batch.union(self.teacher_manager.compute_teacher_outputs(batch))

Scoring is batched once per step after rollout completes -- the teacher
evaluates states the student visited, and a diffusion request carries large
latent tensors plus scheduler conditioning, so there is no per-timestep RPC.

Config and internal state are teacher-keyed from day one; the MVP rejects more
than one teacher, so multi-teacher routing lands without a config migration.
"""

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.teacher_scheduler_checks import (
    validate_request_timesteps,
    validate_scheduler_grids,
)
from verl_omni.workers.config.diffusion import DiffusionModelConfig, DiffusionTeacherConfig
from verl_omni.workers.config.diffusion.teacher import resolve_teacher_model_config
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

__all__ = ["DiffusionTeacherModelManager", "MVP_SUPPORT_MATRIX"]

# (architecture, algorithm) pairs PR A serves. The request contract requires
# pooled_prompt_embeds and scheduler validation loads a checkpoint scheduler
# directory; both are SD3-specific, and generalising needs adapter-owned hooks.
MVP_SUPPORT_MATRIX = (("StableDiffusion3Pipeline", "flow_grpo"),)

REQUIRED_REQUEST_KEYS = ("all_latents", "all_timesteps")


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

        try:
            output = self._teacher_wg.compute_teacher_outputs(batch_td)
        except Exception as exc:
            raise RuntimeError(
                f"Teacher {self.teacher_key!r}: scoring failed on a batch of {len(batch)} rows from "
                f"checkpoint {self.teacher_model_config.path}."
            ) from exc

        if output is None:
            raise ValueError(
                f"Teacher {self.teacher_key!r}: worker group returned no output for a batch of "
                f"{len(batch)} rows. The collect ranks assemble the response before it reaches here, "
                "so this is a broken teacher group rather than a non-collect rank."
            )

        teacher_mean = tu.get(output, "teacher_prev_sample_mean", default=None)
        if teacher_mean is None:
            raise ValueError(
                f"Teacher {self.teacher_key!r}: response has no 'teacher_prev_sample_mean'; "
                f"got keys {sorted(output.keys())}."
            )
        if teacher_mean.shape[0] != len(batch):
            raise ValueError(
                f"Teacher {self.teacher_key!r}: response covers {teacher_mean.shape[0]} rows but the "
                f"request carried {len(batch)}."
            )
        return DataProto.from_tensordict(tu.get_tensordict({"teacher_prev_sample_mean": teacher_mean}))

    def teacher_param_checksums(self) -> list[str]:
        """One checksum per rank, for the frozen-teacher probes (§6.3)."""
        return self._teacher_wg.teacher_param_checksum()

    def start_profile(self, step: int) -> None:
        self._teacher_wg.start_profile(profile_step=step)

    def stop_profile(self) -> None:
        self._teacher_wg.stop_profile()
