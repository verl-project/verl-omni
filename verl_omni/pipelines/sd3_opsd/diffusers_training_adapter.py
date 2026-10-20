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
"""SD3.5 training-side adapter for DiffusionOPSD (on-policy self-distillation)."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.sd3_dgpo.diffusers_training_adapter import StableDiffusion3DGPO

__all__ = ["StableDiffusion3OPSD"]


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="opsd")
class StableDiffusion3OPSD(StableDiffusion3DGPO):
    """Forward-process SD3 adapter used by DiffusionOPSD.

    OPSD supervises the trainable velocity at the behaviour-policy query state ``z_q``
    captured during rollout, so this adapter must **not** re-noise the rollout endpoint the
    way NFT/DGPO do: the batch already carries ``opsd_query_latents`` (the state) and
    ``train_timesteps`` (its scheduler timestep, i.e. ``sigma_q * num_train_timesteps``).

    The fitting forward pass itself is the plain conditional velocity used by DGPO (no
    train-time CFG); the anchor ``y0 = z_q - sigma_q * v_old(z_q)`` and the detached positive
    and negative targets are consumed by the loss, not built here.
    """

    @classmethod
    def configure_engine(cls, engine, engine_config) -> None:
        """Give the OPSD engine the ``diffusion_loss`` block it builds targets from.

        The engine only receives ``DiffusionModelConfig`` and ``EngineConfig``, so the loss
        configuration reaches it through this adapter hook instead of a branch in the shared
        worker.
        """
        algorithm_config = getattr(engine_config, "algorithm_config", None)
        if algorithm_config is None:
            raise ValueError(
                "DiffusionOPSD engine requires the actor's diffusion_loss configuration; "
                "engine_config.algorithm_config was not set."
            )
        engine.configure_opsd_runtime(loss_config=algorithm_config)

    @classmethod
    def prepare_training_inputs(
        cls, micro_batch: TensorDict, step: int, scheduler
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return the stored query state and its timestep instead of a re-noised endpoint."""
        train_timesteps = micro_batch.get("train_timesteps", None)
        if train_timesteps is None:
            raise KeyError("DiffusionOPSD actor batch requires `train_timesteps` from prepare_actor_batch.")
        if train_timesteps.ndim != 2:
            raise ValueError(
                f"DiffusionOPSD `train_timesteps` must have shape [B, T], got {tuple(train_timesteps.shape)}."
            )

        query_latents = micro_batch.get("opsd_query_latents", None)
        if query_latents is None:
            raise KeyError("DiffusionOPSD actor batch requires `opsd_query_latents` from its rollout adapter.")

        timestep = train_timesteps[:, step]
        if query_latents.ndim == 5:  # (B, T, C, H, W): engine loop selects the query slot
            query_latents = query_latents[:, step]
        elif query_latents.ndim != 4:
            raise ValueError(
                "DiffusionOPSD `opsd_query_latents` must have shape [B, C, H, W] or [B, T, C, H, W], "
                f"got {tuple(query_latents.shape)}."
            )
        if query_latents.shape[0] != timestep.shape[0]:
            raise ValueError(
                "DiffusionOPSD query latents and timesteps disagree on batch size: "
                f"{query_latents.shape[0]} vs {timestep.shape[0]}."
            )

        scale = getattr(scheduler.config, "num_train_timesteps", None)
        if scale is None:
            raise ValueError(
                "Converting DiffusionOPSD `train_timesteps` to flow time requires "
                f"scheduler.config.num_train_timesteps, but {type(scheduler).__name__} does not define it."
            )
        tu.assign_non_tensor(micro_batch, num_train_timesteps=int(scale))

        time = timestep.float() / float(scale)
        t_expanded = time.view(-1, *([1] * (query_latents.ndim - 1)))
        loss_context = {"xt": query_latents, "t_expanded": t_expanded}
        return query_latents, timestep, loss_context

    @staticmethod
    def prepare_loss_inputs(micro_batch: TensorDict, step: int) -> dict[str, Any]:
        """Pass the detached OPSD targets to the loss on top of the shared per-step fields."""
        inputs = DiffusionModelBase.prepare_loss_inputs(micro_batch, step)
        for key in ("opsd_target_pos", "opsd_target_neg"):
            value = micro_batch.get(key, None)
            if value is not None:
                inputs[key] = value
        return inputs
