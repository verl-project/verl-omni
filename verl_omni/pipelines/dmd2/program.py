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

"""Continuous-flow DMD2 computation using caller-owned velocity predictors.

This module constructs inputs for the registered DMDLoss. Model construction,
parameter updates, distributed contexts and checkpoint publication stay with
the caller. No model adapter or training engine is imported here.
"""

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.distillation.utils import standard_cfg, timestep_shift, velocity_to_x0
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig

__all__ = ["DMD2Program", "backward_simulation", "re_noise", "shifted_sigmas"]


def shifted_sigmas(time: float | Tensor, shifts: dict[str, float], device: torch.device) -> dict[str, Tensor]:
    """Apply each modality's rational shift on the normalized [0, 1] grid in FP32."""
    time = torch.as_tensor(time, dtype=torch.float32, device=device)
    return {name: timestep_shift(time, num_train_timesteps=1, shift=shift) for name, shift in shifts.items()}


def re_noise(clean: dict[str, Tensor], sigmas: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return detached noisy samples and Gaussian noise, preserving ``(B, ...)`` shapes."""
    noise = {name: torch.randn_like(value, dtype=torch.float32) for name, value in clean.items()}
    noisy = {
        name: (1 - sigmas[name]) * value.detach().float() + sigmas[name] * noise[name] for name, value in clean.items()
    }
    return noisy, noise


def backward_simulation(
    predict: Callable[[dict[str, Tensor], dict[str, Tensor]], dict[str, Tensor]],
    initial: dict[str, Tensor],
    shifts: dict[str, float],
    steps: int,
    exit_step: int,
    *,
    require_grad: bool,
) -> dict[str, Tensor]:
    """Predict clean FP32 samples, retaining gradients only at the selected exit.

    ``initial`` contains standard Gaussian noise of shape ``(B, ...)`` per
    modality. ``predict`` returns flow velocities with the same keys and shapes.
    The caller validates model-specific layouts before entering this sampler.
    """
    if not 0 <= exit_step < steps:
        raise ValueError("exit_step must index an inference step")
    latents = {name: value.detach().float() for name, value in initial.items()}
    device = next(iter(latents.values())).device
    for index in range(exit_step + 1):
        sigmas = shifted_sigmas(1 - index / steps, shifts, device)
        context = nullcontext() if require_grad and index == exit_step else torch.no_grad()
        with context:
            velocities = predict(latents, sigmas)
            clean = {name: velocity_to_x0(value, velocities[name], sigmas[name]) for name, value in latents.items()}
        if index != exit_step:
            next_sigmas = shifted_sigmas(1 - (index + 1) / steps, shifts, device)
            latents, _ = re_noise(clean, next_sigmas)
    return clean


class DMD2Program:
    """Compute DMD2 stages with a model-independent velocity callback.

    Args:
        config: Continuous score sampling with explicit modality shifts/weights.
            ``score_time_min/max`` bound the base time before modality shifts.
            The scalar shift and ``score_sigma_min/max`` fields are unused here
            and must retain their defaults.
        predict: ``predict(role, latents, sigmas, negative)`` returning matching
            velocity tensors. It captures conditions and caller-owned models.
            Roles are ``student``, ``fake_score`` and ``teacher``; ``negative``
            selects unconditional teacher inputs for CFG. With teacher
            ``layer_norm`` CFG, both inputs and outputs must be packed
            ``(B, tokens, features)``; the last axis is the feature vector.
        seed: Seed for the exit-selection RNG, shared across cooperating ranks.

    The caller saves the global CPU RNG (score time) and each used device RNG
    (tensor noise) alongside ``state_dict``. The seed controls only exit selection.
    Forward/recompute contexts and gradient reduction remain caller-owned.
    """

    def __init__(
        self,
        config: DiffusionDMDConfig,
        predict: Callable[[str, dict[str, Tensor], dict[str, Tensor], bool], dict[str, Tensor]],
        seed: int,
    ):
        if config.score_discrete_steps != 0:
            raise ValueError("DMD2Program supports continuous score sampling only; set score_discrete_steps=0")
        if not config.modality_shifts:
            raise ValueError("DMD2Program requires explicit modality_shifts and modality_weights")
        for name in ("rollout_timestep_shift", "score_timestep_shift"):
            if getattr(config, name) != getattr(DiffusionDMDConfig, name):
                raise ValueError(f"DMD2Program uses modality_shifts; leave legacy {name} at its default")
        for name in ("score_sigma_min", "score_sigma_max"):
            if getattr(config, name) != getattr(DiffusionDMDConfig, name):
                raise ValueError(f"DMD2Program uses pre-shift score_time_min/max; leave {name} at its default")
        self.config = config
        self.predict = predict
        self.control_rng = torch.Generator().manual_seed(seed)

    def training_stages(self) -> tuple[str, ...]:
        """Return the fake-first attempt schedule; this method performs no updates."""
        return ("fake_score",) * self.config.fake_update_ratio + ("student",)

    def state_dict(self) -> dict[str, Tensor]:
        """Capture exit selection; the caller separately saves CPU and device RNGs."""
        return {"control_rng": self.control_rng.get_state()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Restore exit selection without modifying the global CPU or device RNGs."""
        self.control_rng.set_state(state["control_rng"])

    def _predict(
        self, role: str, latents: dict[str, Tensor], sigmas: dict[str, Tensor], negative: bool = False
    ) -> dict[str, Tensor]:
        predictions = self.predict(role, latents, sigmas, negative)
        if predictions.keys() != latents.keys():
            raise ValueError("Velocity predictions must have the same modality keys as the latents")
        if any(predictions[name].shape != value.shape for name, value in latents.items()):
            raise ValueError("Velocity predictions must have the same shapes as the latents")
        return predictions

    def _sample(self, initial: dict[str, Tensor], *, require_grad: bool, final: bool = False) -> dict[str, Tensor]:
        if initial.keys() != self.config.modality_shifts.keys():
            raise ValueError("Latent modalities must match modality_shifts and modality_weights")
        first = next(iter(initial.values()))
        if any(
            value.ndim < 2
            or value.numel() == 0
            or not value.is_floating_point()
            or value.shape[0] != first.shape[0]
            or value.device != first.device
            for value in initial.values()
        ):
            raise ValueError("Latents require nonempty floating (B, ...) tensors with a shared batch size and device")
        exit_step = (
            self.config.num_inference_steps - 1
            if final
            else int(torch.randint(self.config.num_inference_steps, (), generator=self.control_rng))
        )
        return backward_simulation(
            lambda latents, sigmas: self._predict("student", latents, sigmas),
            initial,
            self.config.modality_shifts,
            self.config.num_inference_steps,
            exit_step,
            require_grad=require_grad,
        )

    def stage_outputs(self, initial: dict[str, Tensor], stage: str) -> dict[str, dict[str, Tensor]]:
        """Construct loss inputs using one shared base score time per microbatch.

        Each modality transforms that scalar with its own shift. This intentionally
        shares time across samples, unlike per-sample score-time sampling.
        """
        if stage not in {"student", "fake_score"}:
            raise ValueError(f"Unknown DMD stage {stage!r}")
        if stage == "student" and self.config.teacher_guidance_scale != 1 and self.config.cfg_norm == "layer_norm":
            if any(value.ndim != 3 for value in initial.values()):
                raise ValueError(
                    "Teacher layer_norm CFG requires packed (B, tokens, features) latents and velocities; "
                    "pack the latents before calling the program or use cfg_norm='none'/'scalar'"
                )
        generated = self._sample(initial, require_grad=stage == "student")
        time = torch.empty((), dtype=torch.float32, device="cpu").uniform_(
            self.config.score_time_min, self.config.score_time_max
        )
        sigmas = shifted_sigmas(time, self.config.modality_shifts, next(iter(initial.values())).device)
        noisy, noise = re_noise(generated, sigmas)
        if stage == "fake_score":
            prediction = self._predict(stage, noisy, sigmas)
            return {
                name: {"generated_x0": value.detach(), "noise_pred": prediction[name], "noise": noise[name]}
                for name, value in generated.items()
            }
        with torch.no_grad():
            fake = self._predict("fake_score", noisy, sigmas)
            real = self._predict("teacher", noisy, sigmas)
            if self.config.teacher_guidance_scale != 1:
                negative = self._predict("teacher", noisy, sigmas, negative=True)
                real = {
                    name: standard_cfg(value, negative[name], self.config.teacher_guidance_scale, self.config.cfg_norm)
                    for name, value in real.items()
                }
            fake = {name: velocity_to_x0(noisy[name], value, sigmas[name]) for name, value in fake.items()}
            real = {name: velocity_to_x0(noisy[name], value, sigmas[name]) for name, value in real.items()}
        return {
            name: {"generated_x0": value, "fake_x0": fake[name], "teacher_x0": real[name]}
            for name, value in generated.items()
        }

    def compute_loss(
        self,
        initial: dict[str, Tensor],
        stage: str,
        data: TensorDict,
        loss_function: Callable,
        *,
        dp_group: Any = None,
    ) -> tuple[Tensor, dict[str, Any], bool]:
        """Call the registered loss per modality; report nonfinite tensors to the caller.

        ``loss_function`` is the public diffusion loss wrapper, bound to its actor
        config. The caller supplies ``gradient_accumulation_steps`` and ``sp_size``
        in ``data``; neither is overwritten or applied a second time here.
        """
        for name in ("gradient_accumulation_steps", "sp_size"):
            value = tu.get_non_tensor_data(data, name, default=None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Caller must supply a positive integer {name} for the public loss wrapper")
        tu.assign_non_tensor(data, dmd_stage=stage, dmd_normalization_epsilon=self.config.normalization_epsilon)
        outputs = self.stage_outputs(initial, stage)
        loss, metrics, invalid = 0, {}, False
        for modality, output in outputs.items():
            part, part_metrics = loss_function(model_output=output, data=data, dp_group=dp_group)
            loss = loss + self.config.modality_weights[modality] * part
            metrics.update({f"{modality}/{key}": value for key, value in part_metrics.items()})
            invalid |= any(not bool(torch.isfinite(value).all()) for value in output.values())
        return loss, metrics, invalid or not bool(torch.isfinite(loss.detach()))

    @torch.no_grad()
    def sample(self, initial: dict[str, Tensor]) -> dict[str, Tensor]:
        """Generate final FP32 modality latents without advancing exit-selection RNG."""
        return self._sample(initial, require_grad=False, final=True)
