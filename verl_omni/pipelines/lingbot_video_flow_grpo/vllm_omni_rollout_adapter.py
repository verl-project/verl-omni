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

"""Native vLLM-Omni rollout pipeline for LingBot Dense T2V FlowGRPO."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import nullcontext

import torch
from diffusers import AutoencoderKLWan
from verl.utils.device import get_device_name
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    PROMPT_TEMPLATE,
    apply_cfg,
    shifted_sigmas,
    validate_t2v_dimensions,
)
from .manual_lora import ManualLinearLoRAManager

__all__ = ["LingBotVideoPipelineWithLogProb"]


def _coalesce(value, default):
    return default if value is None else value


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _autocast(device: torch.device, dtype: torch.dtype):
    device_type = get_device_name()
    if device.type == device_type and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type, dtype=dtype)
    return nullcontext()


@VllmOmniPipelineBase.register("LingBotVideoPipeline", algorithm="flow_grpo")
class LingBotVideoPipelineWithLogProb(torch.nn.Module):
    """Single-request LingBot Dense rollout with an SDE trajectory.

    The native in-tree ``LingBotVideoPipeline`` drives a deterministic UniPC
    solver and returns no replay trajectory, so this adapter reuses vLLM-Omni's
    in-tree LingBot transformer (#5035) but owns the SDE denoising loop, which
    lets it return old-policy log-probabilities to FlowGRPO.  MoE, TI2V and the
    refiner are deliberately rejected by this adapter's dense-only model check.
    """

    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        try:
            # Rollout drives vLLM-Omni's in-tree LingBot transformer (#5035) so it
            # reuses the engine's optimized kernels; training/replay uses the pip
            # `lingbot_video` transformer under FSDP.  The forward signature and
            # `from_pretrained(subfolder="transformer")` loader are identical
            # across the two implementations, so only the import differs here.
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import (
                LingBotVideoTransformer3DModel,
            )
        except ImportError as exc:
            raise ImportError(
                "LingBot rollout requires vLLM-Omni with the in-tree LingBot transformer "
                "(PR #5035, on main since 2026-07-21)."
            ) from exc

        self.od_config = od_config
        self.device = get_local_device()
        self._interrupt = False
        model_path = od_config.model
        local_files_only = os.path.exists(model_path)
        dtype = od_config.dtype

        self.transformer = LingBotVideoTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        num_experts = int(getattr(self.transformer.config, "num_experts", 0))
        if num_experts != 0:
            raise ValueError(
                "LingBotVideoPipelineWithLogProb supports only the Dense checkpoint "
                f"(transformer.config.num_experts == 0), got {num_experts}."
            )

        self.vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                subfolder="processor",
                local_files_only=local_files_only,
            )
        except OSError:
            # Some converted checkpoints keep the processor at the model root.
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)

        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.weights_sources = ()
        self._manual_lora = ManualLinearLoRAManager(self.transformer)
        self._prompt_template_encode_start_idx = self._compute_template_prefix_length()

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value: bool) -> None:
        self._interrupt = bool(value)

    def _compute_template_prefix_length(self) -> int:
        marker = "<|USER_INPUT_MARKER|>"
        rendered = PROMPT_TEMPLATE.format(marker)
        marker_pos = rendered.find(marker)
        if marker_pos < 0:
            return 0
        inputs = self.processor(text=rendered[:marker_pos], images=None, videos=None, return_tensors="pt")
        return int(inputs["input_ids"].shape[1])

    def _encode_token_ids(
        self,
        prompt_ids: torch.Tensor | list[int],
        prompt_mask: torch.Tensor | None,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device, dtype=torch.long)
        prompt_ids = prompt_ids.to(self.device)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        else:
            prompt_mask = prompt_mask.to(self.device)
            if prompt_mask.ndim == 1:
                prompt_mask = prompt_mask.unsqueeze(0)

        with torch.no_grad(), _autocast(self.device, _module_dtype(self.text_encoder)):
            encoded = self.text_encoder(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                output_hidden_states=True,
            )
        hidden_states = encoded.hidden_states[-1]
        hidden_states = hidden_states[:, self._prompt_template_encode_start_idx : max_sequence_length]
        prompt_mask = prompt_mask[:, self._prompt_template_encode_start_idx : max_sequence_length]
        return hidden_states, prompt_mask

    def _prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        generator: torch.Generator | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        latent_shape = (
            batch_size,
            self.transformer.config.in_channels,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if latents is None:
            return torch.randn(latent_shape, generator=generator, device=self.device, dtype=torch.float32)
        if tuple(latents.shape) != latent_shape:
            raise ValueError(f"`latents` shape must be {latent_shape}, got {tuple(latents.shape)}.")
        return latents.to(device=self.device, dtype=torch.float32)

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32).view(
            1, -1, 1, 1, 1
        )
        std_inv = (1 / torch.as_tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)).view(
            1, -1, 1, 1, 1
        )
        vae_latents = (latents.float() / std_inv + mean).to(_module_dtype(self.vae))
        with torch.no_grad(), _autocast(_module_device(self.vae), _module_dtype(self.vae)):
            decoded = self.vae.decode(vae_latents, return_dict=False)[0]
        # VAE output is BCTHW in [-1, 1]; FlowGRPO rewards consume TCHW [0, 1].
        return ((decoded.float().clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _sample_sde_window(
        sde_window_size: int | None,
        sde_window_range: tuple[int, int] | list[int] | None,
        num_timesteps: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[int, int]:
        if sde_window_size is None:
            return (0, num_timesteps)
        if sde_window_size <= 0 or sde_window_size > num_timesteps:
            raise ValueError(f"Invalid SDE window size {sde_window_size} for {num_timesteps} timesteps.")
        start_min, start_max = (0, num_timesteps - sde_window_size)
        if sde_window_range is not None:
            if len(sde_window_range) != 2:
                raise ValueError("`sde_window_range` must contain [start, end].")
            start_min = max(start_min, int(sde_window_range[0]))
            start_max = min(start_max, int(sde_window_range[1]) - sde_window_size)
        if start_min > start_max:
            raise ValueError("`sde_window_range` cannot fit the requested `sde_window_size`.")
        start = int(torch.randint(start_min, start_max + 1, (1,), generator=generator, device=device).item())
        return start, start + sde_window_size

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        guidance: float,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []
        model_dtype = _module_dtype(self.transformer)
        self.scheduler.set_begin_index(0)

        for step_idx, timestep_value in enumerate(timesteps):
            if self.interrupt:
                break
            if step_idx == sde_window[0]:
                all_latents.append(latents.float())
            cur_noise_level = noise_level if sde_window[0] <= step_idx < sde_window[1] else 0.0
            timestep = timestep_value.expand(latents.shape[0]).to(self.device, dtype=torch.float32)
            with torch.no_grad(), _autocast(self.device, model_dtype):
                noise_pred = self.transformer(
                    hidden_states=latents.to(model_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(model_dtype),
                    encoder_attention_mask=prompt_embeds_mask,
                    return_dict=False,
                )[0]
                if do_true_cfg:
                    negative_noise_pred = self.transformer(
                        hidden_states=latents.to(model_dtype),
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds.to(model_dtype),
                        encoder_attention_mask=negative_prompt_embeds_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = apply_cfg(noise_pred, negative_noise_pred, guidance)
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents.float(),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )
            if sde_window[0] <= step_idx < sde_window[1]:
                all_latents.append(latents.float())
                all_timesteps.append(timestep_value)
                if log_prob is not None:
                    all_log_probs.append(log_prob.float())

        if not all_latents:
            raise RuntimeError("LingBot diffusion was interrupted before collecting an SDE trajectory.")
        trajectory_latents = torch.stack(all_latents, dim=1)
        trajectory_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs else None
        trajectory_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, trajectory_latents, trajectory_log_probs, trajectory_timesteps

    def forward(
        self,
        batch: DiffusionRequestBatch,
        prompt_token_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 37698,
        noise_level: float = 1.0,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: str = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        # The runner always calls custom pipelines with a DiffusionRequestBatch,
        # even for single-request forwards (supports_request_batch = False).
        if batch.num_reqs != 1:
            raise ValueError("LingBot FlowGRPO rollout supports one request per forward.")
        custom_prompt = batch.prompts[0] if batch.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_token_ids = custom_prompt.get("prompt_token_ids", prompt_token_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)
        if prompt_token_ids is None and prompt_embeds is None:
            return DiffusionOutput(output={"payload": {}, "metadata": {}})

        sampling = batch.sampling_params
        height = int(_coalesce(getattr(sampling, "height", None), height))
        width = int(_coalesce(getattr(sampling, "width", None), width))
        num_inference_steps = int(_coalesce(getattr(sampling, "num_inference_steps", None), num_inference_steps))
        max_sequence_length = int(_coalesce(getattr(sampling, "max_sequence_length", None), max_sequence_length))
        extra = getattr(sampling, "extra_args", {}) or {}
        num_frames = int(_coalesce(extra.get("num_frames"), num_frames))
        shift = float(_coalesce(extra.get("shift"), shift))
        noise_level = float(_coalesce(extra.get("noise_level"), noise_level))
        sde_window_size = _coalesce(extra.get("sde_window_size"), sde_window_size)
        sde_window_range = _coalesce(extra.get("sde_window_range"), sde_window_range)
        sde_type = _coalesce(extra.get("sde_type"), sde_type)
        logprobs = bool(_coalesce(extra.get("logprobs"), logprobs))
        if getattr(sampling, "guidance_scale_provided", False):
            guidance_scale = float(sampling.guidance_scale)
        else:
            guidance_scale = float(_coalesce(extra.get("guidance_scale"), guidance_scale))
        validate_t2v_dimensions(height, width, num_frames)

        generator = getattr(sampling, "generator", None) or generator
        if generator is None and getattr(sampling, "seed", None) is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._encode_token_ids(
                prompt_token_ids, prompt_mask, max_sequence_length
            )
        else:
            if prompt_embeds_mask is None:
                raise ValueError("`prompt_embeds_mask` is required with precomputed LingBot prompt embeddings.")
            prompt_embeds, prompt_embeds_mask = prompt_embeds.to(self.device), prompt_embeds_mask.to(self.device)
        batch_size = prompt_embeds.shape[0]
        if batch_size != 1:
            raise ValueError("LingBot FlowGRPO rollout supports one request at a time.")

        do_true_cfg = guidance_scale > 1.0
        if do_true_cfg and negative_prompt_embeds is None:
            if negative_prompt_ids is None:
                raise ValueError("LingBot CFG requires negative prompt IDs or embeddings.")
            negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_token_ids(
                negative_prompt_ids, negative_prompt_mask, max_sequence_length
            )
        if do_true_cfg and negative_prompt_embeds_mask is None:
            raise ValueError("LingBot CFG requires a negative prompt attention mask.")

        latents = self._prepare_latents(batch_size, num_frames, height, width, generator, latents)
        sigmas = shifted_sigmas(num_inference_steps, shift)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, sigmas=sigmas)
        timesteps = self.scheduler.timesteps
        sde_window = self._sample_sde_window(
            None if sde_window_size is None else int(sde_window_size),
            sde_window_range,
            len(timesteps),
            generator,
            self.device,
        )
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            timesteps,
            do_true_cfg,
            guidance_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )
        video = self._decode_video(latents)
        # vLLM-Omni main removed DiffusionOutput.custom_output (#4922).  The
        # replay trajectory + prompt embeddings now travel inside
        # output["payload"]["trajectory"]: the engine's output_formatter copies
        # that whole dict into OmniRequestOutput.multimodal_output["trajectory"]
        # (and mirrors latents/log_probs/timesteps onto the named trajectory_*
        # fields).  The consumer reads it back from multimodal_output.
        trajectory = {
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "all_timesteps": all_timesteps,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            # Named keys the formatter lifts onto OmniRequestOutput.trajectory_*.
            "latents": all_latents,
            "log_probs": all_log_probs,
            "timesteps": all_timesteps,
        }
        return DiffusionOutput(
            output={
                "payload": {"video": video[0], "trajectory": trajectory},
                "metadata": {"trajectory": {"type": "denoising"}},
            },
            trajectory_latents=all_latents,
            trajectory_log_probs=all_log_probs,
            trajectory_timesteps=all_timesteps,
            to_cpu=True,
        )

    def add_lora(self, lora_request) -> bool:
        """Load an in-memory PEFT LoRA adapter for LingBot's nn.Linear layers."""

        return self._manual_lora.add_adapter(lora_request)

    def remove_lora(self, adapter_id: int) -> bool:
        return self._manual_lora.remove_adapter(adapter_id)

    def list_loras(self) -> list[int]:
        return self._manual_lora.list_adapters()

    def pin_lora(self, adapter_id: int) -> bool:
        return self._manual_lora.pin_adapter(adapter_id)

    def set_active_lora(self, lora_request=None, lora_scale: float = 1.0) -> None:
        self._manual_lora.set_active_adapter(lora_request, lora_scale)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Accept trainer LoRA/full-state updates after eager initial loading.

        The initial vLLM-Omni loader sees ``weights_sources=()`` and calls this
        with no tensors.  Returning ``None`` intentionally disables its strict
        initial-load check because the official package has already loaded all
        components.  Later trainer updates are copied by bare-transformer or
        ``transformer.``-prefixed name.
        """

        parameters = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        for name, value in weights:
            target = parameters.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                target = parameters.get(f"transformer.{name}")
            if target is None:
                target = buffers.get(f"transformer.{name}")
            if target is not None and tuple(target.shape) == tuple(value.shape):
                target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        return None
