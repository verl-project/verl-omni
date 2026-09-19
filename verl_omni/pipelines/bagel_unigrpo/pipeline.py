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

"""Trainside "thinking -> image" pipeline for BAGEL UniGRPO (torch-only, no vLLM).

Ported from UniRL ``models/bagel/pipeline.py`` (``BagelUniPipeline._generate_t2ti``)
onto verl-omni's ``BagelForSFT``: generate an AR thinking chain on the understanding
pathway, then run a Flow-SDE diffusion sampler on the generation pathway conditioned
on (prompt + thinking), recording per-SDE-step logprob/mean for the on-policy GRPO
replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..bagel_flow_grpo.bagel_sft_model import BagelForSFT


@dataclass
class UniRolloutSample:
    """One trainside thinking->image rollout trajectory (all tensors on CPU unless noted)."""

    prompt_token_ids: list[int]
    thinking_token_ids: list[int]
    thinking_logprobs: torch.Tensor  # [T_think]
    cond_token_ids: list[int]  # prompt (+system) + thinking, fed to the gen pathway
    all_latents: torch.Tensor  # [S+1, L, D] patchified latent trajectory
    all_timesteps: torch.Tensor  # [S] sigma at each denoise step
    sde_step_indices: list[int]  # which denoise steps used the SDE (eta>0)
    sde_logp: torch.Tensor  # [n_sde] per-SDE-step summed logprob (pi_old)
    sde_means: torch.Tensor  # [n_sde, L, D] per-SDE-step reverse-SDE mean (mu_old)
    std_dev_t: torch.Tensor  # [n_sde]
    sqrt_dt: torch.Tensor  # [n_sde]
    image: torch.Tensor  # [3, H, W] uint8
    latent_grid: tuple[int, int]  # (grid_h, grid_w)


def _sde_step_indices(num_steps: int, fraction: tuple[float, float], num_sde_steps: int) -> list[int]:
    """Pick ``num_sde_steps`` denoise-step indices scattered within ``fraction`` of the schedule."""
    lo = int(round(fraction[0] * num_steps))
    hi = max(lo + 1, int(round(fraction[1] * num_steps)))
    hi = min(hi, num_steps)
    window = list(range(lo, hi))
    if not window:
        return [0]
    if num_sde_steps >= len(window):
        return window
    # even scatter across the window
    picks = torch.linspace(0, len(window) - 1, num_sde_steps).round().long().tolist()
    return sorted({window[i] for i in picks})


def _latent_grid(config, height: int, width: int) -> tuple[int, int]:
    """(grid_h, grid_w) patch-grid for an HxW image, clamped to ``max_latent_size``."""
    ds = config.latent_patch_size * config.vae_downsample
    gh = min(height // ds, config.max_latent_size)
    gw = min(width // ds, config.max_latent_size)
    return gh, gw


def _latent_pos_ids(config, grid_h: int, grid_w: int, device) -> torch.Tensor:
    """Row-major latent position ids ``[L]`` matching ``BagelForSFT._patchify_latents``."""
    ids = (
        torch.arange(grid_h, device=device)[:, None] * config.max_latent_size
        + torch.arange(grid_w, device=device)[None, :]
    ).reshape(-1)
    return ids


def _unpatchify(patches: torch.Tensor, config, grid_h: int, grid_w: int) -> torch.Tensor:
    """Inverse of ``_patchify_latents``: ``[B, L, D] -> [B, C, H, W]`` scaled VAE latent."""
    b = patches.shape[0]
    p = config.latent_patch_size
    c = config.latent_channel
    x = patches.reshape(b, grid_h, grid_w, p, p, c)
    x = torch.einsum("bhwpqc->bchpwq", x).reshape(b, c, grid_h * p, grid_w * p)
    return x


class BagelUniPipeline:
    """Trainside prompt -> thinking -> image sampler over one live ``BagelForSFT`` module."""

    def __init__(
        self,
        module: BagelForSFT,
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        shift: float = 3.0,
        eta: float = 0.8,
        sde_fraction: tuple[float, float] = (0.0, 0.2),
        num_sde_steps: int = 3,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 1024,
        top_p: float = 1.0,
        system_token_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        self.module = module
        self.config = module.config
        self.height = int(height)
        self.width = int(width)
        self.num_inference_steps = int(num_inference_steps)
        self.shift = float(shift)
        self.eta = float(eta)
        self.sde_fraction = (float(sde_fraction[0]), float(sde_fraction[1]))
        self.num_sde_steps = int(num_sde_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.system_token_ids = list(system_token_ids) if system_token_ids else []
        self.stop_token_ids = list(stop_token_ids) if stop_token_ids else None

    @torch.no_grad()
    def generate_one(self, prompt_token_ids, *, generator: torch.Generator | None = None) -> UniRolloutSample:
        """Sample one thinking chain + image trajectory from the live module."""
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        from ..bagel_flow_grpo.bagel_model import BagelForTraining
        from ..bagel_flow_grpo.common import setup_bagel_sigmas
        from .bagel_ar_thinking import generate_thinking

        module = self.module
        cfg = self.config
        device = next(module.parameters()).device
        prompt_ids = [int(t) for t in prompt_token_ids]

        # 1) AR thinking on the understanding pathway.
        think_prompt = self.system_token_ids + prompt_ids
        think_ids, think_lp = generate_thinking(
            module,
            think_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop_token_ids=self.stop_token_ids,
            generator=generator,
        )
        # Never leave the thinking chain empty. The training-side AR replay does one forward per
        # sample; an empty chain would skip that forward, so under FSDP the ranks would issue
        # different numbers of all-gather/reduce-scatter collectives and deadlock. A 1-token chain
        # keeps every sample's forward/backward count identical across ranks.
        if not think_ids:
            fallback = self.stop_token_ids[0] if self.stop_token_ids else int(self.config.text_end_id)
            think_ids = [int(fallback)]
        cond_ids = self.system_token_ids + prompt_ids + list(think_ids)
        cond = torch.tensor(cond_ids, dtype=torch.long, device=device)[None]
        cond_mask = torch.ones_like(cond, dtype=torch.bool)

        # 3) Latent geometry + initial noise in patchified latent space.
        gh, gw = _latent_grid(cfg, self.height, self.width)
        L, D = gh * gw, cfg.patch_latent_dim
        pos_ids = _latent_pos_ids(cfg, gh, gw, device)[None]
        x = torch.randn(1, L, D, device=device, dtype=torch.float32, generator=generator)
        model_dtype = module.vae2llm.weight.dtype

        # 4) Flow-SDE schedule; eta only on the scattered early-noise window.
        scheduler = FlowMatchSDEDiscreteScheduler()
        setup_bagel_sigmas(scheduler, self.num_inference_steps, shift=self.shift, device=str(device))
        scheduler._step_index = None
        timesteps = scheduler.timesteps
        num_steps = int(len(timesteps))
        sde_idx = set(_sde_step_indices(num_steps, self.sde_fraction, self.num_sde_steps))

        all_latents = [x.clone()]
        rec_logp, rec_means, rec_std, rec_sqrtdt, rec_steps = [], [], [], [], []
        for step in range(num_steps):
            t = timesteps[step].reshape(1)
            velocity = BagelForTraining.forward(
                module,
                hidden_states=x.to(model_dtype),
                timestep=t.to(model_dtype),
                text_token_ids=cond,
                latent_pos_ids=pos_ids,
                text_attention_mask=cond_mask,
            )[0]
            noise_level = self.eta if step in sde_idx else 0.0
            prev, log_prob, mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
                sample=x.float(),
                model_output=velocity.float(),
                timestep=t,
                noise_level=noise_level,
                prev_sample=None,
                sde_type="sde",
                return_logprobs=True,
                return_sqrt_dt=True,
                include_logprob_normalizer=False,
                generator=generator,
            )
            if step in sde_idx:
                rec_steps.append(step)
                rec_logp.append(log_prob.mean(dim=tuple(range(1, log_prob.ndim))).squeeze(0).detach().cpu())
                rec_means.append(mean.squeeze(0).detach().cpu())
                rec_std.append(torch.as_tensor(std_dev_t).float().reshape(-1)[0].detach().cpu())
                rec_sqrtdt.append(torch.as_tensor(sqrt_dt).float().reshape(-1)[0].detach().cpu())
            x = prev
            all_latents.append(x.clone())

        # 5) Decode the final latent to an image.
        latent = _unpatchify(x, cfg, gh, gw)
        image = self._decode_image(latent)

        return UniRolloutSample(
            prompt_token_ids=prompt_ids,
            thinking_token_ids=[int(t) for t in think_ids],
            thinking_logprobs=torch.as_tensor(think_lp).detach().cpu(),
            cond_token_ids=cond_ids,
            all_latents=torch.cat(all_latents, dim=0).detach().cpu(),
            all_timesteps=timesteps.detach().cpu(),
            sde_step_indices=rec_steps,
            sde_logp=torch.stack(rec_logp) if rec_logp else torch.zeros(0),
            sde_means=torch.stack(rec_means) if rec_means else torch.zeros(0),
            std_dev_t=torch.stack(rec_std) if rec_std else torch.zeros(0),
            sqrt_dt=torch.stack(rec_sqrtdt) if rec_sqrtdt else torch.zeros(0),
            image=image.detach().cpu(),
            latent_grid=(gh, gw),
        )

    @torch.no_grad()
    def generate_eval(
        self,
        prompt_token_ids,
        *,
        num_inference_steps: int = 50,
        cfg_text_scale: float = 4.0,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        cfg_renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        shift: float = 3.0,
        use_thinking: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Deterministic CFG image gen at BAGEL's official inference setting (eval only).

        Text-CFG only (pure thinking->image, no input image), combined via the
        byte-identical ``BagelDiffusion._combine_cfg``. Returns ``(thinking_token_ids, image[3,H,W] uint8)``.
        ``use_thinking=False`` conditions on the prompt only (apples-to-apples vs official base t2i).
        """
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        from ..bagel_flow_grpo.bagel_model import BagelForTraining
        from ..bagel_flow_grpo.common import setup_bagel_sigmas
        from ..bagel_flow_grpo.diffusers_training_adapter import BagelDiffusion
        from .bagel_ar_thinking import generate_thinking

        module, cfg = self.module, self.config
        device = next(module.parameters()).device
        prompt_ids = [int(t) for t in prompt_token_ids]
        if use_thinking:
            think_prompt = self.system_token_ids + prompt_ids
            think_ids, _ = generate_thinking(
                module,
                think_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                stop_token_ids=self.stop_token_ids,
                generator=generator,
            )
        else:
            think_ids = []
        cond_ids = self.system_token_ids + prompt_ids + list(think_ids)
        cond = torch.tensor(cond_ids, dtype=torch.long, device=device)[None]
        cond_mask = torch.ones_like(cond, dtype=torch.bool)

        gh, gw = _latent_grid(cfg, self.height, self.width)
        L, D = gh * gw, cfg.patch_latent_dim
        pos_ids = _latent_pos_ids(cfg, gh, gw, device)[None]
        x = torch.randn(1, L, D, device=device, dtype=torch.float32, generator=generator)
        model_dtype = module.vae2llm.weight.dtype

        scheduler = FlowMatchSDEDiscreteScheduler()
        setup_bagel_sigmas(scheduler, num_inference_steps, shift=shift, device=str(device))
        scheduler._step_index = None
        timesteps = scheduler.timesteps
        t_max = float(timesteps.max())
        for step in range(int(len(timesteps))):
            t = timesteps[step].reshape(1)
            t_norm = float(t) / t_max if t_max > 0 else 0.0
            v_cond = BagelForTraining.forward(
                module,
                hidden_states=x.to(model_dtype),
                timestep=t.to(model_dtype),
                text_token_ids=cond,
                latent_pos_ids=pos_ids,
                text_attention_mask=cond_mask,
            )[0].float()
            if cfg_text_scale > 1.0 and cfg_interval[0] < t_norm <= cfg_interval[1]:
                v_uncond = BagelForTraining.forward(
                    module,
                    hidden_states=x.to(model_dtype),
                    timestep=t.to(model_dtype),
                    text_token_ids=None,
                    latent_pos_ids=pos_ids,
                )[0].float()
                v = BagelDiffusion._combine_cfg(
                    v_cond, v_uncond, None, cfg_text_scale, 1.0, cfg_renorm_type, cfg_renorm_min
                )
            else:
                v = v_cond
            x, *_ = scheduler.sample_previous_step(
                sample=x.float(),
                model_output=v,
                timestep=t,
                noise_level=0.0,
                prev_sample=None,
                sde_type="sde",
                return_logprobs=True,
                generator=generator,
            )
        image = self._decode_image(_unpatchify(x, cfg, gh, gw))
        return [int(t) for t in think_ids], image.detach().cpu()

    def _decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE-decode a scaled latent ``[1, C, h, w]`` to a ``[3, H, W]`` uint8 image."""
        ae = self.module.vae_encoder.module if self.module.vae_encoder is not None else None
        if ae is None or not hasattr(ae, "decode"):
            raise RuntimeError("BagelUniPipeline: no VAE decoder available on module.vae_encoder")
        ae_dtype = next(ae.parameters()).dtype
        pixels = ae.decode(latent.to(ae_dtype))
        pixels = pixels.float().clamp(-1, 1)
        img = ((pixels[0] + 1.0) * 127.5).round().to(torch.uint8)
        return img


__all__ = ["BagelUniPipeline", "UniRolloutSample"]
