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
"""BAGEL (MoT) rollout-side adapter for FlowGRPO.

Extends ``BagelPipeline`` with an SDE scheduler for stochastic denoising
and log-probability recording.  Applies per-request SDE windowing so noise
is only injected on a contiguous subset of denoising steps, matching the
original flow_grpo BAGEL rollout.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import route_actor_weight_for_und_replica
from verl_omni.pipelines.bagel_flow_grpo.common import (
    BAGEL_FLOWGRPO_CFG_DEFAULTS,
    maybe_to_cpu,
    setup_bagel_sigmas,
)
from verl_omni.pipelines.diffusion_rollout_output import rollout_output, with_visual_artifacts
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.rollout_request import prompt_ids_from_payload
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

logger = logging.getLogger(__name__)


def _extract_bagel_trajectory(output: DiffusionOutput) -> tuple[Any, Any, Any]:
    """Read BAGEL's payload trajectory, preferring native top-level fields."""
    latents = output.trajectory_latents
    timesteps = output.trajectory_timesteps
    log_probs = output.trajectory_log_probs
    if isinstance(output.output, dict):
        payload = output.output.get("payload")
        trajectory = payload.get("trajectory") if isinstance(payload, dict) else None
        if isinstance(trajectory, dict):
            latents = latents if latents is not None else trajectory.get("latents")
            timesteps = timesteps if timesteps is not None else trajectory.get("timesteps")
            log_probs = log_probs if log_probs is not None else trajectory.get("log_probs")
    return latents, timesteps, log_probs


# TODO: Drop decode→re-tokenize helpers once vllm-omni BagelPipeline accepts
# prompt_token_ids directly (currently only reads text from req.prompts[0]["prompt"]).
_CHAT_MARKERS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def _to_token_list(token_ids: Any) -> list[int] | None:
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _extract_prompt_text(decoded: str) -> str:
    if "<|im_start|>" in decoded:
        user_chunks = []
        for segment in decoded.split("<|im_start|>"):
            if not segment.startswith("user"):
                continue
            content = segment[len("user") :].lstrip("\n")
            content = content.split("<|im_end|>", 1)[0]
            user_chunks.append(content)
        if user_chunks:
            decoded = user_chunks[-1]

    for marker in _CHAT_MARKERS:
        decoded = decoded.replace(marker, "")
    return decoded.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()


@dataclass
class _AdapterStepOutput:
    """Adapter output matching what bagel_transformer.generate_image expects."""

    prev_sample: torch.Tensor
    log_prob: torch.Tensor | None


class _BagelSchedulerAdapter:
    """Adapt ``FlowMatchSDEDiscreteScheduler`` to BAGEL's calling convention.

    BAGEL calls ``scheduler.step(v_t, sigma, x_t, dt, **kwargs)`` with 4
    positional args; the diffusers scheduler expects 3.  SDE noise and
    log-prob recording are gated to a per-request window so steps outside
    the window run deterministically (ODE, ``noise_level=0``).
    """

    def __init__(self, inner: FlowMatchSDEDiscreteScheduler):
        self._inner = inner
        self._sde_window: Optional[tuple[int, int]] = None
        self._base_noise_level: float = 0.0
        self._base_return_logprobs: bool = True
        self._step_counter: int = 0
        self._tp_divergence_reported: bool = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def begin_forward(
        self,
        sde_window: Optional[tuple[int, int]],
        noise_level: float,
        return_logprobs: bool,
    ) -> None:
        """Reset adapter state before each rollout ``forward`` call.

        Args:
            sde_window: ``(begin, end_exclusive)`` step range where SDE
                noise is injected and log-probs are recorded.  ``None``
                disables windowing, i.e. ``noise_level`` applies to every
                step (legacy flow_grpo behaviour).
            noise_level: SDE noise level to apply inside the window.
            return_logprobs: whether log-probs are requested at all
                (overridden to ``False`` outside the window even when
                ``True`` here).
        """
        self._sde_window = sde_window
        self._base_noise_level = float(noise_level)
        self._base_return_logprobs = bool(return_logprobs)
        self._step_counter = 0

    def step(
        self,
        model_output: torch.Tensor,
        sigma: float | torch.Tensor,
        sample: torch.Tensor,
        dt: float | torch.Tensor,  # noqa: ARG002 — inner derives dt from timestep schedule
        **kwargs,
    ) -> _AdapterStepOutput:
        """Run one denoising step, gating noise and log-probs by the SDE window.

        Args:
            model_output: Velocity prediction ``v_t`` from the model.
            sigma: Current noise level (BAGEL uses raw sigma, not 0-1000).
            sample: Current latent ``x_t``.
            dt: Step size (ignored; derived from the inner scheduler's
                timestep schedule).

        Returns:
            ``(prev_sample, log_prob)`` where ``log_prob`` is a scalar
            (or ``None`` outside the SDE window).
        """
        i = self._step_counter
        if self._sde_window is not None:
            begin, end = self._sde_window
            in_window = begin <= i < end
        else:
            # No window: "noise on every step" is the legacy flow_grpo behaviour.
            in_window = True
        # Always pass the *caller's* noise level explicitly, window or not.
        # Leaving it implicit (only overriding inside a window) silently hands
        # control to the inner scheduler's own default, which is
        # ``noise_level=0.7`` -- so a caller asking for a deterministic decode
        # (``noise_level=0.0``, e.g. validation) would get maximum SDE noise on
        # every single step instead of an ODE rollout.
        cur_noise_level = self._base_noise_level if in_window else 0.0
        # A zero-noise step has no distribution to score: ``std_dev_t == 0`` makes
        # the Gaussian log-prob ``0/0 -> nan`` (and its normalizer ``log(0)``).
        # Never request log-probs on a step that injects no noise.
        cur_return_logprobs = bool(self._base_return_logprobs and in_window and cur_noise_level > 0.0)
        kwargs = {
            **kwargs,
            "noise_level": cur_noise_level,
            "return_logprobs": cur_return_logprobs,
        }

        sample_in = sample.unsqueeze(0)
        model_output_in = model_output.unsqueeze(0)
        if "prev_sample" in kwargs:
            kwargs = {**kwargs, "prev_sample": kwargs["prev_sample"].unsqueeze(0)}

        out = self._inner.step(
            model_output=model_output_in.float(),  # cast bf16→fp32 for scheduler precision
            timestep=sigma,
            sample=sample_in,
            return_dict=False,
            **kwargs,
        )
        self._step_counter += 1
        prev_sample, log_prob = out[0], out[1]
        if cur_noise_level > 0.0:
            prev_sample = self._sync_tp_sample(prev_sample, i)
        prev_sample = prev_sample.squeeze(0)
        if log_prob is not None:
            log_prob = log_prob.reshape(())
        return _AdapterStepOutput(prev_sample=prev_sample, log_prob=log_prob)

    def _sync_tp_sample(self, prev_sample: torch.Tensor, step: int) -> torch.Tensor:
        """Make every tensor-parallel rank continue from TP rank 0's noisy sample.

        Each TP rank runs the whole denoise loop and draws its own SDE noise. The
        DiT's sharded attention/MLP all-reduce partial results computed from each
        rank's *own* latent, so ranks whose noise differs feed mismatched inputs
        into one forward and the decoded image comes out grainy (ODE steps are
        deterministic and never diverge). Rank 0's sample, and therefore its
        recorded log-prob, is left unchanged. Every rank must reach this call on
        the same steps, which ``_tp_shared_sde_window`` guarantees.
        """
        tp_group = _tp_tensor_parallel_group()
        if tp_group is None:
            return prev_sample
        if not self._tp_divergence_reported:
            checksum = prev_sample.detach().double().sum().reshape(1)
            checksums = tp_group.all_gather(checksum, dim=0)
            if not torch.allclose(checksums, checksums[0].expand_as(checksums)):
                self._tp_divergence_reported = True
                logger.warning(
                    "BagelPipelineWithLogProb: SDE sample diverged across TP ranks at step %d "
                    "(per-rank checksums %s); broadcasting TP rank 0's sample.",
                    step,
                    checksums.tolist(),
                )
        prev_sample = prev_sample.contiguous()
        return tp_group.broadcast(prev_sample, src=0)


def _pick_sde_window(
    window_size: Optional[int],
    window_range: Optional[Any],
    seed: int,
    num_denoise_steps: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    """Pick a random contiguous window ``[begin, begin + window_size)``.

    Uses ``seed`` directly so that all rollouts share the same SDE window,
    matching the official flow_grpo behaviour of
    ``random.seed(process_index)`` per GPU.

    Args:
        window_size: Number of steps in the window.  ``None`` or 0
            disables windowing.
        window_range: ``(low, high)`` inclusive range for the window
            start.  ``None`` defaults to ``[0, window_size)``.
        seed: Seed for the RNG.
        num_denoise_steps: Total number of denoise steps the scheduler will
            run (``len(timesteps)``).  When given, the window is clamped to
            ``end_exclusive <= num_denoise_steps - 1`` so that the *terminal*
            step (index ``num_denoise_steps - 1``, which lands on ``sigma == 0``)
            always stays outside the window.  Noise injected on that step is
            never removed again -- there is no later step to undo it -- so it
            would be baked into the decoded image.

    Returns:
        ``(begin, end_exclusive)`` or ``None`` if windowing is disabled.

    Note:
        ``window_range`` is expressed in *step indices*, but the number of
        denoise steps is ``num_inference_steps - 1``.  A range that fits the
        advertised ``num_inference_steps`` can therefore overrun the real
        schedule once the step count drops (e.g. the 1-GPU recipe branch uses
        ``num_inference_steps=4``, i.e. 3 denoise steps, against the default
        ``[0, 7]``).  Left unclamped, a window past the last step applies noise
        and records log-probs on *no* step at all, which silently empties the
        GEN trajectory instead of failing.
    """
    if window_size is None or int(window_size) <= 0:
        return None

    size = int(window_size)
    if num_denoise_steps is not None:
        # ``setup_bagel_sigmas`` accepts ``num_steps=1`` for warmup runs, which
        # leaves a single denoise step that *is* the terminal one; there is no
        # earlier step to relocate the window to, so shrink to the only step
        # rather than addressing none of them.
        size = min(size, max(int(num_denoise_steps) - 1, 1))

    if window_range is None:
        low, high = 0, size
    else:
        low, high = int(window_range[0]), int(window_range[1])

    high_inclusive = high - size
    if num_denoise_steps is not None:
        last_clean_begin = max(int(num_denoise_steps) - 1 - size, 0)
        high_inclusive = min(high_inclusive, last_clean_begin)
        low = min(low, last_clean_begin)

    if high_inclusive < low:
        # Window doesn't fit; clamp to the lowest valid begin.
        return (low, low + size)

    rng = random.Random(seed)
    begin = rng.randint(low, high_inclusive)
    return (begin, begin + size)


def _tp_tensor_parallel_group():
    """Return vLLM's TP group when the engine runs with TP > 1, else ``None``."""
    from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    if not model_parallel_is_initialized() or get_tensor_model_parallel_world_size() <= 1:
        return None
    return get_tp_group()


def _tp_shared_sde_window(sde_window: tuple[int, int]) -> tuple[int, int]:
    """Use TP rank 0's SDE window on every tensor-parallel rank of the engine.

    ``_pick_sde_window`` seeds from ``LOCAL_RANK``, which differs between the TP
    workers of one engine. Each worker runs the full denoise loop, so differing
    windows put noise on different steps per rank, the sharded DiT all-reduces
    partial results computed from diverged latents, and the image comes out grainy.
    """
    tp_group = _tp_tensor_parallel_group()
    if tp_group is None:
        return sde_window
    window = torch.tensor(sde_window, dtype=torch.int64, device=tp_group.device)
    window = tp_group.broadcast(window, src=0)
    begin, end = window.tolist()
    return (int(begin), int(end))


@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")
class BagelPipelineWithLogProb(BagelPipeline):
    """BAGEL pipeline variant for RL rollouts with verl-omni."""

    #: Declares the named artifacts available to a request, so consumers select
    #: one by name instead of inferring modality from tensor rank or channel size.
    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "LC"),
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        inner = FlowMatchSDEDiscreteScheduler()
        self.scheduler = _BagelSchedulerAdapter(inner)
        # One-shot guard for ``_audit_und_sync``: the sync repeats every step, the
        # verdict does not change.
        self._und_sync_audited = False
        logger.info("BagelPipelineWithLogProb: SDE scheduler enabled for RL rollouts")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, routing by name prefix.

        Weight-sync from the actor uses ``transformer.`` prefix with separate
        q/k/v projections; the parent's ``AutoWeightsLoader`` cannot map these
        to the rollout's fused ``qkv_proj``.  Delegate such weights to
        ``language_model.load_weights`` which handles stacked-param remapping.
        Other weights (initial checkpoint load) defer to the parent.
        """
        actor_weights: list[tuple[str, torch.Tensor]] = []
        und_head_weights: list[tuple[str, torch.Tensor]] = []
        checkpoint_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            routed = route_actor_weight_for_und_replica(name)
            if routed is not None and routed.startswith("lm_head."):
                und_head_weights.append((routed, tensor))
            elif routed is not None:
                actor_weights.append((routed, tensor))
            else:
                checkpoint_weights.append((name, tensor))

        loaded: set[str] = set()
        if checkpoint_weights:
            loaded |= super().load_weights(checkpoint_weights)
        if actor_weights:
            acc = self.language_model.load_weights(actor_weights)
            loaded |= acc
            self._audit_und_sync(actor_weights, acc)
        if und_head_weights:
            loaded |= self.language_model.load_weights(und_head_weights)
        return loaded

    def _audit_und_sync(
        self,
        routed: list[tuple[str, torch.Tensor]],
        accepted: set[str],
    ) -> None:
        """Report what the actor→AR weight sync actually wrote, once per process.

        The sync is the only thing that rewrites the live UND replica, and it is
        *silent*: ``BagelTransformer.load_weights`` logs a single
        ``warning_once`` for every name it cannot place, so a systematically
        mis-remapped sync (correct enqueue, wrong destination) leaves no trace in
        the log at all -- the replica just starts emitting degenerate text and it
        looks like a model-quality problem. This counts the damage and names the
        parameters that the sync did *not* touch, so a partial or mis-targeted
        remap is visible in one run instead of a bisect.

        Args:
            routed: ``(name, tensor)`` pairs handed to ``language_model.load_weights``.
            accepted: the parameter names that loader reported as written.
        """
        if self._und_sync_audited:
            return
        self._und_sync_audited = True
        try:
            all_params = {name for name, _ in self.language_model.named_parameters()}
        except Exception:  # pragma: no cover - audit must never break a rollout
            return
        untouched = sorted(all_params - set(accepted))
        logger.info(
            "und_sync_audit: routed=%d accepted=%d lm_params=%d untouched=%d lm_head_written=%s",
            len(routed),
            len(accepted),
            len(all_params),
            len(untouched),
            sorted(n for n in accepted if "lm_head" in n) or "NO",
        )
        if untouched:
            logger.info(
                "und_sync_audit: %d param(s) left at their loaded checkpoint value, first 20: %s",
                len(untouched),
                untouched[:20],
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "und_sync_audit: routed names (first 20): %s; accepted (first 20): %s",
                [n for n, _ in routed][:20],
                sorted(accepted)[:20],
            )

    def _decode_token_prompt(self, token_ids: Any, *, native: bool = False) -> str | None:
        """Decode BAGEL token IDs to a cleaned prompt text string."""
        token_list = _to_token_list(token_ids)
        if not token_list:
            if native:
                raise ValueError("Native BAGEL prompts must preserve start/end token framing.")
            return None
        if native:
            bos = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if token_list[0] != bos or token_list[-1] != eos:
                raise ValueError("Native BAGEL prompts must preserve start/end token framing.")
            return self.tokenizer.decode(token_list[1:-1], skip_special_tokens=False)
        decoded = self.tokenizer.decode(token_list, skip_special_tokens=False)
        return _extract_prompt_text(decoded)

    def _ensure_bagel_prompt_text(self, req: OmniDiffusionRequest) -> None:
        """Fill ``prompt`` and ``negative_prompt`` from token IDs if missing."""
        if not req.prompts or not isinstance(req.prompts[0], dict):
            return

        custom_prompt = req.prompts[0]
        if not custom_prompt.get("prompt"):
            prompt = self._decode_token_prompt(
                prompt_ids_from_payload(custom_prompt),
                native=req.sampling_params.extra_args.get("native_prompt_tokens", False),
            )
            if prompt is not None:
                custom_prompt["prompt"] = prompt

        extra_args = req.sampling_params.extra_args
        if "negative_prompt" not in extra_args:
            negative_prompt = self._decode_token_prompt(custom_prompt.get("negative_prompt_ids"))
            if negative_prompt is not None:
                extra_args["negative_prompt"] = negative_prompt

    def _decode_image_from_latent(self, bagel, vae, latent, image_shape):
        # Upstream decodes the final sample before optional trajectory previews.
        if self._final_image_latent is None:
            self._final_image_latent = latent
        return super()._decode_image_from_latent(bagel, vae, latent, image_shape)

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        self._ensure_bagel_prompt_text(req)

        # Force trajectory recording on for RL
        req.sampling_params.return_trajectory_latents = True

        extra_args = req.sampling_params.extra_args

        # Apply CFG defaults so rollout and training log-prob recomputation match.
        for k, v in BAGEL_FLOWGRPO_CFG_DEFAULTS.items():
            extra_args.setdefault(k, v)
        if isinstance(extra_args.get("cfg_interval"), list):
            extra_args["cfg_interval"] = tuple(extra_args["cfg_interval"])

        # Pick SDE window: noise and log-prob recording only inside this range.
        logprobs = bool(extra_args.get("logprobs", True))
        noise_level = float(extra_args.get("noise_level", 0.0))
        sde_window_size = extra_args.get("sde_window_size", None)
        sde_window_range = extra_args.get("sde_window_range", None)
        if isinstance(sde_window_range, list):
            sde_window_range = tuple(sde_window_range)

        # Per-request scheduler setup matching training-side sigma schedule.
        # Done before the window pick because the window must be clamped against
        # the *real* denoise-step count: ``setup_bagel_sigmas`` installs
        # ``num_inference_steps`` sigma points but only ``num_inference_steps - 1``
        # denoise steps (the terminal sigma is 0).
        assert req.sampling_params.num_inference_steps is not None, "num_inference_steps must be set for RL rollouts"
        bagel_num_timesteps = int(req.sampling_params.num_inference_steps)
        setup_bagel_sigmas(self.scheduler._inner, bagel_num_timesteps)
        num_denoise_steps = len(self.scheduler._inner.timesteps)

        sde_window: Optional[tuple[int, int]] = None
        if sde_window_size and noise_level > 0.0:
            sde_window = _pick_sde_window(
                window_size=int(sde_window_size),
                window_range=sde_window_range,
                seed=int(os.environ["LOCAL_RANK"]),
                num_denoise_steps=num_denoise_steps,
            )
            sde_window = _tp_shared_sde_window(sde_window)

        # Pass scheduler kwargs; _BagelSchedulerAdapter overrides noise_level
        # and return_logprobs per-step based on the SDE window.
        self.scheduler_kwargs = {k: extra_args[k] for k in ("noise_level", "sde_type", "generator") if k in extra_args}
        self.scheduler_kwargs["return_logprobs"] = logprobs
        # BAGEL FlowGRPO compares quadratic log-prob terms only.
        self.scheduler_kwargs["include_logprob_normalizer"] = False

        # Reset adapter state *after* set_timesteps so inner step_index is None.
        self.scheduler.begin_forward(
            sde_window=sde_window,
            noise_level=noise_level,
            return_logprobs=logprobs,
        )

        # The resolved window is the one number that separates a *bounded* SDE
        # (noise + log-probs on a couple of mid-schedule steps, which later
        # deterministic steps undo) from the ``_sde_window is None`` fallback,
        # where ``_BagelSchedulerAdapter.step`` passes no overrides and the
        # inner scheduler applies its own default ``noise_level=0.7`` to *every*
        # step -- including the terminal sigma->0 step, where nothing can remove
        # it again.  Report it once per request so a dropped ``sde_window_size``
        # (e.g. an OpenAI body-param whitelist) is visible instead of showing up
        # only as grainy rollout images.
        if noise_level > 0.0 and sde_window is None:
            logger.warning(
                "BagelPipelineWithLogProb: noise_level=%.3f but no SDE window "
                "(sde_window_size=%r, sde_window_range=%r, denoise_steps=%d) -- falling back to "
                "the scheduler default, i.e. noise on EVERY step including the terminal "
                "sigma=0 step. That noise is never removed and is baked into the image. "
                "Set rollout.algo.sde_window_size.",
                noise_level,
                sde_window_size,
                sde_window_range,
                num_denoise_steps,
            )
        else:
            logger.info(
                "BagelPipelineWithLogProb: noise_level=%.3f sde_window=%s sde_window_size=%r "
                "sde_window_range=%r denoise_steps=%d",
                noise_level,
                sde_window,
                sde_window_size,
                sde_window_range,
                num_denoise_steps,
            )

        # vllm-omni >= 0.24 (#4509) matches official BAGEL: n schedule points, n-1 denoise steps.
        self._final_image_latent = None
        output = super().forward(req)
        native_latent = self._final_image_latent
        self._final_image_latent = None
        if native_latent is None:
            raise RuntimeError(
                f"pipeline={type(self).__name__}, request_id={req.request_id}: no final native image latent"
            )

        # Slice trajectory to the SDE window so training only sees noisy steps.
        traj_latents, traj_timesteps, traj_log_probs = _extract_bagel_trajectory(output)

        if sde_window is not None:
            begin, end = sde_window
            if traj_latents is not None:
                traj_latents = traj_latents[begin : end + 1]
            if traj_timesteps is not None:
                traj_timesteps = traj_timesteps[begin:end]
            if traj_log_probs is not None:
                # ... (see note below) only clamp when the engine handed back the whole
                # per-step trajectory instead of the window-local vector.
                _n_logp = int(traj_log_probs.shape[0]) if hasattr(traj_log_probs, "shape") else len(traj_log_probs)
                if _n_logp > (end - begin):
                    # ``latents`` / ``timesteps`` are recorded on *every* step, so they still
                    # need this window slice.  ``log_probs`` does not: ``begin_forward`` above
                    # gates ``return_logprobs`` to the window, so the scheduler returns a
                    # log-prob only on in-window steps and the engine's vector is *already*
                    # window-local (length ``end - begin``, aligned with ``traj_timesteps``).
                    # Re-slicing it with the absolute ``[begin:end]`` would select nothing once
                    # ``begin >= end - begin`` -- e.g. ``begin=3, end=5`` on a 2-element vector
                    # -- emptying the GEN log-probs and dropping every GEN row downstream.
                    # Keep the slice only as a guard for an engine that returns the full
                    # per-step trajectory instead.
                    traj_log_probs = traj_log_probs[begin:end]

        # BAGEL trajectories are time-major; add a batch axis for training consumers.
        if traj_latents is not None:
            traj_latents = traj_latents.unsqueeze(0)
        if traj_timesteps is not None:
            traj_timesteps = traj_timesteps.unsqueeze(0)
        if traj_log_probs is not None:
            traj_log_probs = traj_log_probs.unsqueeze(0)

        payload = output.output["payload"]
        context = f"pipeline={type(self).__name__}, request_id={req.request_id}"
        if "image" not in payload or payload.keys() - {"image", "trajectory"}:
            raise ValueError(f"{context}: expected image and optional trajectory, got {list(payload)}")
        media = torch.from_numpy(np.array(payload["image"], copy=True)).permute(2, 0, 1).unsqueeze(0)
        metadata = dict(output.output.get("metadata") or {})
        result = rollout_output(
            media=media,
            trajectory_decoded=payload.get("trajectory", {}).get("decoded"),
            trajectory_latents=maybe_to_cpu(traj_latents),
            trajectory_timesteps=maybe_to_cpu(traj_timesteps),
            trajectory_log_probs=maybe_to_cpu(traj_log_probs),
            metadata=metadata,
            to_cpu=False,
        )
        return with_visual_artifacts(
            result,
            decoded=media,
            pixel_range="uint8",
            latents=maybe_to_cpu(native_latent).unsqueeze(0),
            latent_layout="LC",
            output_type=req.sampling_params.output_type or "pil",
            context=context,
            requested=(req.sampling_params.extra_args or {}).get("requested_outputs"),
        )
