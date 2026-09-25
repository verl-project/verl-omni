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
"""Helpers for vllm-omni 0.26's native rollout output contract.

Trajectories use ``DiffusionOutput.trajectory_*``. Prompt embeddings and
algorithm-specific tensors use the canonical payload/metadata envelope.
Only ``trajectory_*`` and ``rl`` / ``prompt_embeddings`` reach training; ``metadata=`` does not.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput

from verl_omni.pipelines.rollout_artifacts import MediaArtifact, select_artifact, validate_artifacts
from verl_omni.pipelines.rollout_media import MediaSpec

_MEDIA_KEYS = frozenset(("image", "video", "output", "audio"))


def rollout_output(
    *,
    media: Any,
    media_key: str = "image",
    trajectory_latents: Any = None,
    trajectory_log_probs: Any = None,
    trajectory_timesteps: Any = None,
    trajectory_decoded: Any = None,
    prompt_embeddings: Mapping[str, Any] | None = None,
    rl: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    to_cpu: bool = True,
) -> DiffusionOutput:
    """Build a native rollout output without duplicating trajectory tensors."""
    return DiffusionOutput(
        output=_envelope(media, media_key, prompt_embeddings, rl, metadata),
        trajectory_latents=trajectory_latents,
        trajectory_log_probs=trajectory_log_probs,
        trajectory_timesteps=trajectory_timesteps,
        trajectory_decoded=trajectory_decoded,
        to_cpu=to_cpu,
    )


def with_rollout_data(
    base: DiffusionOutput,
    *,
    trajectory_latents: Any = None,
    trajectory_log_probs: Any = None,
    trajectory_timesteps: Any = None,
    trajectory_decoded: Any = None,
    prompt_embeddings: Mapping[str, Any] | None = None,
    rl: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    media_key: str = "image",
    to_cpu: bool = True,
) -> DiffusionOutput:
    """Add rollout data to an existing output while preserving base fields."""
    media, resolved_key, existing_metadata = _unwrap_output(base.output, media_key)
    merged_metadata = {**existing_metadata, **dict(metadata or {})}
    return replace(
        base,
        output=_envelope(media, resolved_key, prompt_embeddings, rl, merged_metadata),
        trajectory_latents=trajectory_latents if trajectory_latents is not None else base.trajectory_latents,
        trajectory_log_probs=trajectory_log_probs if trajectory_log_probs is not None else base.trajectory_log_probs,
        trajectory_timesteps=trajectory_timesteps if trajectory_timesteps is not None else base.trajectory_timesteps,
        trajectory_decoded=trajectory_decoded if trajectory_decoded is not None else base.trajectory_decoded,
        to_cpu=to_cpu,
    )


def with_media_artifacts(
    base: DiffusionOutput,
    *,
    artifacts: list[tuple[str, MediaArtifact]],
    specs: Mapping[str, MediaSpec],
    primary: str,
    context: str,
    audio: str | None = None,
    preview: str | None = None,
    requested: list[str] | None = None,
) -> DiffusionOutput:
    """Replace legacy media with a named, already-normalized per-sample payload.

    One modality-keyed payload holds all named tensors so the pinned upstream
    formatter cannot discard non-primary streams. Trajectories and algorithm
    metadata remain separate. This does not require changing upstream types.
    """
    normalized = validate_artifacts(artifacts, specs, primary=primary, context=context, requested=requested)
    metadata = dict(base.output.get("metadata") or {}) if _is_envelope(base.output) else {}
    if "media_artifacts" in metadata:
        raise ValueError(f"{context}: duplicate media_artifacts metadata")
    if audio is not None:
        select_artifact(normalized, name=audio, modality="audio", representation="decoded")
    if preview is not None:
        artifact = normalized.get(preview)
        if (
            artifact is None
            or artifact.spec.representation != "decoded"
            or artifact.spec.modality not in ("image", "video")
        ):
            raise ValueError(f"{context}: preview artifact={preview!r} must be decoded visual media")
    metadata["media_artifacts"] = {
        "context": context,
        "primary": primary,
        "audio": audio,
        "preview": preview,
        "specs": {name: asdict(artifact.spec) for name, artifact in normalized.items()},
    }
    modality = normalized[primary].spec.modality
    return replace(
        base,
        output={
            "payload": {modality: {name: artifact.data for name, artifact in normalized.items()}},
            "metadata": metadata,
        },
    )


def with_batched_media_artifacts(
    base: DiffusionOutput,
    *,
    data: Mapping[str, torch.Tensor],
    specs: Mapping[str, MediaSpec],
    primary: str,
    context: str,
    preview: str | None = None,
    audio: str | None = None,
    requested: list[str] | None = None,
) -> DiffusionOutput:
    """Normalize explicit B-prefixed tensors and retain the sample list for request splitting."""
    if primary not in data:
        raise ValueError(f"{context}: missing primary artifact={primary!r}")
    batch_size = data[primary].shape[0]
    if batch_size < 1:
        raise ValueError(f"{context}: expected a nonempty artifact batch")
    for name, tensor in data.items():
        if name not in specs:
            raise ValueError(f"{context}: undeclared artifact={name!r}")
        if tensor.ndim != len(specs[name].layout) + 1 or tensor.shape[0] != batch_size:
            raise ValueError(
                f"{context}, artifact={name!r}: expected B{specs[name].layout}, B={batch_size}, got {tensor.shape}"
            )
    samples = []
    for index in range(batch_size):
        result = with_media_artifacts(
            base,
            artifacts=[(name, MediaArtifact(specs[name], tensor[index])) for name, tensor in data.items()],
            specs=specs,
            primary=primary,
            preview=preview,
            audio=audio,
            context=f"{context}, sample={index}",
            requested=requested,
        )
        samples.append(result.output["payload"][specs[primary].modality])
    result.output["payload"][specs[primary].modality] = samples
    return result


def wants_decoded_preview(output_type: str, sampling_params: Any, *, modality: str = "image") -> bool:
    """Honor explicit preview requests even when the primary output is latent."""
    requested = (sampling_params.extra_args or {}).get("requested_outputs") or []
    return output_type != "latent" or f"{modality}_preview" in requested


def quantize_pixels(decoded: torch.Tensor, pixel_range: str, *, context: str) -> torch.Tensor:
    """Quantize an explicitly declared VAE range, never infer encoding from dtype or extrema."""
    if pixel_range == "uint8":
        if decoded.dtype != torch.uint8:
            raise ValueError(f"{context}: expected uint8 pixels, got {decoded.dtype}")
        return decoded
    if pixel_range not in ("minus_one_one", "zero_one") or not decoded.is_floating_point():
        raise ValueError(f"{context}: invalid floating pixel encoding {pixel_range!r}, dtype={decoded.dtype}")
    if not torch.isfinite(decoded).all():
        raise ValueError(f"{context}: nonfinite decoded pixels")
    # Match VaeImageProcessor's denormalization before float32 quantization.
    pixels = decoded / 2 + 0.5 if pixel_range == "minus_one_one" else decoded
    return pixels.float().clamp(0, 1).mul(255).round().to(torch.uint8)


def with_visual_artifacts(
    base: DiffusionOutput,
    *,
    decoded: torch.Tensor | None,
    latents: torch.Tensor,
    latent_layout: str,
    output_type: str,
    context: str,
    requested: list[str] | None = None,
    modality: Literal["image", "video"] = "image",
    decoded_layout: str = "CHW",
    pixel_range: Literal["minus_one_one", "zero_one", "uint8"] = "minus_one_one",
    fps: float | None = None,
) -> DiffusionOutput:
    """Adapter boundary for explicit batched VAE pixels plus unchanged native latents.

    The caller declares the pixel range and every axis; neither dtype nor shape
    is used to infer representation. Training/replay tensors in ``base`` are untouched.
    """
    latent_name, preview_name = f"{modality}_latent", f"{modality}_preview"
    data = {latent_name: latents}
    specs = {latent_name: MediaSpec(modality, "latent", latent_layout)}
    if decoded is not None:
        data[preview_name] = quantize_pixels(decoded, pixel_range, context=context)
        specs[preview_name] = MediaSpec(modality, "decoded", decoded_layout, fps=fps)
    return with_batched_media_artifacts(
        base,
        data=data,
        specs=specs,
        primary=latent_name if output_type == "latent" else preview_name,
        preview=preview_name if decoded is not None else None,
        context=context,
        requested=requested,
    )


def wrap_rollout_postprocessor(postprocess: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a media-only upstream postprocessor to preserve rollout payload and metadata."""

    @functools.wraps(postprocess)
    def wrapped(data: Any, **kwargs: Any) -> Any:
        if not _is_envelope(data):
            return postprocess(data, **kwargs)

        payload = data["payload"]
        metadata = dict(data.get("metadata") or {})
        if "media_artifacts" in metadata:
            return data  # Named tensors are already decoded/normalized by the adapter.
        if len(payload) != 1:
            raise ValueError("A media-only postprocessor cannot consume multiple payload keys; use named artifacts.")
        (media_key,) = payload
        if media_key not in _MEDIA_KEYS:
            raise ValueError("Diffusion output envelope has no media payload.")

        processed = postprocess(payload[media_key], **kwargs)
        if _is_envelope(processed):
            return {
                "payload": dict(processed["payload"]),
                "metadata": {**dict(processed.get("metadata") or {}), **metadata},
            }
        if isinstance(processed, Mapping):
            return {"payload": dict(processed), "metadata": metadata}
        return {"payload": {media_key: processed}, "metadata": metadata}

    return wrapped


def _is_envelope(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("payload"), Mapping)


def _envelope(
    media: Any,
    media_key: str,
    prompt_embeddings: Mapping[str, Any] | None,
    rl: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result_metadata = dict(metadata or {})
    if prompt_embeddings is not None:
        result_metadata["prompt_embeddings"] = dict(prompt_embeddings)
    if rl is not None:
        result_metadata["rl"] = dict(rl)
    return {"payload": {media_key: media}, "metadata": result_metadata}


def _unwrap_output(output: Any, default_key: str) -> tuple[Any, str, dict[str, Any]]:
    if not _is_envelope(output):
        return output, default_key, {}
    payload = output["payload"]
    if len(payload) != 1:
        raise ValueError("Cannot unwrap multiple diffusion payload keys without dropping media; use named artifacts.")
    (key,) = payload
    if key not in _MEDIA_KEYS:
        raise ValueError("Diffusion output envelope has no media payload.")
    return payload[key], key, dict(output.get("metadata") or {})
