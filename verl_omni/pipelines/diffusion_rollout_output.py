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
"""Construct native rollout trajectory fields and payload/metadata envelopes.

Trajectories use ``DiffusionOutput.trajectory_*``. Prompt embeddings and
algorithm-specific tensors use the canonical payload/metadata envelope.
Only ``trajectory_*`` and ``rl`` / ``prompt_embeddings`` reach training; ``metadata=`` does not.
The postprocessor hook keeps that envelope intact when the pinned engine
runs a media-only factory after the adapter returns. The assembly helpers
attach declared decoded media and unchanged native latents to that envelope.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from types import ModuleType
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.rollout_media import (
    MediaArtifact,
    MediaSpec,
    is_decoded_visual,
    requested_artifact_names,
    select_artifact,
    validate_artifacts,
)

_MEDIA_KEYS = frozenset(("image", "video", "output", "audio"))


def contains_meta_tensor(value: Any) -> bool:
    """Detect offloaded placeholder tensors in a nested rollout output."""
    if isinstance(value, torch.Tensor):
        return value.is_meta
    if isinstance(value, Mapping):
        return any(contains_meta_tensor(item) for item in value.values())
    if isinstance(value, (list | tuple)):
        return any(contains_meta_tensor(item) for item in value)
    return False


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


def wrap_rollout_postprocessor(postprocess: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a media-only upstream postprocessor to preserve rollout payload and metadata.

    Args:
        postprocess: Native processor for tensor media, including its request-level kwargs.

    Returns:
        Processor that bypasses named artifacts and preserves training metadata.
    """

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


def install_rollout_postprocessor(module: ModuleType, factory_name: str) -> None:
    """Wrap one upstream factory once; ordinary inference still uses its native processor.

    The pinned engine resolves postprocessors from module factories, not pipeline
    instances or individual outputs. Named media is already normalized by adapters.

    The wrapper only exists in the importing process. It takes effect because the
    pinned engine runs a single-replica diffusion stage inline in the rollout server
    and resolves the postprocessor before creating executors. A stage launched in
    another process keeps the native processor.

    Args:
        module: Upstream pipeline module consulted by the engine's registry.
        factory_name: Name of its postprocessor factory to wrap in this process.
    """
    # TODO (NancyFyong, verl-project/verl-omni#403): Remove once upstream postprocessors
    # pass named rollout envelopes through without wrapping their factories.
    factory = getattr(module, factory_name)
    if getattr(factory, "_preserves_rollout_output", False) is True:
        return

    def wrapped_factory(od_config: Any) -> Callable[..., Any]:
        return wrap_rollout_postprocessor(factory(od_config))

    wrapped_factory._preserves_rollout_output = True
    setattr(module, factory_name, wrapped_factory)


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
    """Replace legacy media with a named, normalized per-sample payload.

    One modality-keyed payload holds all named tensors so the pinned upstream
    formatter cannot discard non-primary streams. Trajectories and algorithm
    metadata remain separate. This does not require changing upstream types.

    Args:
        base: Existing rollout output with trajectory fields and algorithm metadata.
        artifacts: Named artifacts for one sample, without a batch axis. Decoded
            pixels use uint8; audio and latents use floating-point tensors.
        specs: Declarations for all emitted names, including per-sample layouts.
        primary: Name of the primary artifact; determines the payload's modality key.
        context: Pipeline/request description used in validation errors.
        audio: Optional name of a decoded audio artifact.
        preview: Optional name of a decoded image or video artifact.
        requested: Names that must be present, not an exclusive output filter.

    Returns:
        Copy of base with named media and selectors; trajectory fields are preserved.
    """
    normalized = validate_artifacts(artifacts, specs, primary=primary, context=context, requested=requested)
    metadata = dict(base.output.get("metadata") or {}) if _is_envelope(base.output) else {}
    if "media_artifacts" in metadata:
        raise ValueError(f"{context}: duplicate media_artifacts metadata")
    if audio is not None:
        select_artifact(normalized, name=audio, modality="audio", representation="decoded")
    if preview is not None:
        artifact = normalized.get(preview)
        if not is_decoded_visual(artifact):
            raise ValueError(f"{context}: preview artifact={preview!r} must be decoded visual media")
    metadata["media_artifacts"] = {
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
    """Normalize explicit B-prefixed tensors and retain the sample list for request splitting.

    Args:
        base: Existing rollout output with trajectory fields and algorithm metadata.
        data: Named tensors with a common nonempty leading batch axis, e.g. (B, C, H, W)
            for a CHW declaration. Decoded pixels use uint8; audio and latents use
            floating-point tensors.
        specs: Declarations for all emitted names; layouts exclude the batch axis.
        primary: Name of the primary artifact; determines the payload's modality key.
        context: Pipeline/request description; the sample index is appended for validation.
        preview: Optional name of a decoded image or video artifact.
        audio: Optional name of a decoded audio artifact.
        requested: Names that must be present in every sample, not an exclusive output filter.

    Returns:
        Copy of base with a list of per-sample named payloads and shared declarations.
    """
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


def wants_decoded_preview(
    output_type: str,
    sampling_params: OmniDiffusionSamplingParams,
    *,
    modality: str = "image",
    requested: list[str] | None = None,
) -> bool:
    """Honor explicit preview requests even when the primary output is latent.

    Args:
        output_type: Requested primary representation, including ``latent``.
        sampling_params: Request sampling parameters carrying ``requested_outputs`` in extra_args.
        modality: Visual modality whose named preview is checked.
        requested: Optional union of artifact names from a packed request batch.

    Returns:
        Whether decoded media is needed for the primary or an explicitly requested preview.
    """
    if requested is None:
        requested = (sampling_params.extra_args or {}).get("requested_outputs")
    names = requested_artifact_names(requested, context="preview decode")
    return output_type != "latent" or f"{modality}_preview" in names


def quantize_pixels(decoded: torch.Tensor, pixel_range: str, *, context: str) -> torch.Tensor:
    """Quantize an explicitly declared VAE range, never infer encoding from dtype or extrema.

    Args:
        decoded: VAE pixels in their declared range, with any explicitly known layout.
        pixel_range: ``minus_one_one`` or ``zero_one`` for floating pixels, or ``uint8``.
        context: Pipeline/request description used in validation errors.

    Returns:
        uint8 pixels in [0, 255] with the input shape preserved.
    """
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

    Args:
        base: Existing output whose trajectory fields and algorithm metadata are preserved.
        decoded: Batched VAE pixels, e.g. (B, C, H, W), or None when no preview is needed.
        latents: Floating final sampler latents, e.g. (B, L, C), before VAE transformations.
        latent_layout: Per-sample native latent axes, excluding B.
        output_type: Selects the latent primary for ``latent``, otherwise the decoded preview.
        context: Pipeline/request description used in validation errors.
        requested: Artifact names required in every sample, not an exclusive output filter.
        modality: Image or video modality for both artifacts.
        decoded_layout: Per-sample decoded axes, excluding B.
        pixel_range: Explicit VAE pixel range for quantization.
        fps: Emitted video frame rate; required when decoded video is provided.

    Returns:
        Copy of base with named native latents and an optional normalized decoded preview.
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
