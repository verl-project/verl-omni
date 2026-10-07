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
"""Adapter-boundary assembly of declared decoded media and unchanged native latents."""

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.diffusion_rollout_output import _is_envelope
from verl_omni.pipelines.rollout_artifacts import MediaArtifact, select_artifact, validate_artifacts
from verl_omni.pipelines.rollout_media import MediaSpec


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
    output_type: str, sampling_params: OmniDiffusionSamplingParams, *, modality: str = "image"
) -> bool:
    """Honor explicit preview requests even when the primary output is latent.

    Args:
        output_type: Requested primary representation, including ``latent``.
        sampling_params: Request sampling parameters carrying ``requested_outputs`` in extra_args.
        modality: Visual modality whose named preview is checked.

    Returns:
        Whether decoded media is needed for the primary or an explicitly requested preview.
    """
    requested = (sampling_params.extra_args or {}).get("requested_outputs") or []
    return output_type != "latent" or f"{modality}_preview" in requested


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
