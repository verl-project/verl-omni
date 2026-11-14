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
"""Named diffusion declarations and the per-sample tensors that satisfy them.

Adapters declare available names with ``DiffusionIOSpec``. ``MediaArtifact`` validates
one sample and projects it into TensorDict fields. No model runtime is imported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch

Modality = Literal["image", "video", "audio"]


def resolve_is_video(media_kind: str | None) -> bool:
    """Read the declared modality; rank is never a modality discriminator."""
    if media_kind is None:
        raise ValueError("Explicit media_kind required, got None")
    if media_kind not in ("image", "video", "audio"):
        raise ValueError(f"Unsupported media kind: {media_kind!r}")
    return media_kind == "video"


def resolve_batch_media_kind(media_kinds: Iterable[str | None]) -> str | None:
    """Require one declared modality per pipeline batch; ignore absent legacy metadata."""
    resolved = None
    for kind in media_kinds:
        if kind is None:
            continue
        resolve_is_video(kind)
        if resolved is not None and resolved != kind:
            raise ValueError(f"Conflicting media kinds in one rollout batch: {resolved!r} and {kind!r}")
        resolved = kind
    return resolved


def validate_visual_media_batch_rank(ndim: int, media_kind: str | None) -> None:
    """Validate declared image/video batches before legacy layout normalization."""
    if media_kind is None:
        return
    if media_kind == "image" and ndim != 4:
        raise ValueError(f"Declared media_kind='image' requires an NCHW batch, got rank {ndim}.")
    if media_kind == "video" and ndim != 5:
        raise ValueError(f"Declared media_kind='video' requires a rank-5 batch, got rank {ndim}.")
    if media_kind == "audio":
        raise ValueError("Cannot dump declared audio as a visual generation batch.")
    resolve_is_video(media_kind)


@dataclass(frozen=True)
class MediaSpec:
    """One media declaration; dtype belongs to the tensor, not config."""

    modality: Modality
    representation: Literal["decoded", "latent"]
    layout: str
    sample_rate: int | None = None
    fps: float | None = None


@dataclass(frozen=True)
class DiffusionIOSpec:
    """Available named artifacts, with canonical decoded and native latent axes.

    Each request selects its primary and optional preview explicitly. Runtime
    sample rate/FPS live on returned artifacts; declarations constrain only the
    adapter-owned artifact names, representations, and layouts.
    """

    artifacts: Mapping[str, MediaSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, Mapping) or not self.artifacts:
            raise ValueError("DiffusionIOSpec requires named artifacts")
        if any(not isinstance(name, str) or not isinstance(spec, MediaSpec) for name, spec in self.artifacts.items()):
            raise TypeError("DiffusionIOSpec requires artifact-name to MediaSpec declarations")


ARTIFACT_PREFIX = "media_artifact__"
ARTIFACT_SPECS = "media_artifact_specs"
PRIMARY_ARTIFACT = "primary_artifact"
PREVIEW_ARTIFACT = "preview_artifact"
ARTIFACT_CONTEXT = "media_artifact_context"


class ArtifactContractError(ValueError):
    """A schema/selection failure, never an optional scorer or observability failure."""


@dataclass(frozen=True)
class MediaArtifact:
    """A named stream's declaration and one sample's tensor (no batch axis)."""

    spec: MediaSpec
    data: torch.Tensor
    context: str = ""

    def validate(self, *, context: str, name: str) -> None:
        """Fail closed on incorrect metadata, layout, shape or dtype."""
        spec, data = self.spec, self.data
        prefix = f"{self.context or context}, artifact={name!r}"
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"{prefix}: expected an identifier artifact name")
        if spec.modality not in ("image", "video", "audio") or spec.representation not in ("latent", "decoded"):
            raise ValueError(f"{prefix}: invalid modality/representation {spec}")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{prefix}: expected tensor, got {type(data).__name__}")
        if not re.fullmatch(r"[A-Z]+", spec.layout) or len(set(spec.layout)) != len(spec.layout):
            raise ValueError(f"{prefix}: expected distinct uppercase layout axes, got {spec.layout!r}")
        if data.ndim != len(spec.layout) or any(size <= 0 for size in data.shape):
            raise ValueError(f"{prefix}: expected layout={spec.layout}, got shape={tuple(data.shape)}")
        if spec.representation == "latent":
            if not data.is_floating_point():
                raise ValueError(f"{prefix}: expected floating latent dtype, got {data.dtype}")
        else:
            layouts = {"image": ("CHW", "HWC"), "video": ("TCHW", "CTHW", "THWC"), "audio": ("T", "CT")}
            if spec.layout not in layouts[spec.modality]:
                raise ValueError(
                    f"{prefix}: expected decoded {spec.modality} layout in {layouts[spec.modality]}, got {spec.layout}"
                )
            if spec.modality == "audio":
                if not data.is_floating_point():
                    raise ValueError(f"{prefix}: expected floating audio waveform, got {data.dtype}")
                if isinstance(spec.sample_rate, bool) or not isinstance(spec.sample_rate, int) or spec.sample_rate <= 0:
                    raise ValueError(f"{prefix}: expected positive audio sample_rate, got {spec.sample_rate!r}")
            elif data.dtype != torch.uint8:
                raise ValueError(f"{prefix}: expected uint8 decoded pixels, got {data.dtype}")
            if "C" in spec.layout and spec.modality != "audio" and data.shape[spec.layout.index("C")] not in (1, 3, 4):
                raise ValueError(f"{prefix}: expected 1/3/4 image channels, got shape={tuple(data.shape)}")
        if spec.modality == "video" and spec.representation == "decoded" and spec.fps is None:
            raise ArtifactContractError(f"{prefix}: decoded video requires explicit fps")
        if spec.fps is not None and (
            spec.modality != "video"
            or isinstance(spec.fps, bool)
            or not isinstance(spec.fps, int | float)
            or not 0 < spec.fps < float("inf")
        ):
            raise ValueError(f"{prefix}: expected positive finite video fps, got {spec.fps!r}")
        if spec.sample_rate is not None and spec.modality != "audio":
            raise ValueError(f"{prefix}: sample_rate belongs only to audio")

    def normalized(self, *, context: str, name: str) -> MediaArtifact:
        """Normalize decoded axes to CHW/TCHW/CT without touching native latents."""
        self.validate(context=context, name=name)
        if self.spec.representation == "latent":
            return self
        target = {"image": "CHW", "video": "TCHW", "audio": "CT"}[self.spec.modality]
        if self.spec.layout == target:
            return self
        if self.spec.layout == "T":
            data = self.data.unsqueeze(0)
        else:
            data = self.data.permute(*(self.spec.layout.index(axis) for axis in target)).contiguous()
        return replace(self, spec=replace(self.spec, layout=target), data=data)


def is_decoded_visual(artifact: MediaArtifact | None) -> bool:
    """Return whether an artifact declares a decoded image or video preview.

    Args:
        artifact: A validated artifact, or None for an absent selection.

    Returns:
        Whether its declared representation and modality support visual previews.
    """
    return (
        artifact is not None
        and artifact.spec.representation == "decoded"
        and artifact.spec.modality in ("image", "video")
    )


def requested_artifact_names(requested: Any, *, context: str) -> list[str]:
    """Validate output selection without interpreting strings/dicts as name sequences."""
    if requested is None:
        return []
    if (
        isinstance(requested, str)
        or not isinstance(requested, Sequence)
        or any(not isinstance(name, str) for name in requested)
    ):
        raise ArtifactContractError(f"{context}: requested_outputs must be a sequence of artifact names")
    if len(set(requested)) != len(requested):
        raise ArtifactContractError(f"{context}: duplicate requested_outputs")
    return list(requested)


def validate_artifacts(
    items: Iterable[tuple[str, MediaArtifact]],
    specs: Mapping[str, MediaSpec],
    *,
    primary: str,
    context: str,
    requested: Iterable[str] | None = None,
) -> dict[str, MediaArtifact]:
    """Validate names and expected declarations before normalizing each artifact."""
    artifacts = {}
    for name, artifact in items:
        if name in artifacts:
            raise ValueError(f"{context}: duplicate artifact={name!r}")
        if name not in specs:
            raise ValueError(f"{context}: undeclared artifact={name!r}")
        if artifact.spec != specs[name]:
            raise ValueError(f"{context}, artifact={name!r}: expected {specs[name]}, got {artifact.spec}")
        artifact = replace(artifact, context=artifact.context or context)
        artifacts[name] = artifact.normalized(context=context, name=name)
    missing = specs.keys() - artifacts.keys()
    if missing:
        raise ValueError(f"{context}: declared artifacts missing: {sorted(missing)}")
    if primary not in artifacts:
        raise ValueError(f"{context}: primary artifact={primary!r} is absent")
    absent = set(requested_artifact_names(requested, context=context)) - artifacts.keys()
    if absent:
        raise ArtifactContractError(f"{context}: requested artifacts absent: {sorted(absent)}")
    return artifacts


def artifact_fields(artifacts: Mapping[str, MediaArtifact], primary: str, preview: str | None = None) -> dict[str, Any]:
    """Flatten tensors into regular TensorDict fields; metadata contains no tensors."""
    if primary not in artifacts:
        raise ValueError(f"Primary artifact {primary!r} is absent")
    if preview is not None and not is_decoded_visual(artifacts.get(preview)):
        raise ValueError(f"Preview artifact={preview!r} must name decoded visual media")
    for name, artifact in artifacts.items():
        artifact.validate(context="artifact transport", name=name)
    return {
        **{ARTIFACT_PREFIX + name: artifact.data for name, artifact in artifacts.items()},
        ARTIFACT_SPECS: {name: asdict(artifact.spec) for name, artifact in artifacts.items()},
        PRIMARY_ARTIFACT: primary,
        PREVIEW_ARTIFACT: preview,
        ARTIFACT_CONTEXT: artifacts[primary].context,
    }


def artifacts_from_fields(fields: Mapping[str, Any], *, context: str) -> dict[str, MediaArtifact]:
    """Restore one sample without inferring representation or dropping unknown names."""
    origin = fields.get(ARTIFACT_CONTEXT)
    if origin is not None and not isinstance(origin, str):
        raise ArtifactContractError(f"{context}: artifact context must be a string")
    if origin:
        context = f"{origin}, {context}"
    raw_specs = fields.get(ARTIFACT_SPECS)
    tensors = {key[len(ARTIFACT_PREFIX) :]: value for key, value in fields.items() if key.startswith(ARTIFACT_PREFIX)}
    if raw_specs is None:
        if tensors or fields.get(PRIMARY_ARTIFACT) is not None:
            raise ValueError(f"{context}: artifact tensors/primary have no declarations")
        return {}
    if not isinstance(raw_specs, Mapping):
        raise TypeError(f"{context}: artifact declarations must be a mapping")
    try:
        specs = {name: MediaSpec(**spec) for name, spec in raw_specs.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid artifact declarations: {error}") from error
    unknown = tensors.keys() - specs.keys()
    if unknown:
        raise ValueError(f"{context}: undeclared artifact tensors: {sorted(unknown)}")
    return validate_artifacts(
        ((name, MediaArtifact(specs[name], tensor)) for name, tensor in tensors.items()),
        specs,
        primary=fields.get(PRIMARY_ARTIFACT),
        context=context,
    )


def validate_audio(audio: Any, sample_rate: Any, *, context: str) -> None:
    """Validate the explicitly CT audio projection; no batch/axis or sample-rate fallback."""
    if audio is not None:
        if isinstance(sample_rate, torch.Tensor):
            sample_rate = sample_rate.item()
        artifact = MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=sample_rate), torch.as_tensor(audio))
        artifact.validate(context=context, name="audio")


def validate_previews(previews: list[MediaArtifact] | None, count: int) -> list[MediaArtifact] | None:
    """Validate export selection synchronously, before best-effort I/O starts."""
    if previews is None:
        return None
    if not previews:
        return []
    if len(previews) == count and all(preview is None for preview in previews):
        return []
    if len(previews) != count:
        raise ValueError(f"Expected {count} previews, got {len(previews)}")
    for i, preview in enumerate(previews):
        if not isinstance(preview, MediaArtifact):
            raise ValueError(f"Preview sample={i} is missing; cannot substitute a response/latent")
        preview.validate(context=f"preview sample={i}", name="preview")
        if not is_decoded_visual(preview):
            raise ValueError(f"Preview sample={i}: expected decoded visual media, got {preview.spec}")
    if len({preview.spec.modality for preview in previews}) != 1:
        raise ValueError("Cannot export a mixed image/video preview batch")
    return [preview.normalized(context="preview export", name="preview") for preview in previews]


def previews_from_batch(batch: Any) -> list[MediaArtifact] | None:
    """Select explicit decoded previews from DataProto rows for dump/W&B paths."""
    if ARTIFACT_SPECS not in batch.non_tensor_batch:
        if any(key.startswith(ARTIFACT_PREFIX) for key in batch.batch.keys()):
            raise ValueError("Media export received artifact tensors without declarations")
        return None
    previews = []
    for i in range(len(batch)):
        row = batch[i]
        fields = dict(row.non_tensor_batch)
        fields.update({key: value for key, value in row.batch.items() if key.startswith(ARTIFACT_PREFIX)})
        artifacts = artifacts_from_fields(fields, context=f"media export sample={i}")
        if PREVIEW_ARTIFACT not in fields:
            raise ValueError(f"media export sample={i}: missing explicit preview selector")
        name = fields[PREVIEW_ARTIFACT]
        if name is None:
            previews.append(None)
            continue
        if name not in artifacts:
            raise ValueError(f"media export sample={i}: requested decoded preview {name!r} is absent")
        artifact = artifacts[name]
        if not is_decoded_visual(artifact):
            raise ValueError(f"media export sample={i}, artifact={name!r}: expected decoded visual preview")
        previews.append(artifact)
    return previews


def select_artifact(
    artifacts: Mapping[str, MediaArtifact], *, name: str, modality: str, representation: str
) -> MediaArtifact:
    """Select exactly the requested stream; never substitute a latent for a preview."""
    context = next((item.context for item in artifacts.values() if item.context), "artifact consumer")
    if name not in artifacts:
        raise ArtifactContractError(f"{context}: Requested artifact={name!r} is absent; available={sorted(artifacts)}")
    artifact = artifacts[name]
    try:
        artifact.validate(context="artifact consumer", name=name)
    except (TypeError, ValueError) as error:
        raise ArtifactContractError(str(error)) from error
    if (artifact.spec.modality, artifact.spec.representation) != (modality, representation):
        raise ArtifactContractError(
            f"{context}, artifact={name!r}: expected {modality}/{representation}, got {artifact.spec}"
        )
    return artifact
