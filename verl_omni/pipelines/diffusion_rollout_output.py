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
from dataclasses import replace
from typing import Any

from vllm_omni.diffusion.data import DiffusionOutput

_MEDIA_KEYS = ("image", "video", "output", "audio")


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


def wrap_rollout_postprocessor(postprocess: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a media-only upstream postprocessor to preserve rollout payload and metadata."""

    @functools.wraps(postprocess)
    def wrapped(data: Any, **kwargs: Any) -> Any:
        if not _is_envelope(data):
            return postprocess(data, **kwargs)

        payload = data["payload"]
        metadata = dict(data.get("metadata") or {})
        media_key = next((key for key in _MEDIA_KEYS if key in payload), None)
        if media_key is None:
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
    for key in _MEDIA_KEYS:
        if key in payload:
            return payload[key], key, dict(output.get("metadata") or {})
    raise ValueError("Diffusion output envelope has no media payload.")
