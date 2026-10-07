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
"""Scoped vllm-omni factory hooks that preserve rollout envelopes during postprocessing."""

import functools
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from verl_omni.pipelines.diffusion_rollout_output import _MEDIA_KEYS, _is_envelope


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

    Args:
        module: Upstream pipeline module consulted by the engine's registry.
        factory_name: Name of its postprocessor factory to wrap in this process.
    """
    factory = getattr(module, factory_name)
    if getattr(factory, "_preserves_rollout_output", False) is True:
        return

    def wrapped_factory(od_config: Any) -> Callable[..., Any]:
        return wrap_rollout_postprocessor(factory(od_config))

    wrapped_factory._preserves_rollout_output = True
    setattr(module, factory_name, wrapped_factory)
