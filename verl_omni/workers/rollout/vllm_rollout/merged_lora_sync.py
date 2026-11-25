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
"""Trainer-side bridge onto vLLM-Omni's ``omni_lora_ipc`` weight-transfer engine.

Composes ``vllm_omni.lora.weight_transfer.MergedLoRAIPCTrainerWeightTransferEngine``
and ``LoRAAdapterWeightSource`` over ``vLLMOmniHttpServer``'s four-phase + abort
RPC surface so verl-omni does not duplicate IPC staging, handle wire encoding,
cross-rank coordination, or abort cleanup.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

import torch
from verl.utils.device import get_torch_device

__all__ = [
    "MergedLoRASyncSession",
    "RolloutWeightSyncClient",
    "normalize_peft_config_for_merged_sync",
]


def _normalize_value(value: Any) -> Any:
    """Recursively convert sets, tuples, and enums into msgspec-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set | frozenset):
        return sorted(_normalize_value(item) for item in value)
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalize_value(item) for key, item in value.items()}
    return value


def normalize_peft_config_for_merged_sync(peft_config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a msgspec-safe copy of ``peft_config`` with sets sorted into lists.

    PEFT's ``LoraConfig.to_dict()`` emits ``set`` fields (such as
    ``target_modules``) and ``Enum`` values (``peft_type``). Normalizing them
    keeps payloads deterministic across ranks and avoids set-vs-list mismatches
    in the engine's target matcher.
    """
    if peft_config is None:
        return None
    return {key: _normalize_value(value) for key, value in peft_config.items()}


def _load_vllm_omni_trainer_api() -> tuple[Any, Any, Any]:
    """Import phase-2 merged LoRA trainer classes from ``vllm_omni`` lazily."""
    try:
        from vllm_omni.lora.weight_transfer import (
            LoRAAdapterWeightSource,
            MergedLoRAIPCTrainerInitInfo,
            MergedLoRAIPCTrainerWeightTransferEngine,
        )
    except ImportError as exc:
        raise RuntimeError(
            "actor_rollout_ref.model.lora.merge='engine' requires vLLM-Omni with phase-2 merged LoRA "
            "trainer support (MergedLoRAIPCTrainerWeightTransferEngine from vllm-omni#10998, commit "
            "95513b53 or newer)."
        ) from exc
    return LoRAAdapterWeightSource, MergedLoRAIPCTrainerInitInfo, MergedLoRAIPCTrainerWeightTransferEngine


class RolloutWeightSyncClient:
    """``VLLMWeightSyncClient`` adapter forwarding four-phase + abort RPCs to ``vLLMOmniHttpServer``.

    Runs inside the worker thread that executes
    ``MergedLoRAIPCTrainerWeightTransferEngine.send_weights()`` and dispatches
    each Ray actor call back onto the actor's active asyncio event loop.
    """

    def __init__(
        self,
        server_handle: Any,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        call_fn: Callable[..., Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Bind to a ``vLLMOmniHttpServer`` Ray handle (or ``None`` on non-sender ranks)."""
        self._server_handle = server_handle
        self._loop = loop
        self._call_fn = call_fn
        self._cancel_event = cancel_event

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Update the asyncio loop used to schedule remote actor calls."""
        self._loop = loop

    def set_cancel_event(self, cancel_event: threading.Event | None) -> None:
        """Update the cancellation signal checked before non-abort RPC steps."""
        self._cancel_event = cancel_event

    def _check_cancelled(self, method_name: str) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise RuntimeError(f"Merged LoRA weight sync cancelled before {method_name}.")

    def _invoke(self, method_name: str, *args: Any, allow_when_cancelled: bool = False, **kwargs: Any) -> Any:
        if not allow_when_cancelled:
            self._check_cancelled(method_name)
        if self._server_handle is None:
            raise RuntimeError(f"RolloutWeightSyncClient.{method_name} called without a rollout server handle.")
        remote_fn = getattr(self._server_handle, method_name).remote
        if self._call_fn is not None:
            return self._call_fn(remote_fn, *args, **kwargs)
        loop = self._loop
        if loop is None:
            raise RuntimeError("RolloutWeightSyncClient requires an active asyncio loop or call_fn.")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            raise RuntimeError(
                "RolloutWeightSyncClient cannot block the active event loop thread; "
                "run sync() in a worker thread or pass call_fn."
            )

        async def _call_remote() -> Any:
            return await remote_fn(*args, **kwargs)

        return asyncio.run_coroutine_threadsafe(_call_remote(), loop).result()

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        """Send the backend initialization handshake to enabled rollout stages."""
        self._invoke("init_weight_transfer_engine", init_info)

    def start_weight_update(self, update_id: str | None = None) -> None:
        """Open a weight-update session, tagging it with ``update_id`` when provided."""
        if update_id is None:
            self._invoke("start_weight_update")
        else:
            self._invoke("start_weight_update", update_id=update_id)

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Deliver one serialized merged-LoRA IPC chunk to enabled rollout stages."""
        self._invoke("update_weights", update_info)

    def finish_weight_update(self) -> None:
        """Commit the active weight-update session on enabled rollout stages."""
        self._invoke("finish_weight_update")

    def abort_weight_update(self, update_id: str | None = None) -> bool:
        """Abort an active session without committing; returns False if another session owns it."""
        if update_id is None:
            return bool(self._invoke("abort_weight_update", allow_when_cancelled=True))
        return bool(self._invoke("abort_weight_update", allow_when_cancelled=True, update_id=update_id))


class MergedLoRASyncSession:
    """Reusable per-worker trainer session for ``omni_lora_ipc`` weight syncs.

    Initializes ``MergedLoRAIPCTrainerWeightTransferEngine`` on the first sync
    (triggering ``init_weight_transfer_engine`` once on sender ranks) and swaps
    in updated adapter tensors via ``LoRAAdapterWeightSource.update`` on later
    steps.
    """

    def __init__(
        self,
        *,
        server_handle: Any,
        is_sender: bool,
        rank: int = 0,
        process_group: Any = None,
        device: torch.device | int | None = None,
        lora_int_id: int = 1,
        encode_handles: bool = True,
        call_fn: Callable[..., Any] | None = None,
        gpu_uuid: str | None = None,
        reduce_tensor_fn: Callable[[torch.Tensor], tuple[Any, tuple[Any, ...]]] | None = None,
        stage_tensor_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ipc_collect_fn: Callable[[], None] | None = None,
        synchronize_fn: Callable[[], None] | None = None,
        release_ipc_counter_fn: Callable[[tuple[Any, ...]], None] | None = None,
    ) -> None:
        """Configure the session state and optional test/runtime seams."""
        self.server_handle = server_handle
        self.is_sender = is_sender
        self.rank = rank
        self._process_group = process_group
        self._device = device
        self._lora_int_id = lora_int_id
        self._encode_handles = encode_handles
        self._gpu_uuid = gpu_uuid
        self._reduce_tensor_fn = reduce_tensor_fn
        self._stage_tensor_fn = stage_tensor_fn
        self._ipc_collect_fn = ipc_collect_fn
        self._synchronize_fn = synchronize_fn
        self._release_ipc_counter_fn = release_ipc_counter_fn
        self._client = RolloutWeightSyncClient(server_handle, call_fn=call_fn)
        self._source: Any = None
        self._engine: Any = None

    def sync(
        self,
        lora_weights: Mapping[str, torch.Tensor],
        peft_config: Mapping[str, Any] | None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Publish one adapter snapshot through ``MergedLoRAIPCTrainerWeightTransferEngine``."""
        normalized_cfg = normalize_peft_config_for_merged_sync(peft_config)
        if not normalized_cfg:
            raise ValueError("engine-side LoRA merge requires a non-empty peft_config.")

        self._client.set_loop(loop)
        self._client.set_cancel_event(cancel_event)
        try:
            if self._engine is None:
                source_cls, init_info_cls, trainer_engine_cls = _load_vllm_omni_trainer_api()
                source = source_cls(lora_weights, normalized_cfg, lora_int_id=self._lora_int_id)
                device = self._device
                if device is None and self._gpu_uuid is None:
                    device = get_torch_device().current_device()
                # IPCTrainerInitInfo derives is_sender from ``rank == 0``; map each
                # rollout replica's leader (rollout_rank == 0) to rank=0 so multi-
                # replica colocated setups publish to every replica server handle.
                init_rank = 0 if self.is_sender else max(1, self.rank)
                init_info = init_info_cls(
                    rank=init_rank,
                    packed=False,
                    process_group=self._process_group,
                    encode_handles=self._encode_handles,
                    lora_int_id=self._lora_int_id,
                    device=device,
                    gpu_uuid=self._gpu_uuid,
                    reduce_tensor_fn=self._reduce_tensor_fn,
                    stage_tensor_fn=self._stage_tensor_fn,
                    ipc_collect_fn=self._ipc_collect_fn,
                    synchronize_fn=self._synchronize_fn,
                    release_ipc_counter_fn=self._release_ipc_counter_fn,
                )
                engine = trainer_engine_cls.trainer_init(init_info, client=self._client, source=source)
                self._source = source
                self._engine = engine
            else:
                self._source.update(lora_weights, peft_config=normalized_cfg, lora_int_id=self._lora_int_id)

            self._engine.send_weights()
        finally:
            self._client.set_cancel_event(None)
