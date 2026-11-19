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
"""The one LoRA sync decision table.

``resolve_lora_sync_plan`` maps merge mode (``LoRASettings``), adapter scope,
and base-sync state to the wire payload, the rollout action, and the sleep
level. Every transport (colocated ZMQ fast path, colocated bucketed sync,
standalone checkpoint engine) executes a plan instead of re-deriving the
merge-or-adapter decision, so a behavior fix lands once.

The wire contract the receivers derive: a sync sends ``peft_config`` exactly
when the payload carries adapter state, and the rollout replaces the adapter
(in-place ``remove_lora`` + ``add_lora``) exactly when ``peft_config`` arrives
with ``base_sync_done=True``; every other payload is a plain full-weight
``load_weights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from verl_omni.utils.config import LoRASettings

__all__ = ["SyncExport", "SyncRolloutAction", "SyncPlan", "resolve_lora_sync_plan"]


class SyncExport(StrEnum):
    """What the training engine exports for one rollout sync."""

    FULL = "full"  # every weight, HF-named; adapter-renamed on the AR first sync
    MERGED = "merged"  # LoRA fused into the base weights
    ADAPTER = "adapter"  # adapter-only A/B tensors


class SyncRolloutAction(StrEnum):
    """What the rollout executes for the payload it receives."""

    LOAD_WEIGHTS = "load_weights"
    REPLACE_ADAPTER = "replace_adapter"  # remove_lora + add_lora


@dataclass(frozen=True)
class SyncPlan:
    """One resolved row of the sync decision table.

    ``sleep_level`` pins how the rollout sleeps after the sync: level 1 keeps
    the adapter (and base) weights resident so adapter-mode syncs skip the full
    weight resume; level 2 is the full sleep the merged and non-LoRA rows pay
    for their full-weight syncs.
    """

    adapter_name: str
    export: SyncExport
    send_peft_config: bool
    sleep_level: int

    @property
    def rollout_action(self) -> SyncRolloutAction:
        """The rollout action implied by the payload: adapter tensors are
        injected in place; full and merged weights are loaded."""
        if self.export is SyncExport.ADAPTER:
            return SyncRolloutAction.REPLACE_ADAPTER
        return SyncRolloutAction.LOAD_WEIGHTS


def resolve_lora_sync_plan(
    settings: LoRASettings, *, adapter_name: str | None, has_adapter: bool, base_sync_done: bool
) -> SyncPlan:
    """Resolve one sync to its plan; impossible combinations raise.

    The rows:
    - non-LoRA: full weights, no adapter metadata, full (level-2) sleep.
    - merge: merged full weights every sync — the rollout is never
      adapter-aware, so level 2.
    - adapter first sync (``base_sync_done=False``): full base weights plus the
      adapter, with ``peft_config`` so the rollout recreates it, then level 1
      so the base weights survive the sleep.
    - adapter delta (``base_sync_done=True``): adapter-only tensors, level 1.

    Raises:
        ValueError: merge mode asked to ship a named adapter (merge fuses the
            active ``default`` only), or LoRA enabled while the actor holds no
            adapter under the requested name (add it to
            ``policy_state_adapters``) — a silent full-weight sync of the wrong
            policy would otherwise follow.
    """
    adapter = adapter_name or "default"
    if not settings.enabled:
        return SyncPlan(adapter_name=adapter, export=SyncExport.FULL, send_peft_config=False, sleep_level=2)
    if settings.merge:
        if adapter != "default":
            raise ValueError(
                f"model.lora.merge=True exports the active 'default' adapter only; "
                f"got rollout_adapter={adapter_name!r}."
            )
        return SyncPlan(adapter_name=adapter, export=SyncExport.MERGED, send_peft_config=False, sleep_level=2)
    if not has_adapter:
        raise ValueError(
            f"LoRA is enabled (lora_rank={settings.rank}) but the actor engine holds no "
            f"{adapter!r} adapter; add it to actor_rollout_ref.model.policy_state_adapters."
        )
    if not base_sync_done:
        return SyncPlan(adapter_name=adapter, export=SyncExport.FULL, send_peft_config=True, sleep_level=1)
    return SyncPlan(adapter_name=adapter, export=SyncExport.ADAPTER, send_peft_config=True, sleep_level=1)
