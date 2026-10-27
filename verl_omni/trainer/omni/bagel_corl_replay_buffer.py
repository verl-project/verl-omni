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

"""Keep the dual-lane auxiliary rows out of a sampled batch, on any ``verl``."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, so this module imports without torch/TQ
    from transfer_queue import KVBatchMeta

logger = logging.getLogger(__name__)


def is_auxiliary_row(tag: Any) -> bool:
    """Whether a TransferQueue row tag marks a row held for a *different* consumer.

    Only the writer's own flag counts. ``bagel_role == "gen"`` would also match the GEN
    rows today, but the tag name is the contract the sibling ``verl`` filters on, and a
    second, weaker condition here would silently disagree with it the moment a row is
    tagged one way and not the other.
    """
    return isinstance(tag, dict) and bool(tag.get("is_auxiliary"))


def drop_auxiliary_sampled_rows(meta: KVBatchMeta) -> KVBatchMeta:
    """Return ``meta`` with its auxiliary rows removed, preserving the order of the rest.

    Raises:
        RuntimeError: if *every* selected row is auxiliary. That means the prompt group was
            terminal with no UND row under its uid, and continuing would hand the trainer an
            empty batch -- the very failure this filter exists to prevent, only one frame
            later and far less legible (``KeyError: 'prompts'`` inside ``_compute_reward_
            colocate``). The TQ rows are intact at this point, so the uids are worth naming."""
    if meta is None:
        return meta
    keys = list(getattr(meta, "keys", None) or [])
    tags = list(getattr(meta, "tags", None) or [])
    if not keys or not tags:
        return meta
    keep = [i for i, tag in enumerate(tags) if not is_auxiliary_row(tag)]
    if len(keep) == len(tags):
        return meta
    if not keep:
        roles = sorted({str(tag.get("bagel_role") or "untagged") for tag in tags if isinstance(tag, dict)})
        raise RuntimeError(
            "bagel_corl replay-buffer sample selected only auxiliary rows "
            f"(rows={len(tags)}, keys={keys[:3]}, roles={roles}) on partition "
            f"{getattr(meta, 'partition_id', None)!r}. Auxiliary rows are GEN seed rows; every "
            "sampled prompt group must also have its UND episode row, so this group was "
            "terminal without one and the batch cannot be scored."
        )
    return replace(
        meta,
        keys=[keys[i] for i in keep],
        tags=[tags[i] for i in keep],
    )


def install_auxiliary_row_filter(buffer):
    """Filter auxiliary rows out of ``buffer``'s sampled batches; idempotent."""
    materialize = getattr(buffer, "_materialize_batch", None)
    if materialize is None:
        logger.warning(
            "bagel_corl: replay buffer %s exposes no _materialize_batch; rows tagged "
            "is_auxiliary (GEN seeds) cannot be kept out of a sampled batch. A custom sampler "
            "must drop them itself, or the step dies reading UND fields from a mixed batch.",
            type(buffer).__name__,
        )
        return buffer
    if getattr(materialize, "filters_auxiliary_rows", False):
        return buffer

    def _materialize_batch(partition_id, selected_prompt_uids, partition_snapshot):
        return drop_auxiliary_sampled_rows(materialize(partition_id, selected_prompt_uids, partition_snapshot))

    _materialize_batch.filters_auxiliary_rows = True
    buffer._materialize_batch = _materialize_batch
    return buffer
