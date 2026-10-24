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

"""Keep the dual-lane auxiliary rows out of a sampled batch, on any ``verl``.

The Bagel Co-RL writer stores *two* kinds of row under one prompt's uid
(``agent_loop/bagel_corl_tq.py``): the UND episode row carries
``prompts``/``responses``/``raw_prompt``, and each complete FlowGRPO seed group adds one GEN
row per seed carrying ``prompt_token_ids``/``all_latents``/``timesteps``. The GEN rows are
tagged ``is_auxiliary`` because they belong to a *different* consumer -- the GEN lane fetches
them by the UND row's ``child_gen_keys`` column, never by sampling them.

``ReplayBuffer._materialize_batch`` decides what a sampled batch contains, and the sibling
``verl`` checkout (0.9.0.dev, what the 2026-10-13/14 runs reached through ``PYTHONPATH``)
skips auxiliary rows there:

    # Rows tagged ``is_auxiliary`` live in the partition under their prompt's uid but are
    # not trajectories of it: the writer keeps them there for a *different* consumer that
    # addresses them by key. Sampling them would hand the actor a row its forward pass
    # cannot score (a rate-diffusion seed has no ``responses``), which shows up downstream
    # as ``assert len(output) == len(batch)`` in ``_compute_old_log_prob``.
    if tag.get("is_auxiliary", False):
        continue

The installed ``verl`` 0.10.0.dev -- what this tree now resolves to, because the sibling is
too old for it -- has no such filter (``grep -r is_auxiliary`` finds nothing outside this
repo) and materializes every partition key whose uid was selected.

That was invisible while the UND lane was broken: with ``load_format dummy`` on the AR
replica every episode had K = 0, so no GEN row was ever written and the batch was trivially
UND-only. Measured on the first run with a healthy UND lane
(``bagel_corl_prfix_t07_20261015_183002_a1``, devices 4-7): step 0 died at 18:57:03 with

    KeyError: 'key "prompts" not found in TensorDict with keys []'

inside ``_compute_reward_colocate``. The traceback's ``data`` is empty because
``transfer_queue/client.py`` returns ``TensorDict({}, batch_size=0)`` -- with
``transfer_queue/client.py:405`` "Empty BatchMeta provided to get_data" logged one line
earlier -- whenever ``len(metadata.field_names) == 0``, and the sampler's batch had just been
run through ``BatchMeta.select_fields(["prompts", "responses", "raw_prompt"])``
(``trainer_base.py:1444``). ``select_fields`` keeps only fields present across the batch's
schema, so the three UND fields are dropped the moment one GEN row is in the same batch.

``install_auxiliary_row_filter`` closes that hole from this side: the trainer must not depend
on which ``verl`` provides the replay buffer. It is the same rule, applied to whatever buffer
``_build_replay_buffer`` returns, before any consumer sees the batch.
"""

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

    Returns ``meta`` unchanged (same object) when there is nothing to drop, which is the
    K = 0 case every earlier run hit -- the filter must not perturb a lone-UND batch.

    Raises:
        RuntimeError: if *every* selected row is auxiliary. That means the prompt group was
            terminal with no UND row under its uid, and continuing would hand the trainer an
            empty batch -- the very failure this filter exists to prevent, only one frame
            later and far less legible (``KeyError: 'prompts'`` inside ``_compute_reward_
            colocate``). The TQ rows are intact at this point, so the uids are worth naming.
    """
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
    """Filter auxiliary rows out of ``buffer``'s sampled batches; idempotent.

    ``_build_replay_buffer`` gives us a buffer already constructed by whatever ``verl`` is
    installed, so the filter is applied to that instance rather than to a subclass: the base
    class resolves the sampler's own knobs (``filter_groups_metric``,
    ``sync_refill_failed_groups``, ``poll_interval``) and a subclass built next to it would
    have to either duplicate that resolution or take the ``custom_sampler`` path, which skips
    it. Both ``ReplayBuffer.sample`` and ``ReplayBufferAsync.sample`` funnel through
    ``_materialize_batch``, so one hook covers the sync lane this trainer uses and the async
    lane it does not.

    Installing twice is a no-op: the wrapper is marked, so a re-entrant
    ``_build_replay_buffer`` cannot stack filters (which would still be correct, but would
    make the row counts in a debugger confusing).
    """
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
