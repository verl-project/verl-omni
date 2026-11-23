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
"""Audio/video-lean TransferQueue agent loop for the omni V1 trainer.

Omni-modality rollouts embed decoded media directly in ``raw_prompt`` messages
(e.g. :class:`NextQARLHFDataset` stores the full float32 soundtrack waveform in
each audio item). Upstream ``AgentLoopWorkerTQ`` then writes those messages into
every TransferQueue row — once per ``rollout.n`` session — and ships them again
to the async reward worker, so megabytes of waveform per sample ride each of the
~3 storage round-trips per step even though nothing downstream of the rollout
reads them (rule-based scoring runs from live kwargs before the TQ put, and the
training legs only need the token/multimodal-input fields). This is the omni-AR
analogue of the payload-transfer waste described in verl-omni issue #714.

Select with::

    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.av_agent_loop_tq.AVLeanAgentLoopManagerTQ

Environment gates (all default off, i.e. bit-identical upstream behavior):

- ``VERL_OMNI_TQ_STRIP_RAW_PROMPT=1`` — omit ``raw_prompt`` from TQ rows and
  from the async-reward Ray shipment. Incompatible with the colocated
  reward-model path (``_compute_reward_colocate`` re-reads ``raw_prompt`` from
  the queue) and with any registered reward manager that consumes
  ``raw_prompt``; the ``naive`` manager does not.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import numpy as np
import ray
import torch
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# ``AgentLoopWorkerTQ`` is exported as a ray actor class, which cannot be
# subclassed directly; the plain class carrying its methods lives here.
_AgentLoopWorkerTQBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class


def _strip_enabled() -> bool:
    return os.environ.get("VERL_OMNI_TQ_STRIP_RAW_PROMPT", "0") == "1"


# Fields whose only consumers run before the TransferQueue put (reward scoring
# from live kwargs; the rollout engine consumes mm_processor_output at generate
# time via llm_server), so storing them per session row only inflates every
# later worker fetch. ``multi_modal_data`` is already popped upstream.
_DEAD_ROW_FIELDS = ("raw_prompt", "mm_processor_output")


def _mmi_cache_enabled() -> bool:
    return os.environ.get("VERL_OMNI_TQ_CACHE_MMI", "0") == "1"


_MMI_CACHE_MAX_ENTRIES = int(os.environ.get("VERL_OMNI_TQ_CACHE_MMI_ENTRIES", "32"))


def _estimate_payload_bytes(value) -> int:
    """Rough serialized-size estimate for one kwarg value (for A/B logging)."""
    if isinstance(value, dict):
        return sum(_estimate_payload_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_payload_bytes(v) for v in value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (str, bytes)):
        return len(value)
    return 0


class _AVLeanAgentLoopWorkerTQ(_AgentLoopWorkerTQBase):
    """Agent-loop worker that stops shipping dead media bytes through TQ/Ray."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # prompt-token-ids -> processor output; sessions of one prompt share it.
        self._mmi_cache: OrderedDict[tuple, dict] = OrderedDict()

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict:
        """Processor-output cache shared across the ``rollout.n`` session rows.

        The TQ postprocess and the async-reward path each call this once per
        session with identical media; only the trailing response tokens differ,
        and the processor output carries media-derived tensors only (text-side
        ``input_ids``/``attention_mask`` are popped upstream), so keying on the
        prompt ids makes sessions of one prompt share one processor run. The
        call is synchronous on the worker's event loop, so without the cache
        one prompt costs ``n`` serialized processor runs.
        """
        if not _mmi_cache_enabled() or output.multi_modal_data is None or not len(output.prompt_ids):
            return super()._compute_multi_modal_inputs(output, input_ids)
        key = (tuple(output.prompt_ids),)
        cached = self._mmi_cache.get(key)
        if cached is not None:
            self._mmi_cache.move_to_end(key)
            # Downstream pops keys (e.g. mm_token_type_ids) from the dict, so
            # every consumer gets its own shallow copy; tensors stay shared.
            return dict(cached)
        computed = super()._compute_multi_modal_inputs(output, input_ids)
        self._mmi_cache[key] = dict(computed)
        while len(self._mmi_cache) > _MMI_CACHE_MAX_ENTRIES:
            self._mmi_cache.popitem(last=False)
        return computed

    async def _compute_score(self, outputs, kwargs: dict) -> None:
        # The async-reward shipment duplicates every kwarg n times per sample;
        # drop the media-carrying one when enabled (rule rewards never read it).
        if _strip_enabled() and "raw_prompt" in kwargs:
            saved = _estimate_payload_bytes(kwargs["raw_prompt"]) * len(outputs)
            kwargs = {k: v for k, v in kwargs.items() if k != "raw_prompt"}
            logger.info("[AVLean] reward shipment: dropped raw_prompt, ~%.1f MB saved", saved / 1e6)
        await super()._compute_score(outputs, kwargs=kwargs)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> None:
        if _strip_enabled():
            dead = {name: kwargs[name] for name in _DEAD_ROW_FIELDS if name in kwargs}
            if getattr(output, "mm_processor_output", None) is not None:
                dead["mm_processor_output(as_dict)"] = output.mm_processor_output
                # rides into the row via AgentLoopOutput.as_dict(); consumed
                # only by the rollout engine at generate time.
                output.mm_processor_output = None
            if dead:
                saved = sum(_estimate_payload_bytes(v) for v in dead.values())
                logger.info(
                    "[AVLean] TQ rows for %s: dropping %s, ~%.1f MB not stored per session row",
                    kwargs.get("uid", "?"),
                    sorted(dead),
                    saved / 1e6,
                )
                kwargs = {k: v for k, v in kwargs.items() if k in set(kwargs) - set(_DEAD_ROW_FIELDS)}
        await super()._agent_loop_postprocess(output, validate, **kwargs)


AVLeanAgentLoopWorkerTQ = ray.remote(_AVLeanAgentLoopWorkerTQ)


class AVLeanAgentLoopManagerTQ(AgentLoopManagerTQ):
    """Manager wiring :class:`AVLeanAgentLoopWorkerTQ` as its Ray worker class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AgentLoopManagerTQ.__init__ assigns the upstream worker class
        # unconditionally (bypassing the base class's hasattr guard), so the
        # override must happen after it; workers are only created later in
        # create() -> _init_agent_loop_workers().
        self.agent_loop_workers_class = AVLeanAgentLoopWorkerTQ
