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
training legs only need the token/multimodal-input fields).

Select with::

    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.av_agent_loop_tq.AVLeanAgentLoopManagerTQ

Environment gates (all default off, i.e. bit-identical upstream behavior):

- ``VERL_OMNI_TQ_STRIP_RAW_PROMPT=1`` — omit ``raw_prompt`` from TQ rows and
  from the async-reward Ray shipment. Incompatible with the colocated
  reward-model path (``_compute_reward_colocate`` re-reads ``raw_prompt`` from
  the queue) and with any registered reward manager that consumes
  ``raw_prompt``; the ``naive`` manager does not.
- ``VERL_OMNI_TQ_CACHE_MMI=1`` (cap: ``VERL_OMNI_TQ_CACHE_MMI_ENTRIES``,
  default 32) — share one HF processor run across the ``rollout.n`` sessions
  of a prompt, keyed by prompt ids plus the media payload identity.
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


# The one kwarg whose only consumers run before the TransferQueue put (reward
# scoring from live kwargs; the rollout engine consumes mm_processor_output at
# generate time via llm_server), so storing it per session row only inflates
# every later worker fetch. ``multi_modal_data`` is already popped upstream.
_DEAD_ROW_FIELDS = ("raw_prompt",)


def _mmi_cache_enabled() -> bool:
    return os.environ.get("VERL_OMNI_TQ_CACHE_MMI", "0") == "1"


_MMI_CACHE_MAX_ENTRIES = int(os.environ.get("VERL_OMNI_TQ_CACHE_MMI_ENTRIES", "32"))


def _estimate_payload_bytes(value) -> int:
    """Rough serialized-size estimate for one kwarg value (for A/B logging)."""
    if isinstance(value, dict):
        return sum(_estimate_payload_bytes(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(_estimate_payload_bytes(v) for v in value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, str | bytes):
        return len(value)
    return 0


class _AVLeanAgentLoopWorkerTQ(_AgentLoopWorkerTQBase):
    """Agent-loop worker that stops shipping dead media bytes through TQ/Ray."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (prompt ids, media identity) -> processor output; sessions of one
        # prompt share it.
        self._mmi_cache: OrderedDict[tuple, dict] = OrderedDict()

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict:
        """Processor-output cache shared across the ``rollout.n`` session rows.

        The TQ postprocess and the async-reward path each call this once per
        session with identical media; only the trailing response tokens differ,
        and the processor output carries media-derived tensors only (text-side
        ``input_ids``/``attention_mask`` are popped upstream), so keying on the
        prompt ids plus the media object identity makes sessions of one prompt
        share one processor run. The call is synchronous on the worker's event
        loop, so without the cache one prompt costs ``n`` serialized processor
        runs.
        """
        if not _mmi_cache_enabled() or output.multi_modal_data is None or not len(output.prompt_ids):
            return super()._compute_multi_modal_inputs(output, input_ids)
        # The media object completes the key: prompt tokens alone collide
        # across prompts that embed different (same-sized) media. Pinning the
        # media object inside the cached value keeps its id() unique for as
        # long as the entry lives, so a recycled id can never hit a stale one.
        key = (tuple(output.prompt_ids), id(output.multi_modal_data))
        cached = self._mmi_cache.get(key)
        if cached is not None:
            self._mmi_cache.move_to_end(key)
            # Downstream pops keys (e.g. mm_token_type_ids) from the dict, so
            # every consumer gets its own shallow copy; tensors stay shared.
            return dict(cached["mmi"])
        computed = super()._compute_multi_modal_inputs(output, input_ids)
        self._mmi_cache[key] = {"mmi": dict(computed), "media": output.multi_modal_data}
        while len(self._mmi_cache) > _MMI_CACHE_MAX_ENTRIES:
            self._mmi_cache.popitem(last=False)
        return computed

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> None:
        # Single interception point: upstream passes these kwargs both to the
        # async-reward shipment (_compute_score) and into every TQ session row.
        if _strip_enabled():
            outputs = output if isinstance(output, list) else [output]
            # ``raw_prompt`` rides the kwargs into every row; each element's
            # ``mm_processor_output`` rides its as_dict() — consumed only by
            # the rollout engine at generate time.
            dropped_labels = []
            saved = 0
            if "raw_prompt" in kwargs:
                dropped_labels.append("raw_prompt")
                saved += _estimate_payload_bytes(kwargs["raw_prompt"]) * len(outputs)
            for element in outputs:
                mm_output = getattr(element, "mm_processor_output", None)
                if mm_output is not None:
                    dropped_labels.append("mm_processor_output(as_dict)")
                    saved += _estimate_payload_bytes(mm_output)
                    element.mm_processor_output = None
            if dropped_labels:
                logger.info(
                    "[AVLean] TQ rows for %s: dropping %s, ~%.1f MB not stored",
                    kwargs.get("uid", "?"),
                    sorted(set(dropped_labels)),
                    saved / 1e6,
                )
                kwargs = {k: v for k, v in kwargs.items() if k not in _DEAD_ROW_FIELDS}
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
