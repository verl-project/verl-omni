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

"""Qwen-Image rollout-side adapter for the MixGRPO algorithm.

MixGRPO uses the same SDE machinery as FlowGRPO; the algorithmic difference
is purely in *how the SDE window is positioned*:

* ``random`` — draws a window start seeded by ``sde_window_seed + global_steps``
  so the position varies each training iteration while remaining consistent
  across all rollout ranks.  When ``sde_window_seed`` is absent the base
  :class:`QwenImagePipelineWithLogProb` per-call random draw is used instead.
* ``progressive`` — slides the window deterministically as a function of the
  trainer's current ``global_steps``, advancing by ``sde_window_size`` every
  ``iters_per_group`` training iterations. We materialise the deterministic
  start by collapsing the random draw range to a single value before
  delegating to the base ``forward``.

The trainer-side state we depend on (``global_steps``) is forwarded by the
diffusion agent loop into ``sampling_params.extra_args``; the MixGRPO
knobs (``sample_strategy``, ``iters_per_group``, ``sde_window_seed``) live on
:class:`DiffusionRolloutAlgoConfig`.
"""

from __future__ import annotations

import random as _random
from typing import Any

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

__all__ = ["QwenImageMixGRPOPipelineWithLogProb"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="mix_grpo")
class QwenImageMixGRPOPipelineWithLogProb(QwenImagePipelineWithLogProb):
    """Rollout pipeline for Qwen-Image with the MixGRPO algorithm."""

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Fix the SDE window before step-execution ``prepare_encode`` draws it.

        In step-execution mode ``forward()`` is never called, so
        ``_maybe_make_progressive_window`` would never run.  Calling it here,
        against ``state.sampling.extra_args``, ensures that all rollouts in a
        batch receive the same deterministic / seeded window regardless of
        whether the pipeline runs in full-forward or step-execution mode.
        """
        if state.sampling is not None:
            if state.sampling.extra_args is None:
                state.sampling.extra_args = {}
            self._maybe_make_progressive_window(state.sampling.extra_args, kwargs)
        return super().prepare_encode(state, **kwargs)

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch, **kwargs: Any):
        if isinstance(req, DiffusionRequestBatch):
            for sampling_params in req.sampling_params_list:
                self._maybe_make_progressive_window(sampling_params.extra_args, kwargs)
        else:
            self._maybe_make_progressive_window(req.sampling_params.extra_args, kwargs)
        return super().forward(req, **kwargs)

    @staticmethod
    def _maybe_make_progressive_window(extra: dict[str, Any], kwargs: dict[str, Any]) -> None:
        """Mutate *extra* (``sampling_params.extra_args``) in place to fix the
        SDE window start position.

        * ``progressive``: deterministic from ``global_steps``.
        * ``random`` with ``sde_window_seed`` present: seeded per-step draw so
          all ranks agree on the same window position for each training step.
        * Otherwise: no-op -- the base pipeline's per-call random draw is used.
        """
        strategy = extra.get("sample_strategy", "random")
        size = extra.get("sde_window_size") or kwargs.get("sde_window_size")

        if strategy == "random":
            seed = extra.get("sde_window_seed")
            if seed is None or size is None:
                return  # no-op: base pipeline handles the random draw
            envelope = extra.get("sde_window_range") or kwargs.get("sde_window_range") or (0, 5)
            envelope_start, envelope_end = int(envelope[0]), int(envelope[1])
            size = int(size)
            max_start = envelope_end - size
            if max_start < envelope_start:
                raise ValueError(
                    f"MixGRPO random window does not fit: "
                    f"sde_window_range={[envelope_start, envelope_end]}, sde_window_size={size}."
                )
            global_steps = int(extra.get("global_steps", 0))
            start = _random.Random(int(seed) + global_steps).randint(envelope_start, max_start)
            extra["sde_window_range"] = [start, start + size]
            return

        if strategy != "progressive":
            return

        if size is None:
            # No SDE window configured -- nothing to slide.
            return

        envelope = extra.get("sde_window_range") or kwargs.get("sde_window_range") or (0, 5)
        envelope_start, envelope_end = int(envelope[0]), int(envelope[1])
        size = int(size)
        max_start = envelope_end - size
        if max_start < envelope_start:
            raise ValueError(
                f"MixGRPO progressive window does not fit: "
                f"sde_window_range={[envelope_start, envelope_end]}, sde_window_size={size}."
            )

        global_steps = int(extra.get("global_steps", 0))
        iters_per_group = max(1, int(extra.get("iters_per_group", 1)))
        n_advances = max(0, global_steps) // iters_per_group
        start = min(envelope_start + n_advances * size, max_start)

        # Collapse the base pipeline's per-call ``torch.randint(low, low+1)``
        # to a single value by shrinking the envelope to ``[start, start+size]``.
        extra["sde_window_range"] = [start, start + size]
