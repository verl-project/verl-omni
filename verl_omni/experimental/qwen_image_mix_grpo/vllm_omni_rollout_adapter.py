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

"""Experimental step-wise batching execution support for MixGRPO.

This module provides :class:`QwenImageMixGRPOPipelineWithLogProbStepwise`, which
uses multiple inheritance to combine the MixGRPO window-positioning logic with
the stepwise-execution pipeline from FlowGRPO.

In step-execution mode ``forward()`` is never called, so the MixGRPO-specific
``_maybe_make_progressive_window`` would never run.  Calling it in
``prepare_encode`` ensures that all rollouts in a batch receive the same
deterministic / seeded window regardless of whether the pipeline runs in
full-forward or step-execution mode.
"""

from typing import Any

from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_mix_grpo.vllm_omni_rollout_adapter import QwenImageMixGRPOPipelineWithLogProb
from verl_omni.experimental.qwen_image_flow_grpo.vllm_omni_rollout_adapter import (
    QwenImagePipelineWithLogProbStepwise,
)

__all__ = ["QwenImageMixGRPOPipelineWithLogProbStepwise"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="mix_grpo_stepwise")
class QwenImageMixGRPOPipelineWithLogProbStepwise(
    QwenImageMixGRPOPipelineWithLogProb,
    QwenImagePipelineWithLogProbStepwise,
):
    """Experimental stepwise-execution variant of the MixGRPO rollout pipeline.

    Inherits from both :class:`QwenImageMixGRPOPipelineWithLogProb` (for the
    MixGRPO sliding-window logic) and
    :class:`QwenImagePipelineWithLogProbStepwise` (for pre-tokenized prompt
    handling and SDE trajectory collection in step-execution mode).

    Adds a ``prepare_encode`` override that calls
    ``_maybe_make_progressive_window`` before delegating via ``super()`` to
    :class:`QwenImagePipelineWithLogProbStepwise.prepare_encode`, ensuring the
    SDE window is fixed before the stepwise pipeline initialises its state.
    """

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Override to fix the SDE window before the stepwise prepare_encode draws it.

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
