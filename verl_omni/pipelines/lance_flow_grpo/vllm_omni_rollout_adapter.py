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


"""Lance FlowGRPO rollout using BAGEL's SDE and trajectory machinery."""

from __future__ import annotations

from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline

from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import LANCE_TIMESTEP_SHIFT, setup_lance_sigmas


@VllmOmniPipelineBase.register("OmniLanceForConditionalGeneration", algorithm="flow_grpo")
class LancePipelineWithLogProb(BagelPipelineWithLogProb, LancePipeline):
    """Lance pipeline variant for RL rollouts with verl-omni.

    ``LancePipeline.__init__`` deliberately does not call up the MRO - it
    replaces BAGEL's construction wholesale - so ``BagelPipelineWithLogProb``
    still reaches it through ``super().__init__`` and installs the SDE
    scheduler afterwards.
    """

    #: Lance's rollout default (``LANCE_DEFAULTS.timestep_shift``).
    flowgrpo_timestep_shift: float = LANCE_TIMESTEP_SHIFT

    @classmethod
    def flowgrpo_setup_sigmas(cls, scheduler: FlowMatchSDEDiscreteScheduler, num_steps: int, shift: float) -> None:
        """Lance's schedule: one more point than BAGEL's, and a 3.5 shift."""
        setup_lance_sigmas(scheduler, num_steps, shift=shift)
