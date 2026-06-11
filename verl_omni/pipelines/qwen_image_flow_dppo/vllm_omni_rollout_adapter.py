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
"""Qwen-Image vLLM-Omni rollout adapter registration for Flow-DPPO."""

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

__all__ = ["QwenImageFlowDPPOPipelineWithLogProb"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_dppo")
class QwenImageFlowDPPOPipelineWithLogProb(QwenImagePipelineWithLogProb):
    """Flow-DPPO reuses FlowGRPO's stochastic trajectory rollout contract."""
