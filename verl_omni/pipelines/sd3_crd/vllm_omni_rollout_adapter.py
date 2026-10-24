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

"""CRD reuses the deterministic SD3 endpoint rollout contract."""

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.sd3_dgpo.vllm_omni_rollout_adapter import StableDiffusion3DGPOPipeline

__all__ = ["StableDiffusion3CRDPipeline"]


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="crd")
class StableDiffusion3CRDPipeline(StableDiffusion3DGPOPipeline):
    """ODE samples with clean latents for centered reward distillation."""

    supports_request_batch = True
    always_encode_negative_prompt = True
