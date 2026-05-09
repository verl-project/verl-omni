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

"""
Qwen-Image training-side adapter for MixGRPO algorithm.
Inherits model-specific forward/sampling behavior from FlowGRPO but provides
the sliding-window scheduler for MixGRPO.
"""

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["QwenImageMixGRPO"]


@DiffusionModelBase.register("QwenImagePipeline", algo="mix_grpo")
class QwenImageMixGRPO(QwenImage):
    """Training adapter for Qwen-Image with the MixGRPO algorithm.

    Inherits from the baseline :class:`QwenImage` adapter (FlowGRPO) for all
    diffusers model configurations, but overrides the algorithm scheduler
    creation to provide MixGRPO's sliding window strategies.
    """

    @classmethod
    def build_algo_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the sliding-window scheduler for MixGRPO.

        Args:
            model_config (DiffusionModelConfig): Configuration for the diffusion model,
                used to extract the MixGRPO algorithm configuration.

        Returns:
            SDEWindowScheduler: A sliding-window scheduler (e.g., MixGRPORandomScheduler).
        """
        from verl_omni.pipelines.schedulers.sde_window_scheduler import build_sde_window_scheduler

        num_inference_steps = int(model_config.pipeline.num_inference_steps)
        # Delegate to the algorithm scheduler factory
        return build_sde_window_scheduler(model_config.algo, num_inference_steps=num_inference_steps)
