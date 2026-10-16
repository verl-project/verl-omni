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
"""Qwen-Image training adapter for DiffusionART."""

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_diffusion_nft.diffusers_training_adapter import QwenImageDiffusionNFT

__all__ = ["QwenImageDiffusionART"]


@DiffusionModelBase.register("QwenImagePipeline", algorithm="diffusion_art")
class QwenImageDiffusionART(QwenImageDiffusionNFT):
    """Single prediction-space Qwen-Image forward used by DiffusionART.

    The inputs are built exactly as for DiffusionNFT; only the latent handed in differs
    (a cached ODE rollout state instead of a re-noised endpoint), which the
    ``diffusion_art_model`` engine takes care of.
    """
