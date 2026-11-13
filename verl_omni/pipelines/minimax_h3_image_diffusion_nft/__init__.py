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
import logging

from .diffusers_training_adapter import MiniMaxH3ImageDiffusionNFT

__all__ = ["MiniMaxH3ImageDiffusionNFT"]

_IMAGE_PIPELINE_MODULE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image"

try:
    from .vllm_omni_rollout_adapter import MiniMaxH3ImageDiffusionNFTPipeline  # noqa: F401
except (ImportError, AttributeError) as error:
    if isinstance(error, ModuleNotFoundError) and error.name not in (None, _IMAGE_PIPELINE_MODULE):
        if not _IMAGE_PIPELINE_MODULE.startswith(f"{error.name}."):
            raise
    logging.getLogger(__name__).warning(
        "MiniMaxH3ImagePipeline rollout adapter is not registered: installed vLLM-Omni lacks MiniMax H3 image "
        "support (%s).",
        error,
    )
else:
    __all__.append("MiniMaxH3ImageDiffusionNFTPipeline")
