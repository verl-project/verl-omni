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
import os

from .diffusers_training_adapter import Wan22DanceGRPO

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from .vllm_omni_rollout_adapter import Wan22DanceGRPOPipelineWithLogProb
except (ImportError, RuntimeError, AttributeError) as e:
    logger.info(f"Wan22 Dance GRPO not available: {e}. GPU/NPU required.")

    class _UnavailableModule:
        def __getattr__(self, _):
            raise RuntimeError("Wan22 Dance GRPO requires GPU (CUDA/NPU)")

        def __call__(self, *args, **kwargs):
            raise RuntimeError("Wan22 Dance GRPO requires GPU (CUDA/NPU)")

    Wan22DanceGRPOPipelineWithLogProb = _UnavailableModule()

__all__ = ["Wan22DanceGRPO", "Wan22DanceGRPOPipelineWithLogProb"]
