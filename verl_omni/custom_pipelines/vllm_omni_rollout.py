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
vLLM-Omni rollout pipeline exports. ``QwenImagePipelineWithLogProb`` is defined
in ``qwen_image.vllm_omni_rollout_adapter`` and requires the optional
``vllm_omni`` package. This module uses lazy attribute loading so that
``verl_omni`` can be imported without vllm-omni installed.
"""

__all__ = ["QwenImagePipelineWithLogProb"]  # noqa: F822


def __getattr__(name: str):
    if name == "QwenImagePipelineWithLogProb":
        from .qwen_image.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

        return QwenImagePipelineWithLogProb
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
