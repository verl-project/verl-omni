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

"""Diffusion Model FLOPs Utilization (MFU) utilities."""

from verl_omni.utils.mfu import qwen_image  # noqa: F401 — register built-in architectures
from verl_omni.utils.mfu.diffusion_flops_counter import (
    DiffusionFlopsCounter,
    DiffusionModelFlops,
    get_device_peak_tflops,
    get_forward_passes_per_step,
    register_diffusion_architecture,
)
from verl_omni.utils.mfu.qwen_image import QwenImageFlops

__all__ = [
    "DiffusionModelFlops",
    "DiffusionFlopsCounter",
    "QwenImageFlops",
    "register_diffusion_architecture",
    "get_forward_passes_per_step",
    "get_device_peak_tflops",
]
