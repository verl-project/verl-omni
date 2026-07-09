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

import warnings
from importlib import import_module

__all__ = []


def _import_pipeline(module_name: str) -> None:
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name != "vllm":
            raise
        warnings.warn(
            f"Skipping {module_name} pipeline registration because optional dependency 'vllm' is not installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    globals()[module_name] = module
    for exported in getattr(module, "__all__", []):
        globals()[exported] = getattr(module, exported)
        __all__.append(exported)


for _module_name in (
    "bagel_flow_grpo",
    "qwen_image_diffusion_nft",
    "qwen_image_dpo",
    "qwen_image_flow_grpo",
    "qwen_image_mix_grpo",
    "qwen3_omni",
    "sd3_dpo",
    "sd3_flow_grpo",
    "wan22_dance_grpo",
):
    _import_pipeline(_module_name)
