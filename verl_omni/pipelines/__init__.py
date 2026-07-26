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

import importlib
import logging

logger = logging.getLogger(__name__)

_MODULE_NAMES = (
    "bagel_flow_grpo",
    "boogu_image_diffusion_nft",
    "boogu_image_flow_grpo",
    "qwen3_omni",
    "qwen_image_diffusion_nft",
    "qwen_image_dpo",
    "qwen_image_edit_flow_grpo",
    "qwen_image_flow_grpo",
    "qwen_image_mix_grpo",
    "sd3_dpo",
    "sd3_flow_grpo",
    "wan22_dance_grpo",
)

__all__: list[str] = []

for _module_name in _MODULE_NAMES:
    try:
        _module = importlib.import_module(f"{__name__}.{_module_name}")
    except (ModuleNotFoundError, ImportError) as exc:
        logger.warning("Skipping optional pipeline module %s due to import failure: %s", _module_name, exc)
        continue

    globals()[_module_name] = _module
    for _export_name in getattr(_module, "__all__", []):
        globals()[_export_name] = getattr(_module, _export_name)
        __all__.append(_export_name)
