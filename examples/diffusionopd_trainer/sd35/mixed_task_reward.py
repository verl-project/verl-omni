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
"""Per-task reward dispatch for the mixed OCR + PickScore distillation recipe.

OCR rows go to the GenRM OCR scorer through the reward router; PickScore rows
are scored locally with PickScore_v1 on ``PICKSCORE_DEVICE``. Both branches
return the same key set: the reward loop merges per-sample dicts into one
table and requires identical keys on every row.
"""

import os

from verl.utils.device import get_device_name

from verl_omni.utils.reward_score.genrm_ocr import compute_score_ocr
from verl_omni.utils.reward_score.pickscore_reward import compute_score_pickscore

_PICKSCORE_DEVICE = os.environ.get("PICKSCORE_DEVICE", get_device_name())


async def compute_score_mixed(data_source, solution_image, ground_truth, extra_info, **kwargs):
    """Route one sample to the scorer of its task, keyed on ``data_source``."""
    if "pickscore" in data_source:
        result = await compute_score_pickscore(
            data_source, solution_image, ground_truth, extra_info, device=_PICKSCORE_DEVICE
        )
        return {"score": result["score"], "pickscore_raw": result["pickscore_raw"], "genrm_response": ""}
    result = await compute_score_ocr(data_source, solution_image, ground_truth, extra_info, **kwargs)
    return {"score": result["score"], "pickscore_raw": float("nan"), "genrm_response": result.get("genrm_response", "")}
