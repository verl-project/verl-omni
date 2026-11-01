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
"""Every V1 OCR judge recipe bounds the judge's output: the reward rollout response_length is its max_tokens."""

import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"
OCR_RECIPES = sorted(
    p
    for p in EXAMPLES.rglob("*.sh")
    if "compute_score_ocr" in (text := p.read_text()) and ("main_diffusion_v1" in text or "use_v1=true" in text)
)
BUDGET = re.compile(r"reward\.(?:reward_model|models\.[A-Za-z_]+)\.rollout\.response_length=256")


def test_ocr_recipes_exist():
    assert len(OCR_RECIPES) >= 12


@pytest.mark.parametrize("recipe", OCR_RECIPES, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_ocr_recipe_bounds_the_judge_output(recipe):
    assert BUDGET.search(recipe.read_text()), f"{recipe} does not set the OCR judge response_length to 256"
