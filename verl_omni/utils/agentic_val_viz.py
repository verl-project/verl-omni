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

"""Fixed holdout tasks for agentic validation visualization.

These are experiment-tracking fixtures, not part of the agent-loop protocol.
On each validate step the metrics manager runs this holdout *first* (samples
9001/9002 cafe + 9003/9004 CN poster by default), writes PNGs and ``meta.json``
under ``outputs/e2e/<run>/rollout_images/``, then evaluates the UniCoT val set.
The trainer driver logs ``val-core`` at the same step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ValVizCase:
    """One fixed validation visualization sample."""

    sample_index: int
    viz_id: str
    task_type: str
    system_prompt: str
    user_request: str
    expected_num_images: int = 1
    reference_subtasks: tuple[str, ...] | None = None


class AgenticValVizProvider:
    """Build a fixed DataProto batch for holdout validation visualization."""

    def __init__(self, cases: Sequence[ValVizCase]):
        if not cases:
            raise ValueError("AgenticValVizProvider requires at least one ValVizCase")
        self.cases = list(cases)

    def build_batch(self, step, *, eos_token_id: int | None, pad_token_id: int | None) -> Any:
        """Construct a validation DataProto for the configured holdout cases."""
        from verl import DataProto

        non_tensor: dict[str, list[Any]] = {
            key: [] for key in ("raw_prompt", "index", "data_source", "reward_model", "extra_info")
        }
        for case in self.cases:
            ground_truth: dict[str, Any] = {
                "user_request": case.user_request,
                "task_type": case.task_type,
                "expected_num_images": case.expected_num_images,
            }
            if case.reference_subtasks:
                ground_truth["reference_subtasks"] = list(case.reference_subtasks)
            non_tensor["raw_prompt"].append(
                [
                    {"role": "system", "content": case.system_prompt},
                    {"role": "user", "content": case.user_request},
                ]
            )
            non_tensor["index"].append(case.sample_index)
            non_tensor["data_source"].append("agentic_val_viz")
            non_tensor["reward_model"].append({"style": "rule", "ground_truth": ground_truth})
            non_tensor["extra_info"].append(
                {
                    "viz_id": case.viz_id,
                    "task_type": case.task_type,
                    "expected_num_images": case.expected_num_images,
                    "raw_prompt": case.user_request,
                }
            )
        batch = DataProto.from_single_dict({key: np.array(value, dtype=object) for key, value in non_tensor.items()})
        batch.meta_info = {
            "global_steps": step,
            "validate": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
        }
        return batch


#: Plan holdouts need a reference decomposition. These are written as plain content parts
#: (no "keep the outline unchanged and edit …" lead-in), and plan mode sends the whole
#: list as one prompt, so ``delta_subtasks`` only strips framing and keeps each part.
_CAFE_POSTER_SUBTASKS = (
    "A vertical cafe poster rendering: a rustic wooden table lit by soft morning sunlight "
    "through a nearby window, with gentle steam rising from a warm-toned ceramic coffee cup "
    "centered on the table. Warm amber and brown color grading, shallow depth of field, "
    "cozy aesthetic.",
    'A bold serif headline reading "ARTISAN ROAST" set across the top of the poster in '
    "cream lettering on a deep roast-brown band, with generous margins and a thin gold rule "
    "beneath it.",
    'A bottom text block reading "Freshly Brewed Daily — Open at 7 AM" in a smaller clean '
    "sans-serif, centered with even spacing beneath the illustration, completing the poster.",
)

_CN_POSTER_SUBTASKS = (
    "一张垂直构图的平面设计海报，纯色鲜艳宝蓝背景；下半部分由一幅巨大的插画风格老虎占据，"
    "老虎趴着面向观众，黄色眼睛，皮毛由橙、黑、白三色构成；头顶漂浮一个含红色爱心的思想泡泡；"
    "整体风格俏皮、超现实、刻意荒诞。",
    "海报顶部为巨大无衬线字体主标题：上半部分浅灰色 “Sofa Montain Slummerfest”，"
    "下半部分白色 “Annual Napping Festival 2025”；其下方是巨大的黑色书法字体中文标题 "
    "“沙发山打呼节”；严肃主标题与荒诞细节形成强烈对比。",
    "海报布满详尽活动文字：左栏白色字体列出猫主题乐队名 “The Fluffy Paws Grumbers (毛爪咕噜)”、"
    "“DJ Meow Mix”、“九命怪猫 (Nine Lives)”、“激光笔追逐者 (The Laser Dots)”、"
    "“纸箱爱好者 (Cardbock Box Lovers)”、“呼噜神教 (The Purr-fectionists)”、"
    "“猫草成瘾者 (The Catnip Junkies)”、“DJ Chairman Meow (猫主席)”、"
    "“Varh Radator Fesidenl Paw-Five”；右栏活动细节含滑稽拼写错误 “4/1 MONDAY SUNL SUNSET”、"
    "“上海市浦东新区猫抓板路1号顶楼阳台”、“ADV. 1 CAN OF TUNA, DOOR 2 CANS, KITTENS FREE!”；"
    "最底部一排虚构赞助商标志 “Catberd”、“好主人罐罐有限公司 (Good Oinar Canned Food Ltd)”、"
    "“iNONEPAWS”。",
)


def _plan_references(subtasks: tuple[str, ...]) -> tuple[str, ...]:
    """Return the per-part plan references for a holdout decomposition."""
    from verl_omni.utils.agentic.plan_protocol import delta_subtasks

    return delta_subtasks(subtasks)


def _cafe_poster_cases() -> list[ValVizCase]:
    """UniCoT reflect/plan cafe-poster holdout used by the RPCO e2e recipe."""
    from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl

    task = (
        'A vertical artistic cafe poster. The headline at the top reads "ARTISAN ROAST". '
        "The center features a detailed, warm-toned illustration of a ceramic coffee cup sitting "
        "on a rustic wooden table with soft steam rising and gentle morning sunlight coming through "
        'a nearby window. Surrounding text at the bottom reads "Freshly Brewed Daily — Open at 7 AM". '
        "Cozy, warm amber and brown color grading, shallow depth of field, cozy aesthetic."
    )
    user_text = build_unicot_agentic_rl._with_brevity(task)
    return [
        ValVizCase(
            sample_index=9001,
            viz_id="reflect_prompt",
            task_type="reflect",
            system_prompt=build_unicot_agentic_rl.REFLECT_SYSTEM_PROMPT,
            user_request=user_text,
            expected_num_images=1,
        ),
        ValVizCase(
            sample_index=9002,
            viz_id="plan_prompt",
            task_type="plan",
            system_prompt=build_unicot_agentic_rl.PLAN_SYSTEM_PROMPT,
            user_request=user_text,
            expected_num_images=len(_CAFE_POSTER_SUBTASKS),
            reference_subtasks=_plan_references(_CAFE_POSTER_SUBTASKS),
        ),
    ]


_CN_POSTER_TASK = (
    "一张垂直构图的平面设计海报，背景是纯粹而鲜艳的宝蓝色。"
    "顶部的巨大无衬线字体主标题，上半部分为浅灰色的“Sofa Montain Slummerfest”，"
    "下半部分为白色的“Annual Napping Festival 2025”。"
    "其下方是巨大的黑色书法字体中文标题“沙发山打呼节”。"
    "海报的下半部分由一幅巨大的、插画风格的老虎插画占据，它正趴着面向观众，眼睛是黄色的。"
    "其皮毛由橙、黑、白三色构成。一个包含红色爱心的思想泡泡漂浮在它的头顶。"
    "海报上布满了详尽的活动文字。左栏用白色字体列出了以猫为主题的乐队名，"
    "如“The Fluffy Paws Grumbers (毛爪咕噜)”、“DJ Meow Mix”、“九命怪猫 (Nine Lives)”、"
    "“激光笔追逐者 (The Laser Dots)”、“纸箱爱好者 (Cardbock Box Lovers)”、"
    "“呼噜神教 (The Purr-fectionists)”、“猫草成瘾者 (The Catnip Junkies)”、"
    "“DJ Chairman Meow (猫主席)”以及像“Varh Radator Fesidenl Paw-Five”这样的无意义短语。"
    "右栏列出了活动细节，其中许多都带有滑稽的拼写错误或无意义内容，"
    "包括日期“4/1 MONDAY SUNL SUNSET”、地点“上海市浦东新区猫抓板路1号顶楼阳台”"
    "以及票务信息如“ADV. 1 CAN OF TUNA, DOOR 2 CANS, KITTENS FREE!”。"
    "在最底部是一排虚构的赞助商标志，名称包括“Catberd”、"
    "“好主人罐罐有限公司 (Good Oinar Canned Food Ltd)”和“iNONEPAWS”。"
)


def _cn_poster_cases() -> list[ValVizCase]:
    """UniCoT reflect/plan Chinese napping-festival poster holdout (9003/9004)."""
    from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl

    user_text = build_unicot_agentic_rl._with_brevity(_CN_POSTER_TASK)
    return [
        ValVizCase(
            sample_index=9003,
            viz_id="reflect_prompt_cn",
            task_type="reflect",
            system_prompt=build_unicot_agentic_rl.REFLECT_SYSTEM_PROMPT,
            user_request=user_text,
            expected_num_images=1,
        ),
        ValVizCase(
            sample_index=9004,
            viz_id="plan_prompt_cn",
            task_type="plan",
            system_prompt=build_unicot_agentic_rl.PLAN_SYSTEM_PROMPT,
            user_request=user_text,
            expected_num_images=len(_CN_POSTER_SUBTASKS),
            reference_subtasks=_plan_references(_CN_POSTER_SUBTASKS),
        ),
    ]


def _default_holdout_cases() -> list[ValVizCase]:
    """Cafe 9001/9002 + CN poster 9003/9004."""
    return _cafe_poster_cases() + _cn_poster_cases()


def resolve_agentic_val_viz_provider() -> AgenticValVizProvider | None:
    """Return the fixed holdout provider when ``AGENTIC_VAL_VIZ=1``."""
    if os.getenv("AGENTIC_VAL_VIZ", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    return AgenticValVizProvider(_default_holdout_cases())
