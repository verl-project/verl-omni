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

"""Fixed café (9001/9002) and Chinese-poster (9003/9004) validation demos.

The V1 worker runs these separately from queued validation and persists their
images under ``rollout_images/val/step_XXXXXX/sample_9001..9004.nn/``. They do not
contribute training rows or alter the validation aggregate.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Per-row marker column. A prompt carrying this key is a holdout-viz prompt, and the
#: TransferQueue worker runs its episode **without** writing any TransferQueue row --
#: a holdout is an artifact producer, not a training or validation row.
VAL_VIZ_PROMPT_KEY = "agentic_val_viz"

#: Dataset ``data_source`` stamped on every holdout row.
VAL_VIZ_DATA_SOURCE = "agentic_val_viz"


def save_val_viz_images(image_paths: Sequence[str], relpath: str) -> list[str]:
    """Keep demo PNGs under their validation step and fixed case identity."""
    from verl_omni.utils.agentic.artifacts import resolve_rollout_images_root

    written = []
    try:
        target_dir = resolve_rollout_images_root() / "val" / relpath
        for index, path in enumerate(image_paths):
            source = Path(path)
            if not source.is_file():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"image_{index:03d}.png"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            written.append(str(target))
    except OSError:
        logging.getLogger(__name__).warning("Could not save validation demo %s", relpath, exc_info=True)
    return written


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


def _raw_prompt_for_case(case: ValVizCase) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": case.system_prompt},
        {"role": "user", "content": case.user_request},
    ]


def _ground_truth_for_case(case: ValVizCase) -> dict[str, Any]:
    ground_truth: dict[str, Any] = {
        "user_request": case.user_request,
        "task_type": case.task_type,
        "expected_num_images": case.expected_num_images,
    }
    if case.reference_subtasks:
        ground_truth["reference_subtasks"] = list(case.reference_subtasks)
    return ground_truth


def _extra_info_for_case(case: ValVizCase) -> dict[str, Any]:
    return {
        "viz_id": case.viz_id,
        "task_type": case.task_type,
        "expected_num_images": case.expected_num_images,
        "raw_prompt": case.user_request,
    }


def _object_column(values: Sequence[Any]) -> np.ndarray:
    """A 1-D object array holding ``values`` as-is, one element per row.

    ``np.array(values, dtype=object)`` is not that: equal-length message lists collapse
    into an ``(rows, messages)`` matrix, so each row's ``raw_prompt`` reaches the agent loop
    as an ``ndarray`` of dicts. ``_messages_are_text_only`` accepts only a ``list``, so the
    Hermes ``<tools>`` block was never injected and the UND model, told about
    ``generate_image`` in prose only, wrote "I have generated an image" instead of calling
    it: ``num_gen_calls=0`` on all four holdout episodes, hence no PNGs.
    """
    column = np.empty(len(values), dtype=object)
    column[:] = list(values)
    return column


class AgenticValVizProvider:
    """Build a fixed batch of holdout validation-visualization prompts."""

    def __init__(self, cases: Sequence[ValVizCase]):
        if not cases:
            raise ValueError("AgenticValVizProvider requires at least one ValVizCase")
        self.cases = list(cases)

    def _column_dict(self) -> dict[str, list[Any]]:
        non_tensor: dict[str, list[Any]] = {
            key: [] for key in ("raw_prompt", "index", "data_source", "reward_model", "extra_info")
        }
        for case in self.cases:
            non_tensor["raw_prompt"].append(_raw_prompt_for_case(case))
            non_tensor["index"].append(case.sample_index)
            non_tensor["data_source"].append(VAL_VIZ_DATA_SOURCE)
            non_tensor["reward_model"].append({"style": "rule", "ground_truth": _ground_truth_for_case(case)})
            non_tensor["extra_info"].append(_extra_info_for_case(case))
        return non_tensor

    def build_batch(self, step, *, eos_token_id: int | None, pad_token_id: int | None) -> Any:
        """Construct a validation ``DataProto`` for the configured holdout cases.

        Args:
            step: Global step to stamp on the batch.
            eos_token_id: Sampler EOS id (forwarded into ``meta_info``).
            pad_token_id: Sampler pad id (forwarded into ``meta_info``).

        Returns:
            ``DataProto`` with ``meta_info["validate"] = True``.
        """
        from verl import DataProto

        non_tensor = self._column_dict()
        batch = DataProto.from_single_dict({key: _object_column(value) for key, value in non_tensor.items()})
        batch.meta_info = {
            "global_steps": step,
            "validate": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
        }
        return batch

    def build_tensordict_batch(self, step, *, uid_prefix: str = "val-viz") -> Any:
        """Construct a holdout ``TensorDict`` for the V1 TransferQueue lane.

        The two identities are deliberately split:

        * ``uid`` -- the TransferQueue key. It must be unique per step (a repeated key
          would overwrite the previous step's row), so it carries the step.
        * ``dataset_task_uid`` -- the artifact-folder name. ``episode_artifact_keys``
          resolves ``dataset_task_uid`` *before* ``uid`` and passes it to
          ``build_trajectory_relpath``, so pinning it to the holdout's ``sample_index``
          makes the dump land at ``rollout_images/val/step_XXXXXX/sample_9001.01/``
          instead of an opaque uuid folder -- the Mode (2a) poster name.

        Args:
            step: Global step to stamp on the batch (and into the uid).
            uid_prefix: Namespace for the TransferQueue keys.

        Returns:
            ``TensorDict`` carrying ``raw_prompt``/``index``/``uid``/``dataset_task_uid``
            plus the ``global_steps`` / ``validate`` / :data:`VAL_VIZ_PROMPT_KEY`
            metadata the TQ worker reads.
        """
        from verl.utils.tensordict_utils import assign_non_tensor_data, get_tensordict

        non_tensor = self._column_dict()
        batch_dict: dict[str, Any] = {key: _object_column(value) for key, value in non_tensor.items()}
        step_tag = str(step) if step is not None else "unknown"
        batch_dict["uid"] = np.array(
            [f"{uid_prefix}-{step_tag}-{case.sample_index}" for case in self.cases],
            dtype=object,
        )
        batch_dict["dataset_task_uid"] = np.array(
            [str(case.sample_index) for case in self.cases],
            dtype=object,
        )
        batch = get_tensordict(batch_dict)
        assign_non_tensor_data(batch, "global_steps", step)
        assign_non_tensor_data(batch, "validate", True)
        # Batch-level, exactly like ``validate``: the only batch this provider builds is a
        # holdout batch, and a non-tensor scalar survives ``TensorDict.chunk``, so every
        # row the workers see carries the marker. It is what makes the worker run the
        # episode without writing a TransferQueue row (``is_val_viz_prompt``).
        assign_non_tensor_data(batch, VAL_VIZ_PROMPT_KEY, "1")
        return batch


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
            expected_num_images=1,
            reference_subtasks=(task,),
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
            expected_num_images=1,
            reference_subtasks=(_CN_POSTER_TASK,),
        ),
    ]


def _default_holdout_cases() -> list[ValVizCase]:
    """Cafe 9001/9002 + CN poster 9003/9004."""
    return _cafe_poster_cases() + _cn_poster_cases()


def resolve_agentic_val_viz_provider() -> AgenticValVizProvider | None:
    """Return the fixed holdout provider when ``AGENTIC_VAL_VIZ=1``.

    Returns:
        The provider, or ``None`` when the holdout is switched off (the default).
    """
    if os.getenv("AGENTIC_VAL_VIZ", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    return AgenticValVizProvider(_default_holdout_cases())
