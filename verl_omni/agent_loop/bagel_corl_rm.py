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
"""Mid-episode RM scoring adapter for Bagel Co-RL (Joint-Training) (RFC §4.2)."""

from __future__ import annotations

import asyncio
import functools
import logging
import math
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BAGEL_RM_EXTRA_INFO_KEY",
    "RMScoringError",
    "bind_bagel_rm_handles",
    "build_rm_score_payload",
    "get_bagel_rm_gen_handle",
    "make_rm_score_fn",
    "parse_rm_result",
]

_RM_HANDLES_KEY = "_bagel_corl_rm_handles"
# [0] is the GEN pool. The composite worker's generic name for this slot is "dit",
# but Bagel is a MoT model (no DiT), so this module calls it GEN.
_RM_GEN_HANDLE_INDEX = 0
# Key the in-loop payload rides inside ``extra_info``. ``extra_info`` is the one
# field the configured reward manager forwards verbatim to the reward function, so
# the payload goes there (RFC §4.2). Kept in sync with
# ``verl_omni.utils.reward_score.bagel_rm_image_scorer.BAGEL_RM_EXTRA_INFO_KEY``.
BAGEL_RM_EXTRA_INFO_KEY = "bagel_corl"


class RMScoringError(RuntimeError):
    """Raised when the in-loop RM handle is missing, malformed, or fails."""


def bind_bagel_rm_handles(handles: list[Any] | None) -> None:
    """Stash the reward-loop handles process-locally (worker ``__init__`` time).

    Args:
        handles: ``[gen_handle, ar_handle]`` per the composite contract, or
            ``None`` when no RM pool is wired for this worker.
    """
    import sys

    module = sys.modules[__name__]
    if handles is None:
        setattr(module, _RM_HANDLES_KEY, None)
        return
    if not isinstance(handles, list | tuple):
        raise RMScoringError(f"bagel RM handles must be a list, got {type(handles)!r}")
    if len(handles) < _RM_GEN_HANDLE_INDEX + 1:
        raise RMScoringError(
            f"bagel RM handles need at least {_RM_GEN_HANDLE_INDEX + 1} entries (GEN pool first), got {len(handles)}"
        )
    setattr(module, _RM_HANDLES_KEY, list(handles))


def get_bagel_rm_gen_handle() -> Any | None:
    """Return the GEN-side RM handle bound in this process, or ``None``."""
    import sys

    handles = getattr(sys.modules[__name__], _RM_HANDLES_KEY, None)
    if not handles:
        return None
    return handles[_RM_GEN_HANDLE_INDEX]


def build_rm_score_payload(
    image_paths: list[str],
    reference_paths: list[str] | None = None,
    extra_info: dict[str, Any] | None = None,
    scorer_knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the plain-dict payload handed to the RM worker's ``compute_score``."""
    if not image_paths:
        raise RMScoringError("bagel RM payload needs at least one image path")
    return {
        "image_paths": [str(p) for p in image_paths],
        "reference_paths": [str(p) for p in (reference_paths or [])],
        "extra_info": dict(extra_info or {}),
        "scorer_knobs": dict(scorer_knobs or {}),
    }


def parse_rm_result(
    result: Any,
    image_paths: list[str],
) -> tuple[list[float], list[bool | None]]:
    """Parse a ``compute_score`` result into per-image scores + per-image stop bits.

    Accepts ``reward_score`` as a scalar (broadcast to every image) or
    ``reward_extra_info["sample_scores"]`` aligned with ``image_paths``. The stop
    bit comes from ``reward_extra_info["sample_good_enough"]`` (per-image flags,
    preferred) or ``reward_extra_info["good_enough"]`` (broadcast).
    """
    if not isinstance(result, dict) or "reward_score" not in result:
        raise RMScoringError(f"bagel RM result missing 'reward_score': {type(result)!r}")
    extra = result.get("reward_extra_info") or {}
    if not isinstance(extra, dict):
        extra = {}
    sample_scores = extra.get("sample_scores")
    if sample_scores is not None:
        scores = [float(s) for s in sample_scores]
        if len(scores) != len(image_paths):
            raise RMScoringError(f"bagel RM sample_scores length {len(scores)} != image count {len(image_paths)}")
    else:
        scores = [float(result["reward_score"])] * len(image_paths)
    if not all(math.isfinite(score) for score in scores):
        raise RMScoringError("bagel RM returned nonfinite image score")
    sample_flags = extra.get("sample_good_enough")
    if sample_flags is not None:
        if len(sample_flags) != len(image_paths):
            raise RMScoringError(
                f"bagel RM sample_good_enough length {len(sample_flags)} != image count {len(image_paths)}"
            )
        flags = [None if flag is None else bool(flag) for flag in sample_flags]
    elif "good_enough" in extra and extra["good_enough"] is not None:
        flags = [bool(extra["good_enough"])] * len(image_paths)
    else:
        flags = [None] * len(image_paths)
    return scores, flags


def parse_rm_axes(result: Any, image_paths: list[str]) -> list[dict[str, float] | None]:
    """Per-image RM axis breakdown (``reward_extra_info["sample_axes"]``), ``None`` when absent.

    Only the UnifiedReward backend reports axes. They describe the verdict in the judge
    observation and never change the reward, so a missing or misaligned list degrades to
    ``None`` instead of failing the episode.
    """
    extra = result.get("reward_extra_info") if isinstance(result, dict) else None
    axes = extra.get("sample_axes") if isinstance(extra, dict) else None
    if not isinstance(axes, list | tuple) or len(axes) != len(image_paths):
        return [None] * len(image_paths)
    return [{str(k): float(v) for k, v in row.items()} if isinstance(row, dict) else None for row in axes]


def parse_rm_judges(result: Any, image_paths: list[str]) -> list[dict[str, Any] | None]:
    """Per-image rubric verdicts (``reward_extra_info["sample_judge"]``), ``None`` when absent.

    Only the ``vlm_judge`` backend reports them. Like the axes they shape the judge
    observation, not the reward, so a missing or misaligned list degrades to ``None``.
    """
    extra = result.get("reward_extra_info") if isinstance(result, dict) else None
    judges = extra.get("sample_judge") if isinstance(extra, dict) else None
    if not isinstance(judges, list | tuple) or len(judges) != len(image_paths):
        return [None] * len(image_paths)
    return [dict(row) if isinstance(row, dict) else None for row in judges]


def _default_data_builder(payload: dict[str, Any], *, output_type: str = "image") -> Any:
    """Wrap the payload in the one-row ``DataProto`` the reward manager consumes."""
    import torch
    from tensordict import TensorDict
    from verl.protocol import DataProto

    # Mirror ``_validate_visual_response`` exactly: latent stays floating point,
    # everything else is treated as uint8 pixels. Getting this wrong is a hard
    # ValueError inside the RM worker, mid-episode, after the images were produced.
    responses_dtype = torch.float32 if str(output_type) == "latent" else torch.uint8
    return DataProto(
        batch=TensorDict(
            {
                # Placeholders: NaiveRewardManager decodes these to build an unused
                # ``solution_str``; VisualRewardManager dtype-checks them.
                "responses": torch.zeros((1, 1), dtype=responses_dtype),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "data_source": np.array(["bagel_corl_mid_loop_rm"], dtype=object),
            "reward_model": np.array([{"ground_truth": ""}], dtype=object),
            "extra_info": np.array([{BAGEL_RM_EXTRA_INFO_KEY: payload}], dtype=object),
        },
    )


def make_rm_score_fn(
    handle: Any,
    *,
    timeout_s: float | None = None,
    get_reference_paths: Callable[[], list[str]] | None = None,
    get_extra_info: Callable[[], dict[str, Any]] | None = None,
    get_scorer_knobs: Callable[[], dict[str, Any]] | None = None,
    get_image_prompt: Callable[[], str] | None = None,
    data_builder: Callable[[dict[str, Any]], Any] | None = None,
    output_type: str = "image",
) -> Callable[[list[Any]], Any]:
    """Build the ``score_fn(samples) -> samples`` hook consumed by ``run_serial_episode``."""
    if handle is None:
        raise RMScoringError(
            "bagel in-loop RM scoring requires a reward-loop handle; "
            "wire agent-loop reward handles (trainer.get_reward_handles) or "
            "disable agent.rm_scoring explicitly"
        )
    if data_builder is None:
        data_builder = functools.partial(_default_data_builder, output_type=output_type)

    async def _invoke(data: Any) -> Any:
        method = getattr(handle, "compute_score", None)
        if method is None:
            raise RMScoringError(f"bagel RM handle {type(handle)!r} has no compute_score")
        if hasattr(method, "remote"):
            result = method.remote(data)
        else:
            result = method(data)
        result = await result if asyncio.iscoroutine(result) else result
        # A Ray actor call hands back an ObjectRef, and the ref is not a score: if it is returned unresolved,
        # ``parse_rm_result`` dies on ``missing 'reward_score': <class 'ray.ObjectRef'>``, which aborts the episode
        # and can starve the sync replay buffer until the whole run fails with "no materializable trajectories".
        import ray

        if isinstance(result, ray.ObjectRef):
            return ray.get(result, timeout=timeout_s)
        return result

    async def score(samples: list[Any]) -> list[Any]:
        scored = [s.image_path for s in samples if getattr(s, "valid", False) and getattr(s, "image_path", None)]
        if not scored:
            return samples
        payload = build_rm_score_payload(
            scored,
            reference_paths=get_reference_paths() if get_reference_paths is not None else [],
            extra_info=get_extra_info() if get_extra_info is not None else {},
            scorer_knobs=get_scorer_knobs() if get_scorer_knobs is not None else {},
        )
        if get_image_prompt is not None:
            payload["image_prompt"] = str(get_image_prompt() or "")
        data = data_builder(payload)
        result = await _invoke(data)
        scores, flags = parse_rm_result(result, payload["image_paths"])
        by_path = dict(zip(payload["image_paths"], scores, strict=True))
        flag_by_path = dict(zip(payload["image_paths"], flags, strict=True))
        axes_by_path = dict(zip(payload["image_paths"], parse_rm_axes(result, payload["image_paths"]), strict=True))
        judge_by_path = dict(zip(payload["image_paths"], parse_rm_judges(result, payload["image_paths"]), strict=True))
        for sample in samples:
            if not getattr(sample, "valid", False) or not getattr(sample, "image_path", None):
                continue
            sample.rm_score = by_path[str(sample.image_path)]
            sample.rm_score_source = "bagel_corl_tool_rm"
            flag = flag_by_path.get(str(sample.image_path))
            if flag is not None:
                sample.good_enough = flag
            axes = axes_by_path.get(str(sample.image_path))
            if axes is not None and hasattr(sample, "rm_axes"):
                sample.rm_axes = axes
            judge = judge_by_path.get(str(sample.image_path))
            if judge is not None and hasattr(sample, "rm_judge"):
                sample.rm_judge = judge
        return samples

    return score
