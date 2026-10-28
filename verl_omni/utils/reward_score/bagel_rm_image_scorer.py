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
"""Reward function for the Bagel Co-RL (Joint-Training) in-loop RM payload (RFC §4.2).

Wire as the custom reward function of the colocated RM pool — the SAME handles
serve mid-loop GEN scoring and post-hoc episode scoring, so ``compute_score``
dispatches on payload shape::

    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.bagel_rm_image_scorer
    reward.custom_reward_function.name=compute_score

Call contract (this is what the configured manager actually does)
----------------------------------------------------------------
``RewardLoopWorker.compute_score(data)`` → ``reward_manager.run_single(data)``.
``NaiveRewardManager`` (the repo default) reads ``data_source`` /
``reward_model["ground_truth"]`` and calls the reward function with exactly four
keywords — ``data_source``, ``solution_str``, ``ground_truth``, ``extra_info``
(``VisualRewardManager`` passes ``solution_image`` instead), and merges the
returned dict's keys into ``reward_extra_info``. A reward function that takes a
single positional ``data`` therefore fails with ``TypeError``/``KeyError``; so
does one that returns ``reward_score`` instead of ``score``.

The payload must ride a key the manager already forwards. ``extra_info`` is that
key: the in-loop builder
(``verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.reward._default_data_builder``) puts the payload
under ``extra_info["bagel_corl"]``.

- Mid-loop (``verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.reward.build_rm_score_payload``): each
  image is judged (C/A via the frozen VL judge) and the result carries per-image
  ``sample_scores`` / ``sample_good_enough`` aligned to ``image_paths`` — exactly
  what ``parse_rm_result`` consumes off ``reward_extra_info``.
- Episode post-hoc (no payload): delegated to
  ``bagel_corl_multidim_reward.compute_score`` (image-grounded when ``K >= 1``,
  multi-dim RPCO when ``K = 0``) — the authoritative token-GRPO scalar.

Fails loud when any image cannot be scored (missing file, empty judge URL, parse
failure): a silently-unscored GEN call would zero FlowGRPO signal for that group.
"""

from __future__ import annotations

import base64
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["BAGEL_RM_EXTRA_INFO_KEY", "compute_score"]

# Default ``good_enough`` cut on the [0, 1] UnifiedReward scale. The score is
# ``(mean(1-5 axes) - 1) / 4``. ``good_enough_threshold`` (used by the facet judge,
# 0.8 on a 6-level grid) must NOT be reused here: it is a different scale.
#
# Calibrated against the judge it is paired with, not chosen a priori. The knob has failed
# in *both* directions: too high and every episode reads ``good_enough=NO`` (the stop cue
# becomes a constant, carrying no GRPO signal); too low and it reads YES on nearly every
# *first* image, so every episode stops at J=1/K=1 and the ``max_generate_passes``
# reflection loop is dead by construction. 0.62 keeps the stop a real event that a rewrite
# can still reach, which is what makes J/K vary instead of pinning.
#
# This is a curriculum knob, not a quality bar: raise it toward ``good_enough_threshold``
# as the GEN lane actually improves.
DEFAULT_UNIFIED_GOOD_ENOUGH = 0.62

# Wire key shared with the producer (verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/reward.py). Spelled
# out in both modules rather than imported so this module stays importable in
# isolation (the CPU test loads it by path without the agent_loop package).
BAGEL_RM_EXTRA_INFO_KEY = "bagel_corl"

# ``good_enough`` cut for the ``pickscore`` backend, on PickScore's ``raw / 26`` scale
# (see ``pickscore_reward._PickScoreInferencer.score``). PickScore is *relative* by
# design -- it ranks candidates against a prompt -- so this cut is deliberately
# conservative: the stop cue must not fire on every first image (the failure mode
# ``DEFAULT_UNIFIED_GOOD_ENOUGH`` documents at length). Calibrate it from
# ``reward_extra_info["per_image"][*]["score"]`` on the first run, the same way the
# unified cut was derived.
DEFAULT_PICKSCORE_GOOD_ENOUGH = 1.0


def _mid_loop_payload(extra_info: Any) -> dict[str, Any] | None:
    """Return the in-loop payload riding ``extra_info``, or ``None`` for episodes."""
    if not isinstance(extra_info, dict):
        return None
    payload = extra_info.get(BAGEL_RM_EXTRA_INFO_KEY)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(
            f"bagel_rm_image_scorer: extra_info[{BAGEL_RM_EXTRA_INFO_KEY!r}] must be a dict, got {type(payload)!r}"
        )
    return payload


def _resolve_judge_endpoint(knobs: dict[str, Any], manager_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Point the judge at the reward pool's own router when the manager offers one."""
    router = str(manager_kwargs.get("reward_router_address") or "").strip()
    if not router:
        return knobs
    out = dict(knobs)
    out["vllm_url"] = router if router.startswith(("http://", "https://")) else f"http://{router}"
    model_name = str(manager_kwargs.get("model_name") or "").strip()
    if model_name:
        # vLLM serves the checkpoint path as its model id; an empty model id is rejected
        # by the OpenAI route, so carry the RM's own path through.
        out["vllm_model"] = model_name
    return out


def _score_image_unified(
    *, path: str, caption: str, knobs: dict[str, Any]
) -> tuple[float, bool, dict[str, float]] | None:
    """Score one image with UnifiedReward 2.0; ``(normalized_score, good_enough, axes)`` or None."""
    from verl_omni.utils.agentic.vllm_chat import post_vllm_chat
    from verl_omni.utils.reward_score.unified_reward import (
        _aggregate_unified_reward_scores,
        _build_unified_reward_prompt,
        _parse_unified_reward_scores,
    )

    vllm_url = str(knobs.get("vllm_url") or "").strip()
    if not vllm_url:
        logger.warning("unified reward: no judge endpoint (vllm_url) in knobs; cannot score %s", path)
        return None
    try:
        image_b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("unified reward: cannot read image %s: %s", path, exc)
        return None

    raw, err = post_vllm_chat(
        vllm_url=vllm_url,
        image_b64=image_b64,
        prompt_text=_build_unified_reward_prompt(caption),
        max_tokens=int(knobs.get("reflect_max_new_tokens") or 512),
        model=str(knobs.get("vllm_model") or ""),
        timeout=float(knobs.get("reflect_vlm_timeout") or 120.0),
        enable_thinking=bool(knobs.get("judge_enable_thinking", False)),
    )
    if raw is None:
        logger.warning("unified reward: call failed for %s: %s", path, err)
        return None
    axes = _parse_unified_reward_scores(raw)
    if not axes:
        logger.warning(
            "unified reward: reply lacks the 1-5 axes for %s; raw=%r",
            path,
            re.sub(r"\s+", " ", raw.strip())[:200],
        )
        return None

    normalized, raw_score = _aggregate_unified_reward_scores(axes)
    threshold = float(knobs.get("unified_good_enough_threshold", DEFAULT_UNIFIED_GOOD_ENOUGH))
    logger.debug(
        "unified reward %s: axes=%s raw=%.3f normalized=%.3f good_enough=%s",
        path,
        axes,
        raw_score,
        normalized,
        normalized >= threshold,
    )
    return float(normalized), bool(normalized >= threshold), {k: float(v) for k, v in axes.items()}


# PickScore handle, cached per process: the RM worker scores every GEN image of every
# episode, so the CLIP pair is loaded once, not once per image.
_PICKSCORE_INFERENCER: Any = None


def _pickscore_inferencer(device: str | None = None) -> Any:
    """Build (once) and return this process's PickScore inferencer.

    Reuses ``pickscore_reward._PickScoreInferencer`` -- the same object the standalone
    ``flowgrpo_trainer/bagel`` PickScore recipe trains against -- so the repo has one
    PickScore implementation, not two that can drift.
    """
    global _PICKSCORE_INFERENCER
    if _PICKSCORE_INFERENCER is None:
        from verl_omni.utils.reward_score.pickscore_reward import _PickScoreInferencer

        _PICKSCORE_INFERENCER = _PickScoreInferencer(device=device)
    return _PICKSCORE_INFERENCER


def _score_image_pickscore(*, path: str, caption: str, knobs: dict[str, Any]) -> tuple[float, bool] | None:
    """Score one image with PickScore; ``(score, good_enough)`` or ``None`` on failure."""
    from PIL import Image

    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        logger.warning("pickscore reward: cannot read image %s: %s", path, exc)
        return None
    try:
        inferencer = _pickscore_inferencer(device=knobs.get("pickscore_device"))
        score = float(inferencer.score([caption], [image])[0])
    except Exception as exc:  # noqa: BLE001 - a load/OOM failure must not zero-fill
        logger.warning("pickscore reward: scoring %s failed: %s", path, exc)
        return None

    threshold = float(knobs.get("pickscore_good_enough_threshold", DEFAULT_PICKSCORE_GOOD_ENOUGH))
    logger.debug("pickscore reward %s: score=%.3f good_enough=%s", path, score, score >= threshold)
    return score, bool(score >= threshold)


def _rubric_row(judged: dict[str, Any]) -> dict[str, Any]:
    """The parts of a Mode-2a rubric verdict the loop's ``judge_image`` response quotes."""
    return {
        "correctness": float(judged.get("correctness", 0.0)),
        "aesthetics": float(judged.get("aesthetics", 0.0)),
        "findings": str(judged.get("findings") or ""),
        "suggested_fixes": str(judged.get("suggested_fixes") or ""),
        "rubber_stamp": bool(judged.get("rubber_stamp", False)),
        "correctness_scores": dict(judged.get("correctness_scores") or {}),
        "aesthetics_scores": dict(judged.get("aesthetics_scores") or {}),
    }


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch on payload shape: mid-loop image scoring vs post-hoc episode scoring.

    Args:
        data_source: Unused; kept for the verl ``compute_score`` signature.
        solution_str: Decoded trajectory text. The reward managers pass this for a *token*
            response: ``NaiveRewardManager`` from its own decode, and ``VisualRewardManager``
            for the Bagel lane, whose episode is a Hermes text trajectory
            (``_is_token_response``). Unused when the in-loop payload is present — that
            carries image paths, not text.
        ground_truth: Episode ground truth, forwarded to the episode scorer.
        extra_info: Manager-forwarded metadata. Carries ``bagel_corl`` →
            mid-loop GEN scoring; absent → post-hoc episode scoring.
        **kwargs: Manager-supplied scorer extras. ``reward_router_address`` and
            ``model_name`` (``VisualRewardManager`` with a deployed reward model)
            are consumed to scope the judge to the reward pool's own model.

    Returns:
        Flat dict with ``score`` plus metric keys (``sample_scores``,
        ``sample_good_enough``, ``good_enough`` for the in-loop path). Flat, not
        nested under ``reward_extra_info``: the manager does the nesting.

    Raises:
        ValueError: If the payload is malformed, ``good_enough_threshold`` is
            missing, or any image fails to score (fail-loud, no zero-fill).
    """
    payload = _mid_loop_payload(extra_info)
    if payload is None:
        from verl_omni.utils.reward_score.bagel_corl_multidim_reward import compute_score as episode_compute_score

        return episode_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    from verl_omni.utils.reward_score.agentic_image_judge_client import call_reflect_vlm

    image_paths = [str(p) for p in payload.get("image_paths") or []]
    if not image_paths:
        raise ValueError("bagel_rm_image_scorer: payload has no image_paths")

    extra_info = dict(payload.get("extra_info") or {})
    knobs = dict(payload.get("scorer_knobs") or {})
    knobs.setdefault("good_enough_threshold", extra_info.get("good_enough_threshold"))
    if knobs.get("good_enough_threshold") is None:
        raise ValueError(
            "bagel_rm_image_scorer: good_enough_threshold missing from payload "
            "scorer_knobs/extra_info; the loop must stamp agentic scorer knobs"
        )
    knobs["good_enough_threshold"] = float(knobs["good_enough_threshold"])
    # Route the judge at the reward pool when one is deployed (see
    # ``_resolve_judge_endpoint``); falls back to the configured judge otherwise.
    knobs = _resolve_judge_endpoint(knobs, kwargs)
    # ``user_prompt`` is the rendered user turn, which carries the dataset's brevity
    # instructions; the judge caption must be the request alone.
    raw_request = extra_info.get("raw_prompt")
    user_request = (
        raw_request.strip()
        if isinstance(raw_request, str) and raw_request.strip()
        else str(extra_info.get("user_prompt") or "")
    )
    image_prompt = str(payload.get("image_prompt") or "")
    reference_paths = [str(p) for p in payload.get("reference_paths") or []]
    if reference_paths:
        notes = f"reference images: {', '.join(reference_paths)}"
    else:
        notes = str(extra_info.get("judge_notes") or "")

    # Reward backend. ``vlm_judge`` is the original discrete facet-grid judge
    # (correctness/aesthetics snapped to 0.0/0.2/.../1.0). ``unified_reward`` scores
    # the image against its prompt on UnifiedReward 2.0's 1-5 Alignment/Coherence/Style
    # axes and normalizes to [0, 1] -- a *continuous* score. ``pickscore`` scores it
    # with PickScore_v1 instead, in-process, for a GEN reward closer to the one the
    # canonical Bagel FlowGRPO recipes train against.
    #
    # Why the continuous paths exist: the facet judge put every GEN image on the 0.0 floor,
    # so every sample in a FlowGRPO group scored identically and the group advantage was
    # exactly zero. A reward with no within-group variance cannot train a policy no matter
    # how correct it is.
    backend = str(knobs.get("score_backend") or "unified_reward").strip().lower()
    if backend not in ("unified_reward", "unified_reward_rubric", "vlm_judge", "pickscore"):
        raise ValueError(
            f"bagel_rm_image_scorer: unknown score_backend {backend!r}; "
            "expected 'unified_reward', 'unified_reward_rubric', 'vlm_judge' or 'pickscore'"
        )
    # ``unified_reward_rubric``: the reward stays UnifiedReward's point score, and the same
    # reward-pool model is asked the Mode-2a C/A rubric a second time, for the
    # ``judge_image`` response and the stop verdict only.
    version = str(knobs.get("corl_reward_version", "v1"))
    if version not in ("v1", "v2"):
        raise ValueError("corl_reward_version must be v1 or v2")
    if version == "v2" and not user_request:
        raise ValueError("V2 image scorer requires original user request")
    if version == "v2":
        image_prompt = user_request
    with_rubric = backend == "unified_reward_rubric"
    # The user's request is the fixed target across rewrite passes; the policy's own
    # diffusion prompt would move it every pass.
    caption = user_request or image_prompt

    scores: list[float] = []
    flags: list[bool] = []
    per_image: list[dict[str, Any]] = []
    sample_axes: list[dict[str, float]] = []
    sample_judge: list[dict[str, Any] | None] = []
    failed: list[str] = []
    for path in image_paths:
        if backend == "vlm_judge":
            judged = call_reflect_vlm(
                user_request=user_request,
                image_prompt=image_prompt,
                notes=notes,
                image_path=path,
                extra_info=dict(knobs),
            )
            if judged is None or not judged.get("ok"):
                failed.append(path)
                continue
            correctness = float(judged.get("correctness", 0.0))
            aesthetics = float(judged.get("aesthetics", 0.0))
            match = judged.get("match")
            facets = [correctness, aesthetics] + ([float(match)] if match is not None else [])
            score = sum(facets) / len(facets)
            good_enough = (
                score >= knobs["good_enough_threshold"] if version == "v2" else bool(judged.get("good_enough", False))
            )
            sample_judge.append(_rubric_row(judged))
        elif backend == "pickscore":
            picked = _score_image_pickscore(path=path, caption=caption, knobs=knobs)
            if picked is None:
                failed.append(path)
                continue
            score, good_enough = picked
        else:
            unified = _score_image_unified(path=path, caption=caption, knobs=knobs)
            if unified is None:
                failed.append(path)
                continue
            score, good_enough, axes = unified
            sample_axes.append(axes)
            if with_rubric:
                judged = call_reflect_vlm(
                    user_request=user_request,
                    image_prompt=image_prompt,
                    notes=notes,
                    image_path=path,
                    extra_info=dict(knobs),
                )
                if judged is not None and judged.get("ok"):
                    sample_judge.append(_rubric_row(judged))
                    if version == "v1":
                        good_enough = bool(judged.get("good_enough", False))
                else:
                    # The reward is already scored; an unparsed rubric only costs this
                    # image its written verdict, so fall back to the axes text.
                    logger.warning("unified_reward_rubric: rubric judge failed for %s; using axes", path)
                    sample_judge.append(None)
        if not math.isfinite(float(score)):
            raise ValueError("bagel_rm_image_scorer: nonfinite quality; refusing zero-fill")
        scores.append(score)
        flags.append(good_enough)
        per_image.append({"image_path": path, "score": score, "good_enough": good_enough})
    if failed:
        raise ValueError(
            f"bagel_rm_image_scorer: failed to score {len(failed)}/{len(image_paths)} image(s) "
            f"(first: {failed[0]}); refusing zero-fill for FlowGRPO"
        )

    reward_score = sum(scores) / len(scores)
    # Flat, manager-shaped dict: NaiveRewardManager does result["score"] and merges
    # every key into reward_extra_info, which is where parse_rm_result looks for
    # sample_scores / sample_good_enough. Nesting these under reward_extra_info
    # here would KeyError on "score" before scoring ever returns.
    result = {
        "score": float(reward_score),
        "sample_scores": scores,
        "sample_good_enough": flags,
        "good_enough": all(flags),
        "per_image": per_image,
    }
    if backend in ("unified_reward", "unified_reward_rubric"):
        result["sample_axes"] = sample_axes
    if backend in ("vlm_judge", "unified_reward_rubric"):
        # The Mode-2a rubric verdict per image (C/A, facet grid, findings, fixes), so the
        # loop's ``judge_image`` response carries the judge's own words, not a template.
        result["sample_judge"] = sample_judge
    return result
