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
"""Judge-outcome lift across a rewrite chain, for the agentic image-gen reward.

The reflect task asks the policy to rewrite a diffusion prompt until the frozen VL
judge stops complaining. This module scores whether those rewrites actually *moved*
the judge, which is the thing the protocol is for.

It replaces an earlier character-n-gram novelty measure. That measure was built to
detect the ``sample_9003`` regime the docstring of the day called an append-only
failure: every pass re-sent the same recipe plus one more quality assertion. Re-read
against the judge scores, though, that chain was climbing — ``correctness`` went
``0.00 -> 0.36 -> 0.56`` and ``aesthetics`` ``0.44 -> 0.60 -> 0.80`` across two
appends, and the rollout was terminated by ``max_generate_image_passes`` while still
improving. The n-gram measure scored that same chain ``0.097``, i.e. it taxed the one
behaviour that was working, because a diffusion prompt is conditioned on *descriptions*
and an append is a legitimate way to add them.

Measuring the judge instead makes the objective what the task actually asks for: did
the last image score better than the first? It also removes two crutches the n-gram
form needed — a request-retention damping (the judge already punishes dropping the
requested content, since that lowers C) and an empty-request guard (there is no
request term left to guard).

Convention follows ``agentic_reward._delta_c_bonus``, which already scores a
correctness lift from the first judge to the preferred one under
``reward_delta_c``; this generalizes it to ``(C + A) / 2`` as the multi-dim reward's
own ``improve`` dim. Both gate on the *first* judge being an explicit ``good_enough=NO``,
so a rollout that satisfied the judge immediately — and correctly stopped — is not
asked to have improved anything.

Known limits, stated rather than papered over:

* ``qwen_image_diversify_seed`` gives each pass a different seed, so part of any lift
  is seed luck rather than prompt quality. A per-rollout delta cannot separate the two;
  pinning the seed for a comparison run is the way to measure the prompt's contribution.
* Any improvement term can be gamed by starting low. The absolute level is anchored by
  the ``reflect`` dim, which scores the preferred judge's quality directly (and carries
  the larger default weight), so sandbagging the first pass to inflate the lift buys a
  term worth at most 1.0 against a loss on the dim that scores the image itself.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["judge_delta_reward", "preferred_judge"]

#: A judge record: ``(correctness, aesthetics, good_enough)``. ``good_enough`` is
#: ``None`` when the observation carried no marker, which is not a first-pass failure.
JudgeRecord = tuple[float, float, bool | None]


def preferred_judge(judges: Sequence[JudgeRecord]) -> JudgeRecord:
    """Return the rollout's best judge: the first ``good_enough=YES``, else the last.

    Mirroring ``agentic_multidim_reward._reflection_reward`` so the lift and the
    absolute quality it is measured against describe the same image. A chain that
    never satisfies the judge is judged on where it finished.

    Args:
        judges: Judge records in submission order. Must be non-empty.

    Returns:
        The first record with ``good_enough`` true if there is one, otherwise the
        final record.
    """
    return next((record for record in judges if record[2] is True), judges[-1])


def judge_delta_reward(judges: Sequence[JudgeRecord]) -> tuple[float, float]:
    """Return ``(score, raw_delta)`` for one rollout's judge chain.

    ``score`` is the mean ``(correctness, aesthetics)`` lift from the first judge to
    :func:`preferred_judge`, clamped into ``[0, 1]``. ``raw_delta`` is the same value
    unclamped, so a regression reports negative in the metrics while contributing
    nothing to the reward.

    A single judge has no chain to lift, and a chain whose first judge was not an
    explicit ``good_enough=NO`` had nothing to recover from; both score ``0.0``.

    Args:
        judges: Judge records in submission order.

    Returns:
        ``(score, raw_delta)``. ``(0.0, 0.0)`` when no lift is measurable.
    """
    if len(judges) < 2:
        return 0.0, 0.0
    first_correctness, first_aesthetics, first_good_enough = judges[0]
    if first_good_enough is not False:
        return 0.0, 0.0
    last_correctness, last_aesthetics, _ = preferred_judge(judges)
    raw_delta = ((last_correctness - first_correctness) + (last_aesthetics - first_aesthetics)) / 2.0
    return max(0.0, min(1.0, raw_delta)), raw_delta
