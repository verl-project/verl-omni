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

"""CPU tests for the terminal / forced-context parsing of the RPCO multi-dim reward.

``_terminal_decision`` reports whether an episode closed and whether its closing
reflection was policy prose or harness-injected context. ``build_forced_reflection``
writes its stop cue on the *same* line as the verdict it paraphrases, so the cue scan has
to start at the verdict rather than at the next line break. Anchoring one line late made
``forced_context`` a constant ``False``, which pinned ``reward_done`` and
``forced_reflection_context`` to 0 for every Bagel Co-RL episode (both read 0 at all 150
validation steps of ``bagel_corl_rm1_0123_20261012_084724``) while letting ``R_format``'s
forced-context check be satisfied by the harness's own ``Reflection:`` line.
"""

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "verl_omni" / "utils" / "reward_score" / "agentic_multidim_reward.py"
_SPEC = importlib.util.spec_from_file_location("agentic_multidim_reward_under_test", _SRC)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_score = _MODULE.compute_score
_normalize_trajectory_dialect = _MODULE._normalize_trajectory_dialect
_terminal_decision = _MODULE._terminal_decision

# The Bagel Co-RL loop renders calls in the <tools> schema dialect, announces the accepted
# image as a bare absolute path, and paraphrases the in-loop RM verdict into the forced
# Reflection line (see agentic_multidim_reward's module docstring).
_GENERATE_CALL = '<tools>{"name": "generate_image", "arguments": {"prompt": "a cat riding a bicycle"}}</tools>'
_PNG_OBSERVATION = "path=/tmp/bagel_corl/gen_1.png"
_STOP_CUE = "agentic_stop_decision_required=1"
_FORCED_REFLECTION = (
    "Reflection: VL judge reports correctness=0.9000, aesthetics=0.8500, good_enough=YES. "
    "rubber_stamp=False. findings: see VL facet scores above. Stop now; do not call another "
    "tool. Your next and only action must be exactly Done."
)


def _bagel_transcript(*, cue: str = _STOP_CUE, trailing: str = "Done.") -> str:
    """One forced-closure episode: generate -> image -> forced Reflection -> policy Done."""
    forced = f"{_FORCED_REFLECTION} {cue}".strip()
    return "\n".join([_GENERATE_CALL, _PNG_OBSERVATION, forced, trailing])


def _score(transcript: str, *, task_type: str = "reflect") -> dict:
    return compute_score(
        solution_str=transcript,
        ground_truth={"task_type": task_type, "expected_num_images": 1},
    )


def test_stop_cue_on_verdict_line_sets_forced_context():
    transcript = _normalize_trajectory_dialect(_bagel_transcript())
    terminal_done, policy_reflection, forced_context = _terminal_decision(transcript)
    assert forced_context is True
    # The harness's own line must not be read as the policy's reflection.
    assert policy_reflection is False
    assert terminal_done is True


def test_forced_closure_scores_reward_done():
    result = _score(_bagel_transcript())
    assert result["reward_done"] == 1.0
    assert result["forced_reflection_context"] == 1
    assert result["terminal_done"] == 1
    assert result["terminal_policy_reflection"] == 0


def test_closure_without_any_cue_does_not_count():
    """The fix scans from the verdict; it must not invent a cue that is not there."""
    transcript = _normalize_trajectory_dialect(_bagel_transcript(cue=""))
    terminal_done, policy_reflection, forced_context = _terminal_decision(transcript)
    assert forced_context is False
    assert policy_reflection is False
    assert terminal_done is True

    result = _score(_bagel_transcript(cue=""))
    assert result["reward_done"] == 0.0
    assert result["forced_reflection_context"] == 0


def test_cue_before_the_verdict_does_not_count():
    """A cue the policy never followed a verdict with is not forced context."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            f"{_PNG_OBSERVATION} {_STOP_CUE}",
            _FORCED_REFLECTION,
            "Done.",
        ]
    )
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert forced_context is False
    assert policy_reflection is False
    assert terminal_done is True


def test_policy_reflection_after_the_verdict_still_counts():
    """The legacy path is untouched: a policy-written reflection closes the episode."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            _PNG_OBSERVATION,
            _FORCED_REFLECTION,
            "Reflection: the wheels were too small, so I enlarged them. Done.",
        ]
    )
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert policy_reflection is True
    assert terminal_done is True
    assert _score(transcript)["reward_done"] == 1.0


def test_mode_2a_cue_on_a_later_line_is_detected():
    """The Mode-2a loops append their cue on its own line; the anchor still advances."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            _PNG_OBSERVATION,
            _FORCED_REFLECTION,
            "agentic_forced_reflection=1",
            "Done.",
        ]
    )
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert forced_context is True
    assert terminal_done is True


@pytest.mark.parametrize("task_type", ["reflect", "plan"])
def test_no_verdict_means_no_terminal_decision(task_type):
    transcript = "\n".join([_GENERATE_CALL, _PNG_OBSERVATION, "Done."])
    assert _terminal_decision(transcript) == (False, False, False)
    result = _score(transcript, task_type=task_type)
    assert result["reward_done"] == 0.0
    assert result["forced_reflection_context"] == 0


# ---------------------------------------------------------------------------
# The terminal marker in the checkpoint's own dialect.
#
# ``build_forced_reflection`` tells the policy its next action "must be exactly
# Done.", but the checkpoint answers in the tool grammar it was trained on and emits
# ``{"name": "Done"}`` glued straight onto the stop cue. Only prose ``Done.`` used to
# satisfy the trailing check, so those episodes scored as never terminating, which
# also zeroed ``reward_result`` and docked ``reward_format``.
# ---------------------------------------------------------------------------


def _json_done_transcript(*, glued: bool = True) -> str:
    forced = f"{_FORCED_REFLECTION} {_STOP_CUE}"
    return "\n".join([_GENERATE_CALL, _PNG_OBSERVATION, forced + '{"name": "Done"}' if glued else ""]) + (
        "" if glued else f"\n{forced}\n" + '{"name": "Done"}'
    )


def test_json_done_is_a_termination_in_this_grammar():
    transcript = _json_done_transcript()
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert terminal_done is True
    assert forced_context is True
    assert policy_reflection is False

    result = _score(transcript)
    assert result["reward_done"] == 1.0
    assert result["terminal_done"] == 1


def test_json_done_on_its_own_line_is_a_termination_too():
    """On its own line the object is wrapped in ``<tool_call>`` and ``_assistant_prose`` drops it.

    So the marker has to be read from the post-verdict region rather than from the
    cleaned prose suffix.
    """
    transcript = _json_done_transcript(glued=False)
    assert _score(transcript)["reward_done"] == 1.0


def test_json_done_before_the_verdict_does_not_close_the_episode():
    """The region guard still excludes an earlier turn, exactly as the cue floor does."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            _PNG_OBSERVATION,
            '{"name": "Done"}',
            f"{_FORCED_REFLECTION} {_STOP_CUE}",
        ]
    )
    result = _score(transcript)
    assert result["reward_done"] == 0.0
    assert result["terminal_done"] == 0


# ---------------------------------------------------------------------------
# The composite dims: ``protocol``, ``image`` and ``done`` are scored, not reported.
# ---------------------------------------------------------------------------


def test_protocol_ok_is_parse_level_not_structural_completeness():
    """``protocol_ok`` answers "did anything silently fail", not "did the loop close".

    An episode that parses cleanly but never terminated has ``protocol_ok == 1`` and
    ``reward_done == 0``. Conflating the two pinned ``protocol_ok`` to the whole of
    ``R_format`` and hid the parse-level term.
    """
    transcript = "\n".join([_GENERATE_CALL, _PNG_OBSERVATION, f"{_FORCED_REFLECTION} {_STOP_CUE}"])
    result = _score(transcript)

    assert result["protocol_ok"] == 1
    assert result["reward_protocol"] == 1.0
    assert result["reward_done"] == 0.0


def test_protocol_dim_costs_an_unparsed_block_or_a_failed_verdict():
    """``R_protocol`` is four independent parses; any one of them failing costs a quarter."""
    protocol = _MODULE._protocol_reward
    assert protocol(raw_blocks=1, parsed_calls=1, judge_ok=1, judge_failed=0, blocked=False) == 1.0
    # An unparsed ``<tool_call>`` block.
    assert protocol(raw_blocks=2, parsed_calls=1, judge_ok=1, judge_failed=0, blocked=False) == pytest.approx(0.75)
    # An ``agentic_judge ok=0``.
    assert protocol(raw_blocks=1, parsed_calls=1, judge_ok=1, judge_failed=1, blocked=False) == pytest.approx(0.75)
    # A ``blocked_after_*`` stop.
    assert protocol(raw_blocks=1, parsed_calls=1, judge_ok=1, judge_failed=0, blocked=True) == pytest.approx(0.75)
    # No call at all is not "valid protocol": there is nothing to be valid.
    assert protocol(raw_blocks=0, parsed_calls=0, judge_ok=0, judge_failed=0, blocked=False) == 0.0


def _two_pass_transcript(first: tuple[float, str], second: tuple[float, str]) -> str:
    """One episode with two generate+judge passes, so the rewrite gain is in play."""

    def reflection(score: float, verdict: str, *, cue: bool = False) -> str:
        line = (
            f"Reflection: VL judge reports correctness={score:.4f}, aesthetics={score:.4f}, "
            f"good_enough={verdict}. rubber_stamp=False. findings: a cat on a bicycle. Stop now."
        )
        return f"{line} {_STOP_CUE}" if cue else line

    return "\n".join(
        [
            _GENERATE_CALL,
            "path=/tmp/bagel_corl/gen_1.png",
            reflection(*first),
            _GENERATE_CALL,
            "path=/tmp/bagel_corl/gen_2.png",
            reflection(*second, cue=True),
            "Done.",
        ]
    )


def test_image_dim_rewards_the_final_pass_plus_the_gain_a_rewrite_bought():
    """The undo of a failed first pass is the behaviour this lane exists to train."""
    improved = _normalize_trajectory_dialect(_two_pass_transcript((0.2, "NO"), (0.8, "YES")))
    values = _MODULE._image_reward(improved, successful_generates=2, blocked=False, rewrite_after_yes=0)
    # 0.8 final + (0.8 - 0.2) gain, capped at 1.
    assert values == pytest.approx(1.0)

    flat = _normalize_trajectory_dialect(_two_pass_transcript((0.6, "NO"), (0.7, "YES")))
    flat_value = _MODULE._image_reward(flat, successful_generates=2, blocked=False, rewrite_after_yes=0)
    # A smaller gap must score strictly lower, or the dim cannot separate the two.
    assert flat_value == pytest.approx(0.8)
    assert flat_value < values


def test_image_dim_does_not_pay_for_a_degraded_rewrite():
    degraded = _normalize_trajectory_dialect(_two_pass_transcript((0.8, "NO"), (0.2, "YES")))
    value = _MODULE._image_reward(degraded, successful_generates=2, blocked=False, rewrite_after_yes=0)
    # The gain term is one-sided; only the (lower) final quality survives.
    assert value == pytest.approx(0.2)


def test_image_dim_scores_a_single_pass_on_its_own_quality():
    transcript = _normalize_trajectory_dialect(_bagel_transcript())
    value = _MODULE._image_reward(transcript, successful_generates=1, blocked=False, rewrite_after_yes=0)
    assert value == pytest.approx(0.875), "one pass is never penalised for having no rewrite"


def test_image_dim_is_zero_on_a_hard_block_or_a_rewrite_after_yes():
    transcript = _normalize_trajectory_dialect(_bagel_transcript())
    assert _MODULE._image_reward(transcript, successful_generates=1, blocked=True, rewrite_after_yes=0) == 0.0
    assert _MODULE._image_reward(transcript, successful_generates=1, blocked=False, rewrite_after_yes=1) == 0.0


def test_image_is_the_heavy_dim_and_done_is_scored():
    """``image`` is the only dim with a continuous magnitude, so it carries the weight."""
    assert _MODULE.DIMS == ("reflect", "plan", "format", "protocol", "tool", "result", "image", "done")
    assert _MODULE.DEFAULT_DIM_WEIGHTS["image"] > _MODULE.DEFAULT_DIM_WEIGHTS["format"]
    for dim in _MODULE.DIMS:
        assert f"reward_{dim}" in _score(_bagel_transcript())


def test_the_weighted_total_uses_the_default_weights_for_the_new_dims():
    result = _score(_bagel_transcript())
    weights = _MODULE.DEFAULT_DIM_WEIGHTS
    active = [dim for dim in _MODULE.DIMS if dim != "plan"]
    expected = sum(weights[dim] * result[f"reward_{dim}"] for dim in active) / sum(weights[dim] for dim in active)
    assert result["score"] == pytest.approx(expected)
