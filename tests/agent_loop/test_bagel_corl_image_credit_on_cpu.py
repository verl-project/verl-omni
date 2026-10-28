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

"""CPU tests for the Bagel Co-RL UND image credit (``und/no_image_credit``)."""

import pytest

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode import (
    EpisodeRollout,
    GenSample,
    episode_und_reward,
    gen_sample_uid,
    image_credit_reward,
)


def _sample(*, call: str, index: int, score: float | None, valid: bool = True) -> GenSample:
    return GenSample(
        gen_sample_uid=gen_sample_uid(call, index),
        gen_group_uid=call,
        seed_index=index,
        valid=valid,
        prompt_token_ids=[1],
        rm_score=score,
    )


def _episode(*, gens: list[GenSample], und_reward: float = 0.0) -> EpisodeRollout:
    return EpisodeRollout(
        und_group_uid="group",
        episode_uid="episode",
        policy_version=1,
        prompt_ids=[1],
        response_ids=[1],
        response_mask=[1],
        turns=1,
        gen_samples=list(gens),
        und_reward=und_reward,
    )


def test_a_successful_rewrite_beats_a_flat_one():
    """Both of these used to score 0.5, so the token-GRPO signal could not tell them apart."""
    improved = [_sample(call="pass0", index=0, score=0.2), _sample(call="pass1", index=0, score=0.8)]
    flat = [_sample(call="pass0", index=0, score=0.4), _sample(call="pass1", index=0, score=0.6)]

    assert image_credit_reward(improved) == pytest.approx(1.4)  # 0.8 + (0.8 - 0.2)
    assert image_credit_reward(flat) == pytest.approx(0.8)  # 0.6 + (0.6 - 0.4)
    assert image_credit_reward(improved) > image_credit_reward(flat)


def test_one_call_is_scored_on_its_own_quality():
    """A single-pass episode is never penalised for having no rewrite to improve on."""
    assert image_credit_reward([_sample(call="pass0", index=0, score=0.2)]) == pytest.approx(0.2)


def test_a_degraded_rewrite_keeps_only_its_lower_final_quality():
    degraded = [_sample(call="pass0", index=0, score=0.8), _sample(call="pass1", index=0, score=0.2)]
    assert image_credit_reward(degraded) == pytest.approx(0.2)


def test_the_seeds_of_one_call_are_averaged():
    """``S >= 2`` seeds are one prompt at different noise, so no seed is "final"."""
    seeds = [_sample(call="call0", index=0, score=0.4), _sample(call="call0", index=1, score=0.6)]
    assert image_credit_reward(seeds) == pytest.approx(0.5)


def test_the_rewrite_bonus_is_capped_but_the_total_is_not():
    """Clamping the total would make two improved siblings saturate to the same number."""
    # A 1.5 gain is clipped to 1.0.
    huge = [_sample(call="pass0", index=0, score=0.0), _sample(call="pass1", index=0, score=1.5)]
    assert image_credit_reward(huge) == pytest.approx(2.5)

    better = [_sample(call="pass0", index=0, score=0.2), _sample(call="pass1", index=0, score=1.2)]
    worse = [_sample(call="pass0", index=0, score=0.2), _sample(call="pass1", index=0, score=0.7)]
    assert image_credit_reward(better) == pytest.approx(2.2)  # 1.2 + min(1, 1.0)
    assert image_credit_reward(worse) == pytest.approx(1.2)  # 0.7 + 0.5
    assert image_credit_reward(better) > image_credit_reward(worse)


def test_unrated_and_invalid_seeds_do_not_contribute():
    samples = [
        _sample(call="pass0", index=0, score=None),
        _sample(call="pass0", index=0, score=0.9, valid=False),
        _sample(call="pass1", index=0, score=0.3),
    ]
    assert image_credit_reward(samples) == pytest.approx(0.3)
    # No rated seed at all -> None, so the caller falls back to the non-image scalar.
    assert image_credit_reward([]) is None
    assert image_credit_reward([_sample(call="call", index=0, score=None)]) is None


def test_episode_und_reward_uses_the_same_rule_as_the_episode_field():
    """The TQ row and the flatten row are one derivation, or UND and GEN disagree."""
    gens = [_sample(call="pass0", index=0, score=0.2), _sample(call="pass1", index=0, score=0.8)]
    reward, used_non_image = episode_und_reward(_episode(gens=gens))

    assert reward == pytest.approx(1.4)
    assert used_non_image is False
    assert reward == pytest.approx(image_credit_reward(gens))


def test_a_pattern_3_episode_still_falls_back_to_the_non_image_scalar():
    """``K = 0`` must not have its token-GRPO signal zeroed."""
    reward, used_non_image = episode_und_reward(_episode(gens=[], und_reward=0.73))

    assert reward == pytest.approx(0.73)
    assert used_non_image is True


def test_the_row_field_and_the_trained_signal_can_disagree():
    """``und_reward`` on the row is NOT the reward the actor trains on."""
    gens = [_sample(call="pass0", index=0, score=0.2), _sample(call="pass1", index=0, score=0.8)]
    # The reward manager scored the episode 0.5 (say, the composite of the text dims).
    episode = _episode(gens=gens, und_reward=0.5)
    trained, _ = episode_und_reward(episode)

    assert trained == pytest.approx(1.4)  # image_credit_reward: 0.8 final + 0.6 gain
    assert episode.und_reward == pytest.approx(0.5)
    # Every dimension of the divergence, made explicit: the trained signal is not the
    # stored field, not the composite, and not bounded by 1.
    assert trained != pytest.approx(episode.und_reward)
    assert trained > 1.0
