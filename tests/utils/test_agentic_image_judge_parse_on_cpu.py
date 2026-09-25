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
"""CPU tests for the judge observation formatter and its sentence-aware clipping."""

from __future__ import annotations

import pytest

from verl_omni.utils.agentic.text_clip import clip_to_sentences
from verl_omni.utils.agentic_image_judge_parse import format_judge_observation


def test_clip_to_sentences_leaves_short_text_untouched():
    assert clip_to_sentences("text is legible.", 220) == "text is legible."
    assert clip_to_sentences("  spaced   out  ", 220) == "spaced out"
    assert clip_to_sentences("", 220) == ""
    assert clip_to_sentences(None, 220) == ""


def test_clip_to_sentences_keeps_whole_sentences_and_marks_the_cut():
    sentence = "The requested English title is replaced by illegible white glyphs. "
    source = sentence * 6
    clipped = clip_to_sentences(source, 220)

    assert clipped.endswith("…")
    assert clipped[: -len(" …")].rstrip().endswith("glyphs.")
    # Whole sentences only: the retained body is a prefix of the source text.
    assert source.strip().startswith(clipped[: -len(" …")].rstrip())


def test_clip_to_sentences_falls_back_to_a_word_boundary():
    """One sentence longer than the budget must still not split a word."""
    source = " ".join(f"word{index:02d}" for index in range(80))
    clipped = clip_to_sentences(source, 220)

    assert clipped.endswith("…")
    retained = clipped[: -len(" …")].rstrip()
    assert retained.split() == source.split()[: len(retained.split())]
    assert len(retained) <= 220


def test_clip_to_sentences_handles_cjk_full_stops():
    sentence = "这张海报的文字完全错误需要重做。"
    clipped = clip_to_sentences(sentence * 20, 40)

    assert clipped.endswith("…")
    assert clipped[: -len(" …")].rstrip().endswith("。")
    assert len(clipped) <= 42


def test_format_judge_observation_clips_on_sentence_boundaries():
    """The live regression: a fixed-width slice split words in the policy's instruction.

    ``sample_9003``'s observation carried ``"...to match the 'Sofa M"`` in
    ``suggested_fixes`` and ``"The requested Chinese tit"`` in ``findings``. The policy
    read those half-words as its rewrite instruction, and the slice also dropped the
    concrete fixes the judge wrote later in the field.
    """
    parsed = {
        "correctness": 0.36,
        "aesthetics": 0.60,
        "good_enough": False,
        "findings": (
            "The image contains a cat in a prone position on a blue background, but the "
            "text is completely wrong. The requested English title 'Sofa Montain "
            "Slummerfest' is replaced by illegible white glyphs. The requested Chinese "
            "title is missing entirely."
        ),
        "suggested_fixes": (
            "Use a high-resolution font rendering engine for all text. Replace the "
            "calligraphic style with a bold, sans-serif font for the Chinese title. "
            "Restore both text columns and the sponsor row."
        ),
    }
    observation, meta = format_judge_observation(image_path="/tmp/image_01.png", parsed=parsed, backend="vllm")

    assert "path=/tmp/image_01.png" in observation
    assert "correctness=0.36" in observation
    assert "aesthetics =0.60" in observation
    assert "good_enough =NO" in observation
    assert "agentic_judge ok=1" in observation
    # Both over-budget fields end on a sentence with an explicit cut marker.
    assert "'Sofa M\n" not in observation
    assert "Chinese tit\n" not in observation
    assert "Restore both\n" not in observation
    assert "glyphs. …" in observation
    assert "Chinese title. …" in observation
    # The dump side keeps the untruncated fields for diagnosis.
    assert meta["findings"] == parsed["findings"]
    assert meta["suggested_fixes"] == parsed["suggested_fixes"]


def test_format_judge_observation_omits_an_ellipsis_when_fields_fit():
    parsed = {
        "correctness": 0.80,
        "aesthetics": 0.76,
        "good_enough": True,
        "findings": "text is legible",
        "suggested_fixes": "none",
    }
    observation, _ = format_judge_observation(image_path="/tmp/image_00.png", parsed=parsed, backend="vllm")

    assert "findings: text is legible" in observation
    assert "suggested_fixes: none" in observation
    assert "…" not in observation


def test_format_judge_observation_coerces_non_string_fields():
    """A malformed judge payload must not raise on the observation path."""
    parsed = {
        "correctness": 0.5,
        "aesthetics": 0.5,
        "good_enough": False,
        "findings": ["a list, not a sentence"],
        "suggested_fixes": None,
    }
    observation, _ = format_judge_observation(image_path="/tmp/image_00.png", parsed=parsed, backend="vllm")

    assert "findings: ['a list, not a sentence']" in observation
    assert "suggested_fixes: none" in observation


@pytest.mark.parametrize("budget", [1, 5, 20])
def test_clip_to_sentences_never_splits_a_word(budget: int):
    source = "alpha beta gamma delta epsilon zeta"
    clipped = clip_to_sentences(source, budget)

    assert clipped.endswith("…")
    retained = clipped[: -len(" …")].rstrip()
    assert retained.split() == source.split()[: len(retained.split())]
