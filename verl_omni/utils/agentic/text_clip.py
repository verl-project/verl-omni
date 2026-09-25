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
"""Sentence-aware clipping for judge text that reaches the policy.

A frozen VL judge writes a diagnosis, and the harness shortens it before it becomes a
tool observation and then a reflection cue. Slicing at a fixed character count splits
words: live ``sample_9003`` carried ``"...to match the 'Sofa M"`` in
``suggested_fixes`` and ``"The requested Chinese tit"`` in ``findings``, both cut at a
220/160 character budget with no marker. The policy reads those fields as its
instruction for the next rewrite, so a half-word is a malformed instruction, and a
fixed-width slice also silently keeps whichever sentence happens to come first while
discarding the concrete fixes the judge wrote later in the field.
"""

from __future__ import annotations

import re

__all__ = ["clip_to_sentences"]

#: Judge fields end sentences on ASCII or CJK full stops. CJK punctuation is not
#: followed by whitespace, so those boundaries are zero-width while ASCII ones require a
#: space (otherwise ``1.2`` and ``'Sofa M.`` would split mid-token).
_SENTENCE_END_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")


def clip_to_sentences(text: str, budget: int) -> str:
    """Clip ``text`` to whole sentences within ``budget`` characters.

    Args:
        text: Raw judge field text.
        budget: Character budget for the returned text.

    Returns:
        Whitespace-collapsed ``text``, unchanged when it already fits; otherwise whole
        sentences up to ``budget`` followed by ``" …"``. A single sentence longer than
        ``budget`` falls back to whole words, and to a character cut for CJK where every
        character carries meaning and there are no spaces to cut on. Empty input returns
        ``""``.
    """
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    if not collapsed or len(collapsed) <= budget:
        return collapsed
    kept = ""
    for sentence in _SENTENCE_END_RE.split(collapsed):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > budget:
            break
        kept = candidate
    if not kept:
        if " " in collapsed:
            # One long sentence: keep whole words only. Below the first word's length
            # there is no whole word to keep, and a partial word is what this module
            # exists to avoid, so drop it rather than corrupt the instruction.
            for word in collapsed.split():
                candidate = f"{kept} {word}".strip()
                if len(candidate) > budget:
                    break
                kept = candidate
        else:
            kept = collapsed[:budget].strip()
    return f"{kept} …" if kept else "…"
