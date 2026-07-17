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
"""Data-side written -> spoken text normalization for TTS prompts.

``verbalize(text)`` rewrites written forms the talker cannot pronounce into their spoken form,
applied to prompts BEFORE training/synthesis: URLs say "dot"/"slash" out loud and emails spell
coined handles. Deterministic and idempotent.

The matching REWARD/EVAL-side fold is ``verl_omni.utils.reward_score.tts_reward.normalize_for_cer``
(Whisper EnglishTextNormalizer). The two halves are consistent, so a verbalized prompt and its
ASR transcript meet in the same comparison form for CER.
"""

from __future__ import annotations

import re

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# path must not END on '.' (else a sentence-final period gets swallowed: ".../membership. Go")
_URL_RE = re.compile(r"\b(?:https?://)?(?:[\w-]+\.)+[a-z]{2,6}(?:/[\w./~%+-]*[\w/~%+-])?", re.IGNORECASE)


def _spell(token: str) -> str:
    """Character-by-character spell-out for opaque codes/handles (case-insensitive speech)."""
    out = []
    for ch in token.lower():
        if ch.isdigit():
            out.append(_ONES[int(ch)])
        elif ch.isalpha():
            out.append(ch)
        # separators inside codes are silent
    return " ".join(out)


def _is_coined(piece: str) -> bool:
    """Heuristic: does an email/path token need spelling (coined handle) vs reading as a word?
    Digits, long consonant runs, or very long blobs => spell. 'settings'/'kane' read fine."""
    if any(c.isdigit() for c in piece):
        return True
    return re.search(r"[bcdfghjklmnpqrstvwxz]{4}", piece.lower()) is not None


def _verbalize_email(m: re.Match) -> str:
    local, domain = m.group(0).split("@", 1)
    parts = []
    for piece in re.split(r"[._+-]", local):
        if not piece:
            continue
        parts.append(_spell(piece) if _is_coined(piece) else piece)
    return " ".join(parts) + " at " + domain.replace(".", " dot ")


def _verbalize_url(m: re.Match) -> str:
    url = re.sub(r"^https?://", "", m.group(0))
    segs = url.split("/")
    host = segs[0].replace(".", " dot ")
    spoken = [host]
    for seg in segs[1:]:
        if not seg:
            continue
        spoken.append("slash")
        # URL path segments are real words (settings/account/membership) unless they carry a
        # digit (short-link blobs like "2fmMGTq"); only those get spelled out.
        spoken.append(_spell(seg) if any(c.isdigit() for c in seg) else seg)
    return " ".join(spoken)


def verbalize(text: str) -> str:
    """Written -> spoken form: emails first, then URLs. Deterministic, idempotent (output has no
    URL punctuation left for a second pass to rewrite)."""
    t = _EMAIL_RE.sub(_verbalize_email, text)
    t = _URL_RE.sub(_verbalize_url, t)
    return re.sub(r"\s+", " ", t).strip()
