# Copyright 2026 verl-omni contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import _collapse_interleaved_audio_video_tokens


def tokenizer(**overrides):
    names = ("vision_start", "audio_start", "video_pad", "audio_pad", "audio_end", "vision_end")
    mapping = {f"<|{name}|>": i for i, name in enumerate(names, 101)}
    mapping.update(overrides)
    return SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda tok: mapping.get(tok, 0))


SPAN = [101, 102, 103, 103, 104, 104, 103, 104, 105, 106]
RAW = [101, 103, 106]


@pytest.mark.parametrize("prefix,suffix", [([], []), ([7, 7], [8, 8]), ([1, 2, 3], [4, 5, 6])])
def test_complete_interleaved_span_preserves_surrounding_tokens(prefix, suffix):
    original = prefix + SPAN + suffix
    saved = original.copy()
    actual = _collapse_interleaved_audio_video_tokens(original, tokenizer())
    assert actual == prefix + RAW + suffix
    assert original == saved
    assert _collapse_interleaved_audio_video_tokens(actual, tokenizer()) == actual


def test_multiple_media_and_history_keep_order():
    original = [7] + SPAN + [8, 9, 10] + SPAN + [11]
    assert _collapse_interleaved_audio_video_tokens(original, tokenizer()) == [7] + RAW + [8, 9, 10] + RAW + [11]


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        [7, 7, 8],
        RAW,
        [101, 103, 103, 106],
        [102, 104, 104, 105],
        SPAN[:-1],
        SPAN[:-2],
        [101, 102, 103, 7, 104, 105, 106],
        [101, 102, 103, 105, 106],
        [101, 102, 104, 105, 106],
        [101, 102, 105, 106],
        [101, 102, 103, 104, 106, 105],
    ],
)
def test_non_interleaved_or_malformed_span_is_unchanged(tokens):
    assert _collapse_interleaved_audio_video_tokens(tokens, tokenizer()) == tokens


@pytest.mark.parametrize("value", [None, 0, 101])
def test_unknown_or_aliased_special_token_is_unchanged(value):
    assert _collapse_interleaved_audio_video_tokens(SPAN, tokenizer(**{"<|audio_end|>": value})) == SPAN
