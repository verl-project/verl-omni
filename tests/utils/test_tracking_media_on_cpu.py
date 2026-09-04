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
"""Tests for the declared-media-kind resolution used by the diffusion trainer dump."""

import pytest

from verl_omni.utils.tracking import resolve_is_video


class TestResolveIsVideo:
    def test_declared_video_wins_over_rank(self):
        # A short 3-frame video can share an image batch's rank; the declaration
        # keeps it classified as video.
        assert resolve_is_video(ndim=4, media_kind="video") is True

    def test_declared_image_wins_over_rank(self):
        assert resolve_is_video(ndim=5, media_kind="image") is False

    def test_declared_audio_is_not_video(self):
        assert resolve_is_video(ndim=2, media_kind="audio") is False

    def test_falls_back_to_rank_when_undeclared(self):
        assert resolve_is_video(ndim=5, media_kind=None) is True
        assert resolve_is_video(ndim=4, media_kind=None) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
