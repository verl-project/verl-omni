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
        # A per-sample video and a batched image can both have rank four.
        assert resolve_is_video(ndim=4, media_kind="video") is True

    def test_declared_image_wins_over_rank(self):
        assert resolve_is_video(ndim=5, media_kind="image") is False

    def test_declared_audio_is_not_video(self):
        assert resolve_is_video(ndim=2, media_kind="audio") is False

    def test_unknown_media_kind_is_rejected(self):
        with pytest.raises(ValueError, match="Explicit media_kind required"):
            resolve_is_video(ndim=5, media_kind="depth")

    @pytest.mark.parametrize("rank", [4, 5])
    def test_never_falls_back_to_rank_when_undeclared(self, rank):
        with pytest.raises(ValueError, match="Explicit media_kind required"):
            resolve_is_video(ndim=rank, media_kind=None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
