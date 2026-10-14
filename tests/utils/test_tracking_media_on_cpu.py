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

from verl_omni.pipelines.rollout_media import resolve_is_video


class TestResolveIsVideo:
    @pytest.mark.parametrize("kind,is_video", [("video", True), ("image", False), ("audio", False)])
    def test_reads_declared_kind(self, kind, is_video):
        assert resolve_is_video(kind) is is_video

    def test_unknown_media_kind_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported media kind"):
            resolve_is_video("depth")

    def test_undeclared_media_kind_is_rejected(self):
        with pytest.raises(ValueError, match="Explicit media_kind required"):
            resolve_is_video(None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
