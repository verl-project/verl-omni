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

"""CPU tests for HPSv3 reward input handling."""

import pytest
import torch

hpsv3_reward = pytest.importorskip("verl_omni.utils.reward_score.hpsv3_reward")


def _assert_rgb_frames(frames, *, count, size):
    assert len(frames) == count
    assert all(frame.mode == "RGB" and frame.size == size for frame in frames)


@pytest.mark.parametrize("shape", [(3, 480, 832), (480, 832, 3)])
def test_extract_frames_handles_single_image(shape):
    frames = hpsv3_reward._extract_frames(torch.rand(*shape))

    _assert_rgb_frames(frames, count=1, size=(832, 480))


def test_extract_frames_handles_tchw_video():
    frames = hpsv3_reward._extract_frames(torch.rand(5, 3, 480, 832), frame_interval=2)

    _assert_rgb_frames(frames, count=3, size=(832, 480))


def test_extract_frames_handles_thwc_video():
    frames = hpsv3_reward._extract_frames(torch.rand(5, 480, 832, 3), frame_interval=2)

    _assert_rgb_frames(frames, count=3, size=(832, 480))


@pytest.mark.parametrize("shape", [(2, 4, 3, 16, 16), (2, 4, 16, 16, 3)])
def test_extract_frames_handles_batched_video(shape):
    frames = hpsv3_reward._extract_frames(torch.rand(*shape), frame_interval=2)

    _assert_rgb_frames(frames, count=4, size=(16, 16))


def test_compute_score_hpsv3_uses_env_checkpoint_and_device(monkeypatch):
    calls = {}

    class _FakeInferencer:
        def reward(self, images, prompts):
            calls["images"] = images
            calls["prompts"] = prompts
            return torch.tensor([[5.0]])

    def _fake_get_inferencer(checkpoint_path, device):
        calls["checkpoint_path"] = checkpoint_path
        calls["device"] = device
        return _FakeInferencer()

    monkeypatch.setenv("custom_reward_model_path", "/tmp/hpsv3.safetensors")
    monkeypatch.setattr(hpsv3_reward, "_get_inferencer", _fake_get_inferencer)

    result = hpsv3_reward.compute_score_hpsv3(
        data_source="unit-test",
        solution_image=torch.rand(3, 16, 16),
        ground_truth="a prompt",
        extra_info={"frame_interval": 1},
        device="cuda:3",
        reward_scale=0.1,
    )

    assert calls["checkpoint_path"] == "/tmp/hpsv3.safetensors"
    assert calls["device"] == "cuda:3"
    assert calls["prompts"] == ["a prompt"]
    assert isinstance(calls["images"][0], hpsv3_reward.Image.Image)
    assert result == {"score": pytest.approx(0.5), "hpsv3_raw": pytest.approx(5.0)}
