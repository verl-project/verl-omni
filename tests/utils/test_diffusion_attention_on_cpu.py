# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import patch

from verl_omni.utils import diffusion_attention as da


def test_sm90_supports_rollout_flash_attention():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        assert da._cuda_supports_rollout_fa3()


def test_sm120_does_not_support_rollout_flash_attention():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(12, 0)),
    ):
        assert not da._cuda_supports_rollout_fa3()


def test_rollout_keeps_flash_attention_when_available():
    with patch.object(da, "rollout_fa3_available", return_value=True):
        assert da.fallback_rollout_fa3_if_unavailable("FLASH_ATTN") == "FLASH_ATTN"


def test_rollout_falls_back_when_flash_attention_is_unavailable():
    with patch.object(da, "rollout_fa3_available", return_value=False):
        assert da.fallback_rollout_fa3_if_unavailable("FLASH_ATTN") == "TORCH_SDPA"
