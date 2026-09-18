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
"""CPU tests for diffusion attention backend validation."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.diffusion_attention import validate_attention_consistency


def test_attention_validation_rejects_unknown_backend():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"attn_backend": "typo"},
                "actor": {"strategy": "fsdp2"},
                "rollout": {"rollout_attn_backend": "TORCH_SDPA"},
            }
        }
    )
    with pytest.raises(ValueError, match="Unknown attn_backend"):
        validate_attention_consistency(config)
