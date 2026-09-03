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

import logging

from omegaconf import OmegaConf

from tests.utils.smoke_attention import resolve_smoke_attention_backends
from verl_omni.utils.diffusion_attention import fallback_fa3_if_unavailable, validate_attention_consistency


def test_veomni_strategy_skips_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="verl_omni.utils.diffusion_attention"):
        validate_attention_consistency(
            OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "model": {"attn_backend": "native"},
                        "actor": {"strategy": "veomni"},
                        "rollout": {"rollout_attn_backend": "FLASH_ATTN"},
                    }
                }
            )
        )
    assert "veomni" in caplog.text


def test_fa3_actor_allows_flash_attn_3_hub():
    validate_attention_consistency(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": "_flash_3_varlen_hub"},
                    "actor": {"strategy": "fsdp2"},
                    "rollout": {"rollout_attn_backend": "FLASH_ATTN_3_HUB"},
                }
            }
        )
    )


def test_fa3_actor_allows_flash_attn():
    validate_attention_consistency(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": "_flash_3_varlen_hub"},
                    "actor": {"strategy": "fsdp2"},
                    "rollout": {"rollout_attn_backend": "FLASH_ATTN"},
                }
            }
        )
    )


def test_native_actor_rejects_flash_attn_3_hub():
    try:
        validate_attention_consistency(
            OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "model": {"attn_backend": "native"},
                        "actor": {"strategy": "fsdp2"},
                        "rollout": {"rollout_attn_backend": "FLASH_ATTN_3_HUB"},
                    }
                }
            )
        )
    except ValueError as exc:
        assert "rollout_attn_backend='TORCH_SDPA'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_fallback_hub_fa3_without_kernels_sets_sdpa(monkeypatch):
    monkeypatch.setattr("verl_omni.utils.diffusion_attention.actor_fa3_available", lambda: False)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"attn_backend": "_flash_3_varlen_hub"},
                "rollout": {"rollout_attn_backend": "FLASH_ATTN_3_HUB"},
            }
        }
    )
    fallback_fa3_if_unavailable(config)
    assert config.actor_rollout_ref.model.attn_backend == "native"
    assert config.actor_rollout_ref.rollout.rollout_attn_backend == "TORCH_SDPA"


def test_resolve_smoke_attention_backends_prefers_local_fa(monkeypatch):
    monkeypatch.setattr("tests.utils.smoke_attention.fa3_available", lambda: True)
    assert resolve_smoke_attention_backends() == ("_flash_3_varlen_hub", "FLASH_ATTN")


def test_resolve_smoke_attention_backends_falls_back_without_fa3(monkeypatch):
    monkeypatch.setattr("tests.utils.smoke_attention.fa3_available", lambda: False)
    assert resolve_smoke_attention_backends() == ("native", "TORCH_SDPA")
