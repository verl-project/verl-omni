# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.config import validate_config
from verl_omni.utils.diffusion_attention import validate_attention_consistency
from verl_omni.utils.reward_score.mmk12_reward import compute_score as mmk12_score
from verl_omni.utils.reward_score.unified_reward import _parse_unified_reward_scores


def _config(**trainer):
    return OmegaConf.create({"trainer": {"resume_mode": "disable", **trainer}})


def test_unified_reward_requires_all_labeled_axes():
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5") == {}
    assert _parse_unified_reward_scores(
        "Alignment Score: 4\nCoherence Score: 5\nStyle Score: 3"
    ) == {"alignment": 4.0, "coherence": 5.0, "style": 3.0}


def test_validate_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="Available options"):
        validate_config(_config(resume_mode="resumee"))


def test_validate_config_requires_resume_path():
    with pytest.raises(ValueError, match="resume_from_path"):
        validate_config(_config(resume_mode="resume_path"))


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


def test_mmk12_rejects_malformed_present_options():
    with pytest.raises(ValueError, match="valid JSON"):
        mmk12_score(
            "42",
            "A",
            extra_info={"options": "not-json"},
        )
