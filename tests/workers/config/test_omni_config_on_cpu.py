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
"""CPU tests for omni worker config dataclasses."""

from unittest.mock import patch

import pytest

from verl_omni.workers.config.omni.actor import (
    OmniLossConfig,
    VeOmniOmniActorConfig,
    VeOmniOmniEngineConfig,
    VeOmniOmniOptimizerConfig,
)
from verl_omni.workers.config.omni.model import OmniModelConfig


class TestOmniLossConfig:
    def test_defaults(self):
        cfg = OmniLossConfig()
        assert cfg.loss_mode == "dpo"
        assert cfg.beta == pytest.approx(0.1)
        assert cfg.label_smoothing == pytest.approx(0.0)
        assert cfg.loss_type == "sigmoid"
        assert cfg.reference_free is False
        assert cfg.average_log_prob is False

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"loss_mode": "gspo"}, "Unsupported omni loss_mode"),
            ({"loss_type": "hinge"}, "Invalid omni DPO loss_type"),
            ({"beta": 0.0}, "beta must be positive"),
        ],
    )
    def test_invalid_values_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OmniLossConfig(**kwargs)

    def test_ipo_loss_type_allowed(self):
        cfg = OmniLossConfig(loss_type="ipo", beta=0.2, reference_free=True)
        assert cfg.loss_type == "ipo"
        assert cfg.reference_free is True


class TestVeOmniOmniEngineConfig:
    def test_defaults(self):
        cfg = VeOmniOmniEngineConfig()
        assert cfg.strategy == "veomni"
        assert cfg.init_device == "meta"
        assert cfg.ulysses_parallel_size == 1
        assert cfg.model_dtype == "bfloat16"

    def test_non_veomni_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy='veomni'"):
            VeOmniOmniEngineConfig(strategy="fsdp")


class TestVeOmniOmniOptimizerConfig:
    def test_scheduler_types(self):
        for scheduler in ("constant", "linear", "cosine"):
            assert VeOmniOmniOptimizerConfig(lr_scheduler_type=scheduler).lr_scheduler_type == scheduler

    def test_invalid_scheduler_raises(self):
        with pytest.raises(ValueError, match="Invalid VeOmni lr_scheduler_type"):
            VeOmniOmniOptimizerConfig(lr_scheduler_type="warmup")


class TestVeOmniOmniActorConfig:
    def test_actor_wires_engine_config(self):
        actor_cfg = VeOmniOmniActorConfig(ppo_micro_batch_size_per_gpu=2, rollout_n=4)
        assert actor_cfg.strategy == "veomni"
        assert actor_cfg.engine is actor_cfg.veomni_config
        assert isinstance(actor_cfg.omni_loss, OmniLossConfig)
        assert isinstance(actor_cfg.optim, VeOmniOmniOptimizerConfig)

    def test_missing_rollout_n_raises(self):
        with pytest.raises(AssertionError):
            VeOmniOmniActorConfig(ppo_micro_batch_size_per_gpu=2)


class TestOmniModelConfig:
    def test_resolves_paths_without_loading_tokenizer(self):
        with patch(
            "verl_omni.workers.config.omni.model.resolve_model_local_dir",
            return_value="/tmp/local-omni-model",
        ):
            cfg = OmniModelConfig(path="remote/model", load_tokenizer=False, policy_state_adapters=("default", "old"))

        assert cfg.local_path == "/tmp/local-omni-model"
        assert cfg.config_path == "/tmp/local-omni-model"
        assert cfg.model_path == "/tmp/local-omni-model"
        assert cfg.tokenizer_path == "/tmp/local-omni-model"
        assert cfg.policy_state_adapters == ("default", "old")
        assert cfg.get_processor() is None

    def test_invalid_target_modules_type_raises(self):
        with patch(
            "verl_omni.workers.config.omni.model.resolve_model_local_dir",
            return_value="/tmp/local-omni-model",
        ):
            with pytest.raises(TypeError, match="target_modules must be a string or a list"):
                OmniModelConfig(path="remote/model", load_tokenizer=False, target_modules=object())
