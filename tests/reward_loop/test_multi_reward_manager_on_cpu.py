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
"""CPU tests for MultiVisualRewardManager."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager, _filter_kwargs

# Path to this file — load_extern_object will import dummy functions from here.
DUMMY_REWARDS_PATH = "tests/reward_loop/test_multi_reward_manager_on_cpu.py"


# ---------------------------------------------------------------------------
# Dummy reward functions (loaded by MultiVisualRewardManager via load_extern_object)
# ---------------------------------------------------------------------------


def reward_fixed_score(data_source, solution_image, ground_truth, extra_info):
    """Always returns 0.5."""
    return 0.5


def reward_dict_result(data_source, ground_truth):
    """Returns a dict with score and extra metadata."""
    return {"score": 1.0, "detail": "perfect"}


def reward_raises(data_source, solution_image, ground_truth, extra_info):
    """Always raises to test error handling."""
    raise ValueError("intentional failure")


async def reward_async(data_source, solution_image, ground_truth, extra_info):
    """Async reward that returns 0.8."""
    return 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(reward_functions: dict):
    """Build a minimal config with reward_functions populated."""
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")

    config.reward.reward_functions = OmegaConf.create(reward_functions)
    config.reward.reward_model.enable = False
    return config


def _make_single_data() -> DataProto:
    """Create a single-item DataProto for run_single."""
    return DataProto.from_dict(
        tensors={"responses": torch.randn(1, 3, 64, 64)},
        non_tensors={
            "data_source": ["test_source"],
            "reward_model": [{"ground_truth": "hello"}],
            "extra_info": [{}],
        },
    )


def _build_manager(reward_functions: dict) -> MultiVisualRewardManager:
    config = _make_config(reward_functions)
    tokenizer = MagicMock()
    return MultiVisualRewardManager(config, tokenizer, compute_score=None)


# ---------------------------------------------------------------------------
# _filter_kwargs
# ---------------------------------------------------------------------------


class TestFilterKwargs:
    def test_filters_to_declared_params(self):
        import inspect

        def fn(a, b):
            pass

        sig = inspect.signature(fn)
        result = _filter_kwargs({"a": 1, "b": 2, "c": 3}, sig)
        assert result == {"a": 1, "b": 2}

    def test_passes_all_when_var_keyword(self):
        import inspect

        def fn(a, **kwargs):
            pass

        sig = inspect.signature(fn)
        result = _filter_kwargs({"a": 1, "b": 2, "c": 3}, sig)
        assert result == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# MultiVisualRewardManager.run_single
# ---------------------------------------------------------------------------


class TestMultiVisualRewardManagerRunSingle:
    def test_weighted_aggregation(self):
        """Two reward functions with different weights produce correct combined score."""
        reward_fns = {
            "fixed": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 2.0},
            "dict_result": {"path": DUMMY_REWARDS_PATH, "name": "reward_dict_result", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        result = manager.loop.run_until_complete(manager.run_single(data))

        # combined = 2.0 * 0.5 + 1.0 * 1.0 = 2.0
        assert result["reward_score"] == pytest.approx(2.0)
        assert result["reward_extra_info"]["reward/fixed"] == pytest.approx(0.5)
        assert result["reward_extra_info"]["reward/dict_result"] == pytest.approx(1.0)
        assert result["reward_extra_info"]["reward/dict_result/detail"] == "perfect"
        assert result["reward_extra_info"]["reward/combined"] == pytest.approx(2.0)

    def test_exception_contributes_zero(self):
        """A failing sub-reward contributes 0 without breaking others."""
        reward_fns = {
            "good": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0},
            "bad": {"path": DUMMY_REWARDS_PATH, "name": "reward_raises", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        result = manager.loop.run_until_complete(manager.run_single(data))

        # combined = 1.0 * 0.5 + 1.0 * 0.0 = 0.5
        assert result["reward_score"] == pytest.approx(0.5)
        assert result["reward_extra_info"]["reward/bad"] == pytest.approx(0.0)

    def test_async_reward_function(self):
        """Async reward functions are awaited correctly."""
        reward_fns = {
            "async_fn": {"path": DUMMY_REWARDS_PATH, "name": "reward_async", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.8)


class TestMultiVisualRewardManagerInit:
    def test_empty_reward_functions_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _build_manager({})

    def test_loads_sub_rewards(self):
        reward_fns = {
            "a": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 0.5},
        }
        manager = _build_manager(reward_fns)
        assert len(manager._sub_rewards) == 1
        assert manager._sub_rewards[0]["key"] == "a"
        assert manager._sub_rewards[0]["weight"] == 0.5
