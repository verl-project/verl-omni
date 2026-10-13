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
"""CPU tests for modality-neutral multi-reward aggregation."""

import inspect
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.multi import MultiRewardManager, MultiVisualRewardManager, _filter_kwargs
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager

# Path to this file — load_extern_object will import dummy functions from here.
DUMMY_REWARDS_PATH = "tests/reward_loop/test_multi_reward_manager_on_cpu.py"


# ---------------------------------------------------------------------------
# Dummy reward functions (loaded by MultiRewardManager via load_extern_object)
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


async def reward_asserts_uint8_contract(data_source, solution_image, ground_truth, extra_info):
    """Verify reward managers preserve the uint8 response contract."""
    assert solution_image.dtype == torch.uint8
    return int(solution_image[0, 0, 0])


async def reward_asserts_float_latent_contract(data_source, solution_image, ground_truth, extra_info):
    """Verify reward managers preserve floating-point latent responses."""
    assert solution_image.dtype == torch.float32
    assert solution_image.shape == (16, 2, 2)
    return float(solution_image[0, 0, 0])


async def reward_uses_named_engine_router(reward_router_address, model_name):
    assert reward_router_address == "engine-router"
    assert model_name == "ocr-model"
    return {"score": 0.6, "backend": "engine-function"}


async def reward_captures_ocr_request(
    data_source,
    solution_image,
    ground_truth,
    extra_info,
    reward_router_address,
    model_name,
    sampling_params,
    reward_model_tokenizer=None,
):
    del data_source, solution_image, ground_truth, extra_info, reward_model_tokenizer
    return {
        "score": 0.5,
        "request": {
            "reward_router_address": reward_router_address,
            "model_name": model_name,
            "sampling_params": sampling_params,
        },
    }


async def reward_captures_sampling_params(**kwargs):
    return {"score": 0.5, "has_sampling_params": "sampling_params" in kwargs}


async def reward_uses_legacy_router(reward_router_address, model_name, sampling_params):
    assert reward_router_address == "legacy-router"
    assert model_name == "legacy-model"
    assert sampling_params == {"max_tokens": 321, "seed": 17}
    return 0.65


async def reward_uses_native_model(reward_model, ground_truth, solution_image):
    output = await reward_model.infer(prompt=ground_truth, image=solution_image)
    return {"score": output["value"], "backend": "native-function"}


class _NativeModelExecutor:
    def reward_kwargs(self):
        return {"reward_model": self}

    async def infer(self, prompt, image):
        assert prompt == "hello"
        assert image.dtype == torch.uint8
        return {"value": 0.75}


class _EngineRouterClient:
    def reward_kwargs(self):
        return {"reward_router_address": "engine-router", "model_name": "ocr-model"}


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


def _make_single_data(data_source: str = "test_source") -> DataProto:
    """Create a single-item DataProto for run_single."""
    return DataProto.from_dict(
        tensors={"responses": torch.randint(256, (1, 3, 64, 64), dtype=torch.uint8)},
        non_tensors={
            "data_source": [data_source],
            "reward_model": [{"ground_truth": "hello"}],
            "extra_info": [{}],
        },
    )


def _build_manager(reward_functions: dict) -> MultiVisualRewardManager:
    config = _make_config(reward_functions)
    tokenizer = MagicMock()
    return MultiVisualRewardManager(config, tokenizer, compute_score=None)


def _build_visual_latent_manager() -> VisualRewardManager:
    config = _make_config({})
    OmegaConf.update(config, "actor_rollout_ref.rollout.pipeline.output_type", "latent", force_add=True)
    return VisualRewardManager(config, MagicMock(), reward_asserts_float_latent_contract)


def _build_latent_multi_manager() -> MultiVisualRewardManager:
    manager = _build_manager(
        {
            "latent": {
                "path": DUMMY_REWARDS_PATH,
                "name": "reward_asserts_float_latent_contract",
                "weight": 1.0,
            }
        }
    )
    OmegaConf.update(manager.config, "actor_rollout_ref.rollout.pipeline.output_type", "latent", force_add=True)
    return manager


def _build_visual_pixel_manager() -> VisualRewardManager:
    return VisualRewardManager(_make_config({}), MagicMock(), reward_asserts_uint8_contract)


def _build_pixel_multi_manager() -> MultiVisualRewardManager:
    return _build_manager(
        {
            "pixel": {
                "path": DUMMY_REWARDS_PATH,
                "name": "reward_asserts_uint8_contract",
                "weight": 1.0,
            }
        }
    )


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
# MultiRewardManager.run_single
# ---------------------------------------------------------------------------


class TestVisualRewardManagerDefaults:
    def test_default_jpeg_reward_accepts_manager_call_contract(self):
        manager = VisualRewardManager(_make_config({}), MagicMock(), compute_score=None)
        data = _make_single_data(data_source="jpeg_compressibility")
        data.batch["responses"] = torch.zeros_like(data.batch["responses"])

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] < 0
        assert result["reward_extra_info"]["acc"] == pytest.approx(result["reward_score"])


class TestMultiRewardManagerRunSingle:
    @pytest.mark.parametrize("deterministic", [False, True])
    @pytest.mark.parametrize("explicit_sampling", [None, {"max_tokens": 17, "seed": 9, "temperature": 0.2}])
    def test_fallback_ocr_preserves_upstream_sampling_and_explicit_overrides(
        self, monkeypatch, deterministic, explicit_sampling
    ):
        from verl_omni.utils.reward_score import genrm_ocr

        requests = []

        async def chat_complete(router_address, chat_complete_request):
            assert router_address == "engine-router"
            requests.append(chat_complete_request)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

        monkeypatch.setenv("GENRM_OCR_SEED", "777")
        monkeypatch.delenv("GENRM_OCR_MAX_TOKENS", raising=False)
        monkeypatch.setattr(genrm_ocr, "_chat_complete", chat_complete)
        entry = {"path": "pkg://verl_omni.utils.reward_score.genrm_ocr", "name": "compute_score_ocr"}
        if explicit_sampling is not None:
            entry["sampling_params"] = explicit_sampling
        config = _make_config({"ocr": entry})
        config.reward.reward_model.enable = True
        config.reward.reward_model.model_path = "ocr-model"
        config.reward.reward_model.rollout.response_length = 2048
        config.reward.reward_model.rollout.full_determinism = deterministic
        config.reward.reward_model.rollout.seed = 42
        manager = MultiVisualRewardManager(
            config, MagicMock(), compute_score=None, reward_router_address="engine-router"
        )
        data = _make_single_data()

        expected_sampling = {"max_tokens": 2048}
        if deterministic:
            expected_sampling["seed"] = 42
        if explicit_sampling is not None:
            expected_sampling = explicit_sampling
        previous = manager.loop.run_until_complete(
            genrm_ocr.compute_score_ocr(
                data_source="test_source",
                solution_image=data[0].batch["responses"],
                ground_truth="hello",
                extra_info={},
                reward_router_address="engine-router",
                model_name="ocr-model",
                sampling_params=expected_sampling,
            )
        )
        current = manager.loop.run_until_complete(manager.run_single(data))

        assert len(requests) == 2
        assert requests[1] == requests[0]
        assert requests[0]["max_tokens"] == (2048 if explicit_sampling is None else 17)
        if explicit_sampling is not None:
            assert requests[0]["seed"] == 9
        elif deterministic:
            assert requests[0]["seed"] == 42
        else:
            assert "seed" not in requests[0]
        assert current["reward_score"] == previous["score"] == 1.0

    @pytest.mark.parametrize("deterministic", [False, True])
    @pytest.mark.parametrize("inherit", [False, True])
    def test_existing_named_ocr_preserves_scorer_defaults_unless_recipe_opts_in(
        self, monkeypatch, deterministic, inherit
    ):
        from verl_omni.utils.reward_score import genrm_ocr

        requests = []

        async def chat_complete(router_address, chat_complete_request):
            requests.append(chat_complete_request)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="test"))])

        monkeypatch.setenv("GENRM_OCR_SEED", "777")
        monkeypatch.delenv("GENRM_OCR_MAX_TOKENS", raising=False)
        monkeypatch.setattr(genrm_ocr, "_chat_complete", chat_complete)
        entry = {"path": "pkg://verl_omni.utils.reward_score.genrm_ocr", "name": "compute_score_ocr"}
        if inherit:
            entry["use_rollout_sampling_params"] = True
        config = _make_config({"ocr": entry})
        config.reward.reward_model.rollout.response_length = 2048
        config.reward.reward_model.rollout.full_determinism = deterministic
        config.reward.reward_model.rollout.seed = 42
        config.reward.models = OmegaConf.create({"ocr": {"backend": "engine"}})
        manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
        manager.set_reward_executors({"ocr": _EngineRouterClient()}, None)
        data = _make_single_data()

        # The original named path passed router/model only, leaving sampling to the scorer.
        item = data[0]
        manager.loop.run_until_complete(
            genrm_ocr.compute_score_ocr(
                data_source="test",
                solution_image=item.batch["responses"],
                ground_truth="test",
                extra_info={},
                reward_router_address="engine-router",
                model_name="ocr-model",
            )
        )
        manager.loop.run_until_complete(manager.run_single(data))
        previous, current = requests
        assert previous["max_tokens"] == 4096
        assert previous["seed"] == 777
        if not inherit:
            assert current == previous
        else:
            assert current["max_tokens"] == 2048
            if deterministic:
                assert current["seed"] == 42
            else:
                assert "seed" not in current

    def test_named_engine_preserves_explicit_scorer_sampling_params(self):
        explicit = {"max_tokens": 7, "temperature": 0.2}
        config = _make_config(
            {
                "ocr": {
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_captures_ocr_request",
                    "sampling_params": explicit,
                    "use_rollout_sampling_params": True,
                }
            }
        )
        config.reward.models = OmegaConf.create(
            {"ocr": {"backend": "engine", "rollout": {"response_length": 321, "full_determinism": True}}}
        )
        manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
        manager.set_reward_executors({"ocr": _EngineRouterClient()}, None)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_extra_info"]["reward/ocr/request"]["sampling_params"] == explicit

    def test_named_engine_sampling_resolves_config_interpolations(self):
        config = _make_config(
            {
                "ocr": {
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_captures_ocr_request",
                    "use_rollout_sampling_params": True,
                }
            }
        )
        config.reward.reward_model.rollout.response_length = "${data.max_response_length}"
        config.reward.models = OmegaConf.create(
            {"ocr": {"backend": "engine", "rollout": {"full_determinism": True, "seed": "${data.seed}"}}}
        )
        manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
        manager.set_reward_executors({"ocr": _EngineRouterClient()}, None)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_extra_info"]["reward/ocr/request"]["sampling_params"] == {
            "max_tokens": config.data.max_response_length,
            "seed": config.data.seed,
        }

    def test_named_engine_ocr_request_matches_legacy_sampling_contract(self):
        legacy_config = _make_config({})
        legacy_config.reward.reward_model.enable = True
        legacy_config.reward.reward_model.model_path = "ocr-model"
        legacy_config.reward.reward_model.rollout.response_length = 32
        legacy_config.reward.reward_model.rollout.full_determinism = True
        legacy_config.reward.reward_model.rollout.seed = 123
        legacy_manager = VisualRewardManager(
            legacy_config,
            MagicMock(),
            reward_captures_ocr_request,
            reward_router_address="engine-router",
        )

        named_config = _make_config(
            {
                "ocr": {
                    "model": "ocr_engine",
                    "use_rollout_sampling_params": True,
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_captures_ocr_request",
                }
            }
        )
        named_config.reward.models = OmegaConf.create(
            {
                "ocr_engine": {
                    "backend": "engine",
                    "model_path": "ocr-model",
                    "rollout": {"response_length": 32, "full_determinism": True, "seed": 123},
                }
            }
        )
        named_config.reward.reward_model.rollout.response_length = 64
        named_config.reward.reward_model.rollout.full_determinism = True
        named_config.reward.reward_model.rollout.seed = 777
        named_manager = MultiVisualRewardManager(named_config, MagicMock(), compute_score=None)
        named_manager.set_reward_executors({"ocr_engine": _EngineRouterClient()}, None)

        legacy_result = legacy_manager.loop.run_until_complete(legacy_manager.run_single(_make_single_data()))
        named_result = named_manager.loop.run_until_complete(named_manager.run_single(_make_single_data()))

        assert named_result["reward_score"] == legacy_result["reward_score"]
        assert named_result["reward_extra_info"]["reward/ocr"] == legacy_result["reward_score"]
        assert named_result["reward_extra_info"]["reward/ocr/request"] == legacy_result["reward_extra_info"]["request"]

    def test_named_native_scorer_does_not_receive_engine_sampling_params(self):
        config = _make_config(
            {
                "quality": {
                    "model": "quality_model",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_captures_sampling_params",
                }
            }
        )
        config.reward.models = OmegaConf.create({"quality_model": {"backend": "native"}})
        manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
        manager.set_reward_executors(None, {"quality_model": MagicMock(reward_kwargs=MagicMock(return_value={}))})

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_extra_info"]["reward/quality/has_sampling_params"] is False

    @pytest.mark.parametrize("manager_factory", [_build_visual_latent_manager, _build_latent_multi_manager])
    def test_float_latent_response_is_forwarded_to_reward(self, manager_factory):
        manager = manager_factory()
        data = _make_single_data()
        data.batch["responses"] = torch.full((1, 16, 2, 2), 0.5)

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.5)

    @pytest.mark.parametrize("manager_factory", [_build_visual_pixel_manager, _build_pixel_multi_manager])
    def test_float_pixel_response_is_rejected(self, manager_factory):
        manager = manager_factory()
        data = _make_single_data()
        data.batch["responses"] = data.batch["responses"].float()

        with pytest.raises(
            ValueError,
            match=r"Expected uint8 pixel responses for output_type='image', got torch\.float32\.",
        ):
            manager.loop.run_until_complete(manager.run_single(data))

    @pytest.mark.parametrize("manager_factory", [_build_visual_latent_manager, _build_latent_multi_manager])
    def test_uint8_latent_response_is_rejected(self, manager_factory):
        manager = manager_factory()
        data = _make_single_data()

        with pytest.raises(ValueError, match=r"Expected floating-point latent responses, got torch\.uint8\."):
            manager.loop.run_until_complete(manager.run_single(data))

    @pytest.mark.parametrize("manager_factory", [_build_visual_pixel_manager, _build_pixel_multi_manager])
    def test_validation_uses_validation_output_type(self, manager_factory):
        manager = manager_factory()
        OmegaConf.update(
            manager.config,
            "actor_rollout_ref.rollout.val_kwargs.pipeline.output_type",
            "latent",
            force_add=True,
        )
        manager.compute_score = reward_asserts_float_latent_contract
        manager.is_async_reward_score = True
        if isinstance(manager, MultiRewardManager):
            manager._sub_rewards[0]["fn"] = reward_asserts_float_latent_contract
            manager._sub_rewards[0]["is_async"] = True

        data = _make_single_data()
        data.batch["responses"] = torch.full((1, 16, 2, 2), 0.5)
        data.meta_info["validate"] = True

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.5)

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
        assert "reward/dict_result/score" not in result["reward_extra_info"]
        assert result["reward_extra_info"]["reward/combined"] == pytest.approx(2.0)

    def test_required_exception_fails_fast(self):
        """A failing required sub-reward aborts reward computation."""
        reward_fns = {
            "good": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0},
            "bad": {
                "path": DUMMY_REWARDS_PATH,
                "name": "reward_raises",
                "weight": 1.0,
                "required": True,
            },
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        with pytest.raises(RuntimeError, match="Required sub-reward 'bad' failed: intentional failure"):
            manager.loop.run_until_complete(manager.run_single(data))

    def test_optional_exception_contributes_zero(self):
        """A failing optional sub-reward records the error and contributes zero."""
        reward_fns = {
            "good": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 2.0},
            "bad": {"path": DUMMY_REWARDS_PATH, "name": "reward_raises", "weight": 3.0},
        }
        manager = _build_manager(reward_fns)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(1.0)
        assert result["reward_extra_info"]["reward/bad"] == pytest.approx(0.0)
        assert result["reward_extra_info"]["reward/bad/errors"] == 1

    def test_async_reward_function(self):
        """Async reward functions are awaited correctly."""
        reward_fns = {
            "async_fn": {"path": DUMMY_REWARDS_PATH, "name": "reward_async", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(0.8)

    def test_mixes_rule_engine_and_native_models(self):
        manager = _build_manager(
            {
                "rule": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": 1.0},
                "engine": {
                    "model": "ocr_engine",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_named_engine_router",
                    "weight": 2.0,
                },
                "native": {
                    "model": "native_pickscore",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_native_model",
                    "weight": 1.0,
                },
            }
        )
        manager.set_reward_executors(
            {"ocr_engine": _EngineRouterClient()}, {"native_pickscore": _NativeModelExecutor()}
        )

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(2.45)
        assert result["reward_extra_info"]["reward/rule"] == pytest.approx(0.5)
        assert result["reward_extra_info"]["reward/engine"] == pytest.approx(0.6)
        assert result["reward_extra_info"]["reward/native"] == pytest.approx(0.75)
        assert result["reward_extra_info"]["reward/engine/backend"] == "engine-function"
        assert result["reward_extra_info"]["reward/native/backend"] == "native-function"

    def test_native_reward_function_receives_only_inference_handle(self):
        manager = _build_manager(
            {
                "native": {
                    "model": "native_model",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_native_model",
                }
            }
        )
        manager.set_reward_executors(None, {"native_model": _NativeModelExecutor()})

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.75)
        assert result["reward_extra_info"]["reward/native/backend"] == "native-function"

    def test_engine_reward_function_uses_its_named_router(self):
        manager = _build_manager(
            {
                "ocr": {
                    "model": "ocr_engine",
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_uses_named_engine_router",
                    "weight": 1.0,
                },
            }
        )
        manager.set_reward_executors({"ocr_engine": _EngineRouterClient()}, None)

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.6)
        assert result["reward_extra_info"]["reward/ocr/backend"] == "engine-function"

    def test_jpeg_reward_via_file_path(self):
        """JPEG reward loaded via file path must not fail on relative imports."""
        reward_fns = {
            "jpeg": {
                "path": "verl_omni/utils/reward_score/jpeg_compressibility.py",
                "name": "compute_score",
                "weight": 1.0,
            },
        }
        manager = _build_manager(reward_fns)
        data = DataProto.from_dict(
            tensors={"responses": torch.randint(256, (1, 3, 64, 64), dtype=torch.uint8)},
            non_tensors={
                "data_source": ["jpeg_compressibility"],
                "reward_model": [{"ground_truth": "hello"}],
                "extra_info": [{}],
            },
        )

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] != pytest.approx(0.0)
        assert "reward/jpeg" in result["reward_extra_info"]

    def test_uint8_response_is_forwarded_to_custom_reward(self):
        reward_fns = {
            "contract": {"path": DUMMY_REWARDS_PATH, "name": "reward_asserts_uint8_contract", "weight": 1.0},
        }
        manager = _build_manager(reward_fns)
        data = _make_single_data()
        data.batch["responses"] = torch.full_like(data.batch["responses"], 128, dtype=torch.uint8)

        result = manager.loop.run_until_complete(manager.run_single(data))

        assert result["reward_score"] == pytest.approx(128)

    def test_legacy_visual_router_preserves_sampling_params(self):
        config = _make_config(
            {"legacy": {"path": DUMMY_REWARDS_PATH, "name": "reward_uses_legacy_router", "weight": 1.0}}
        )
        config.reward.reward_model.model_path = "legacy-model"
        config.reward.reward_model.rollout.response_length = 321
        config.reward.reward_model.rollout.full_determinism = True
        config.reward.reward_model.rollout.seed = 17
        manager = MultiVisualRewardManager(
            config,
            MagicMock(),
            compute_score=None,
            reward_router_address="legacy-router",
            reward_model_tokenizer=MagicMock(),
        )

        result = manager.loop.run_until_complete(manager.run_single(_make_single_data()))

        assert result["reward_score"] == pytest.approx(0.65)


class TestMultiRewardManagerInit:
    def test_shared_manager_is_an_input_agnostic_base(self):
        assert inspect.isabstract(MultiRewardManager)
        assert issubclass(MultiVisualRewardManager, MultiRewardManager)

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
        assert manager._sub_rewards[0]["required"] is False

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [(True, True), (False, False), ("true", True), ("false", False)],
    )
    def test_parses_required(self, configured, expected):
        manager = _build_manager(
            {
                "a": {
                    "path": DUMMY_REWARDS_PATH,
                    "name": "reward_fixed_score",
                    "required": configured,
                }
            }
        )

        assert manager._sub_rewards[0]["required"] is expected

    @pytest.mark.parametrize("configured", ["yes", 1, None])
    def test_rejects_invalid_required(self, configured):
        with pytest.raises((TypeError, ValueError), match="required"):
            _build_manager(
                {
                    "a": {
                        "path": DUMMY_REWARDS_PATH,
                        "name": "reward_fixed_score",
                        "required": configured,
                    }
                }
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("weight, required", [(0.5, True), (1.0, False)])
async def test_component_rewards_require_unit_weight_and_required_terms(weight, required):
    config = _make_config(
        {"quality": {"path": DUMMY_REWARDS_PATH, "name": "reward_fixed_score", "weight": weight, "required": required}}
    )
    config.reward.aggregation = "preserve_components"
    with pytest.raises(ValueError, match="required=true and weight=1.0"):
        MultiVisualRewardManager(config, MagicMock(), compute_score=None)


async def reward_explicit_batch(batch, **kwargs):
    assert len(batch) == 1
    assert batch[0].non_tensor_batch["reward_model"]["ground_truth"] == "hello"
    return {"score": float(batch.batch["audio"].mean())}


async def reward_kwargs_without_batch(**kwargs):
    assert "batch" not in kwargs
    return {"score": 0.25}


@pytest.mark.asyncio
async def test_batch_injection_is_signature_gated():
    manager = _build_manager(
        {
            "explicit": {"path": DUMMY_REWARDS_PATH, "name": "reward_explicit_batch", "required": True},
            "legacy": {"path": DUMMY_REWARDS_PATH, "name": "reward_kwargs_without_batch", "required": True},
        }
    )
    data = _make_single_data()
    data.batch["audio"] = torch.full((1, 1, 8), 0.5)
    result = await manager.run_single(data)
    assert result["reward_score"] == pytest.approx(0.75)
    assert result["reward_extra_info"]["reward/explicit"] == pytest.approx(0.5)
    assert result["reward_extra_info"]["reward/legacy"] == pytest.approx(0.25)


def _native_av_manager(scorer, *, fps=24.0, include_fps=True, bind_model=True):
    from unittest.mock import AsyncMock

    entry = {
        "path": f"pkg://verl_omni.utils.reward_score.{scorer}",
        "name": "compute_score",
        "required": False,
        "audio_layout": "CT",
    }
    if include_fps:
        entry["fps"] = fps
    if bind_model:
        entry["model"] = "av_model"
    config = _make_config({"av": entry})
    config.reward.models = OmegaConf.create({"av_model": {"backend": "native"}} if bind_model else {})
    manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
    logits = torch.zeros((2, 1, 21) if scorer == "desync" else (1, 3))
    if scorer == "desync":
        logits[:, :, 10] = 1
    model = SimpleNamespace(infer=AsyncMock(return_value=logits))
    manager.set_reward_executors(None, {"av_model": SimpleNamespace(reward_kwargs=lambda: {"reward_model": model})})
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(1, 4, 3, 4, 4, dtype=torch.uint8),
            "audio": torch.ones(1, 1, 480),
            "audio_sample_rate": torch.tensor([24000]),
        },
        non_tensors={"data_source": ["test"], "reward_model": [{"ground_truth": "a video"}], "extra_info": [{}]},
    )
    return manager, model, data


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", ["desync", "videoalign"])
@pytest.mark.parametrize("fps", [24, 23.976, "23.976"])
async def test_av_fps_config_reaches_real_manager_without_rollout_fps(scorer, fps):
    manager, model, data = _native_av_manager(scorer, fps=fps)
    assert "fps" not in data.batch and "fps" not in data.non_tensor_batch
    result = await manager.run_single(data)
    assert model.infer.await_count == 1
    assert "reward/av/errors" not in result["reward_extra_info"]
    assert isinstance(result["reward_score"], float)


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", ["desync", "videoalign"])
async def test_missing_fps_config_is_fatal_even_for_optional_av_reward(scorer):
    manager, model, data = _native_av_manager(scorer, include_fps=False)
    with pytest.raises(ValueError, match="Invalid arguments.*fps"):
        await manager.run_single(data)
    model.infer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", ["desync", "videoalign"])
@pytest.mark.parametrize(
    "fps", [None, 0, -1, float("nan"), float("inf"), -float("inf"), "invalid", True, False, [], {}]
)
async def test_invalid_fps_config_is_fatal_even_for_optional_av_reward(scorer, fps):
    manager, model, data = _native_av_manager(scorer, fps=fps)
    with pytest.raises(ValueError, match="Invalid arguments.*av.*fps.*finite and positive"):
        await manager.run_single(data)
    model.infer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", ["desync", "videoalign"])
async def test_optional_av_inference_failure_still_contributes_zero(scorer):
    manager, model, data = _native_av_manager(scorer)
    model.infer.side_effect = RuntimeError("model inference failed")

    result = await manager.run_single(data)

    model.infer.assert_awaited_once()
    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"]["reward/av/errors"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer", ["desync", "videoalign", "audiobox"])
async def test_missing_native_model_binding_reports_configuration_error(scorer):
    manager, model, data = _native_av_manager(scorer, bind_model=False)
    with pytest.raises(ValueError, match="Invalid arguments.*reward_model"):
        await manager.run_single(data)
    model.infer.assert_not_awaited()


async def reward_missing_legacy_argument(custom_option):
    return float(custom_option)


async def reward_legacy_fps(fps):
    return float(fps)


@pytest.mark.asyncio
async def test_legacy_scorer_fps_argument_keeps_existing_semantics():
    manager = _build_manager({"legacy": {"path": DUMMY_REWARDS_PATH, "name": "reward_legacy_fps", "fps": -1}})
    result = await manager.run_single(_make_single_data())
    assert result["reward_score"] == -1.0
    assert "reward/legacy/errors" not in result["reward_extra_info"]


@pytest.mark.asyncio
async def test_legacy_optional_scorer_missing_argument_still_contributes_zero():
    manager = _build_manager({"legacy": {"path": DUMMY_REWARDS_PATH, "name": "reward_missing_legacy_argument"}})
    result = await manager.run_single(_make_single_data())
    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"]["reward/legacy/errors"] == 1


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAP_RECIPES = [
    path for path in sorted((_REPO_ROOT / "examples").rglob("*.sh")) if "reward_score/clap.py" in path.read_text()
]


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe", _CLAP_RECIPES, ids=lambda path: str(path.relative_to(_REPO_ROOT)))
async def test_existing_clap_recipe_config_accepts_stereo_without_layout(recipe, monkeypatch):
    entry = dict(re.findall(r"reward\.reward_functions\.clap\.(\w+)=([^'\"\s]+)", recipe.read_text()))
    assert "audio_layout" not in entry
    entry["path"] = entry["path"].replace("$repo_root", str(_REPO_ROOT))
    entry["device"] = "cpu"
    manager = _build_manager({"clap": entry})
    scorer_globals = manager._sub_rewards[0]["fn"].__globals__
    captured_audio = []

    def processor(*, audio, sampling_rate, **kwargs):
        assert sampling_rate == 48000
        captured_audio.extend(audio)
        return {"features": torch.ones(len(audio), 2)}

    def model(*, features):
        return SimpleNamespace(audio_embeds=features, text_embeds=features)

    monkeypatch.setitem(scorer_globals, "_load_clap", lambda *args: (model, processor))
    source_rate = 24000 if "ltx2" in str(recipe) else 32000
    data = _make_single_data()
    data.batch["audio"] = torch.ones(1, 2, 480)
    data.batch["audio_sample_rate"] = torch.tensor([source_rate])
    try:
        result = await manager.run_single(data)
        assert result["reward_score"] == pytest.approx(1.0)
        assert "reward/clap/errors" not in result["reward_extra_info"]
        assert len(captured_audio) == 1
        assert captured_audio[0].shape == (480 * 48000 // source_rate,)
        assert result["reward_extra_info"]["reward/clap/source_sample_rate"] == source_rate
    finally:
        state = scorer_globals["_get_batching_state"]()
        if state.consumer_task is not None:
            state.queue.put_nowait((None, None, None, None, None))
            await state.consumer_task
