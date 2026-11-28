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
"""CPU tests for pre-ray.init() determinism env propagation in V0 and V1 diffusion entrypoints."""

import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer import main_diffusion, main_diffusion_v1, main_omni


def _make_diffusion_config(
    *,
    actor_det: bool = False,
    actor_seed: int | None = 42,
    rm_enable: bool = False,
    rm_det: bool = False,
    rm_seed: int | None = 42,
    named_models: dict | None = None,
):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "full_determinism": actor_det,
                    "seed": actor_seed,
                }
            },
            "reward": {
                "reward_model": {
                    "enable": rm_enable,
                    "rollout": {
                        "full_determinism": rm_det,
                        "seed": rm_seed,
                    },
                },
                "models": named_models or {},
            },
            "transfer_queue": {"enable": False},
            "global_profiler": {"steps": None, "tool": None},
            "ray_kwargs": {"ray_init": {}, "timeline_json_file": None},
        }
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_requested", "expected_seed"),
    [
        ({}, False, None),
        ({"rm_enable": False, "rm_det": True}, False, None),
        ({"actor_det": True, "actor_seed": 123}, True, "123"),
        ({"actor_det": True, "actor_seed": 0}, True, "0"),
        ({"actor_det": True, "actor_seed": None}, True, "42"),
        ({"rm_enable": True, "rm_det": True, "rm_seed": 77}, True, "77"),
        (
            {"named_models": {"ocr": {"backend": "engine", "rollout": {"full_determinism": True, "seed": 99}}}},
            True,
            "99",
        ),
        ({"actor_det": True, "actor_seed": 11, "rm_enable": True, "rm_det": True, "rm_seed": 22}, True, "11"),
    ],
)
def test_determinism_env_selection(monkeypatch, kwargs, expected_requested, expected_seed):
    monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    config = _make_diffusion_config(**kwargs)

    assert main_diffusion._determinism_requested(config) is expected_requested
    if expected_requested:
        main_diffusion._export_full_determinism_env(config)
        assert os.environ["VERL_FULL_DETERMINISM"] == "1"
        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
        assert os.environ["PYTHONHASHSEED"] == expected_seed


def test_run_diffusion_v1_exports_determinism_env_before_ray_init(monkeypatch):
    monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    env_at_ray_init: dict[str, str | None] = {}

    def fake_get_runtime_env():
        env_at_ray_init["VERL_FULL_DETERMINISM"] = os.environ.get("VERL_FULL_DETERMINISM")
        env_at_ray_init["VLLM_BATCH_INVARIANT"] = os.environ.get("VLLM_BATCH_INVARIANT")
        env_at_ray_init["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
        return {"env_vars": {}}

    class _FakeRunner:
        @staticmethod
        def remote():
            return SimpleNamespace(run=SimpleNamespace(remote=lambda _cfg: "done"))

    monkeypatch.setattr(main_diffusion_v1, "normalize_reward_config", lambda cfg: cfg)
    monkeypatch.setattr(main_diffusion_v1, "warn_deprecated_reward_config", lambda _cfg: None)
    monkeypatch.setattr(main_diffusion_v1, "enable_rl_insight", lambda _cfg: None)
    monkeypatch.setattr(main_diffusion_v1, "get_ppo_ray_runtime_env", fake_get_runtime_env)
    monkeypatch.setattr(main_diffusion_v1.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_diffusion_v1.ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(main_diffusion_v1.ray, "get", lambda _ref: None)

    config = _make_diffusion_config(actor_det=True, actor_seed=314)
    main_diffusion_v1.run_diffusion_v1(config, task_runner_class=_FakeRunner)

    assert env_at_ray_init == {
        "VERL_FULL_DETERMINISM": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "PYTHONHASHSEED": "314",
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_seed"),
    [
        ({}, None),
        ({"actor_det": True, "actor_seed": None}, "42"),
        ({"actor_det": True, "actor_seed": 0}, "0"),
        ({"rm_enable": True, "rm_det": True, "rm_seed": 77}, "77"),
        (
            {"named_models": {"ocr": {"backend": "engine", "rollout": {"full_determinism": True, "seed": 88}}}},
            "88",
        ),
    ],
)
def test_main_omni_maybe_set_determinism_env(monkeypatch, kwargs, expected_seed):
    monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    config = _make_diffusion_config(**kwargs)

    main_omni.maybe_set_determinism_env(config)

    if expected_seed is None:
        assert "VERL_FULL_DETERMINISM" not in os.environ
        assert "VLLM_BATCH_INVARIANT" not in os.environ
        assert "PYTHONHASHSEED" not in os.environ
    else:
        assert os.environ["VERL_FULL_DETERMINISM"] == "1"
        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
        assert os.environ["PYTHONHASHSEED"] == expected_seed


def test_run_omni_offline_forwards_propagate_determinism(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(main_omni, "normalize_reward_config", lambda cfg: cfg)
    monkeypatch.setattr(main_omni, "warn_deprecated_reward_config", lambda _cfg: None)
    monkeypatch.setattr(main_omni, "enable_rl_insight", lambda _cfg: None)
    monkeypatch.setattr(main_omni, "uses_v1_trainer", lambda _cfg: False)
    monkeypatch.setattr(
        main_omni,
        "launch_ray_task_runner",
        lambda config, task_runner_class, **kwargs: captured.update(
            {"config": config, "runner": task_runner_class, **kwargs}
        ),
    )

    config = _make_diffusion_config(actor_det=True, actor_seed=55)
    main_omni.run_omni(config, task_runner_class="fake-runner")

    assert captured["runner"] == "fake-runner"
    assert captured["propagate_determinism"] is True
