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
"""V1-by-default trainer selection and v0 deprecation warnings."""

import warnings
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl_omni
from verl_omni.trainer.main_diffusion import _deprecate_v0_trainer, uses_v1_trainer


def compose_diffusion_config(overrides=None):
    """Compose the diffusion trainer config the way the Hydra entrypoint does."""
    config_dir = Path(verl_omni.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides or [])


def test_diffusion_config_defaults_to_v1():
    config = compose_diffusion_config()
    assert config.trainer.use_v1 is True
    assert config.trainer.v1.trainer_mode == "sync"


@pytest.mark.parametrize(
    ("sample_source", "trainer_type", "expected"),
    [
        ("online", "policy_gradient", True),
        ("online", "direct_preference", True),
        ("offline", "policy_gradient", True),
        ("offline", "direct_preference", False),
    ],
)
def test_uses_v1_trainer_keeps_offline_dpo_on_v0(sample_source, trainer_type, expected):
    config = OmegaConf.create({"algorithm": {"sample_source": sample_source, "trainer_type": trainer_type}})
    assert uses_v1_trainer(config) is expected


def _v0_launch_config(sample_source="online", trainer_type="policy_gradient", use_v1=True):
    return OmegaConf.create(
        {
            "algorithm": {"sample_source": sample_source, "trainer_type": trainer_type},
            "trainer": {"use_v1": use_v1},
        }
    )


def test_v0_launch_warns_and_normalizes_use_v1():
    config = _v0_launch_config()
    with pytest.warns(DeprecationWarning, match="legacy \\(v0\\) diffusion trainer is deprecated"):
        _deprecate_v0_trainer(config)
    assert config.trainer.use_v1 is False


def test_offline_dpo_v0_launch_stays_silent():
    config = _v0_launch_config(sample_source="offline", trainer_type="direct_preference")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _deprecate_v0_trainer(config)
    assert config.trainer.use_v1 is False


@pytest.mark.parametrize("entrypoint", ["v0", "v1"])
@pytest.mark.parametrize("reward_path", ["legacy", "named"])
@pytest.mark.parametrize("reward_enabled, full_determinism", [(False, True), (True, False), (True, True)])
def test_diffusion_ray_runtime_env_matches_reward_config(
    monkeypatch, entrypoint, reward_path, reward_enabled, full_determinism
):
    import verl_omni.trainer.main_diffusion as main_v0
    import verl_omni.trainer.main_diffusion_v1 as main_v1

    config = compose_diffusion_config()
    assert config.transfer_queue.enable is False
    config.reward.reward_model.enable = reward_enabled if reward_path == "legacy" else False
    config.reward.custom_reward_function.path = "verl_omni/utils/reward_score/genrm_ocr.py"
    OmegaConf.update(config, "reward.reward_model.rollout.full_determinism", full_determinism, force_add=True)
    OmegaConf.update(config, "reward.reward_model.rollout.seed", 17, force_add=True)
    if reward_path == "named" and reward_enabled:
        OmegaConf.update(
            config,
            "reward.models.ocr",
            {"backend": "engine", "model_path": "ocr", "rollout": {"full_determinism": full_determinism, "seed": 17}},
            force_add=True,
        )
    for key in ("VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "PYTHONHASHSEED"):
        monkeypatch.delenv(key, raising=False)

    captured = {}

    class FakeRunner:
        @staticmethod
        def options(**kwargs):
            return FakeRunner

        @staticmethod
        def remote():
            return FakeRunner

        class run:
            @staticmethod
            def remote(config):
                return None

    monkeypatch.setattr(main_v1.ray, "is_initialized", lambda: False)

    def fake_ray_init(**kwargs):
        captured["runtime_env"] = kwargs["runtime_env"]

    monkeypatch.setattr(main_v1.ray, "init", fake_ray_init)
    monkeypatch.setattr(main_v1.ray, "get", lambda handle: None)

    if entrypoint == "v0":
        with pytest.warns(DeprecationWarning, match="legacy \\(v0\\) diffusion trainer"):
            main_v0.run_diffusion(config, task_runner_class=FakeRunner)
    else:
        main_v1.run_diffusion_v1(config, task_runner_class=FakeRunner)

    env_vars = captured["runtime_env"]["env_vars"]
    if entrypoint == "v1":
        assert config.transfer_queue.enable is True
        assert env_vars.get("TRANSFER_QUEUE_ENABLE") == "1"
    expected = "1" if reward_enabled and full_determinism else "0"
    assert env_vars["VERL_FULL_DETERMINISM"] == expected
    assert env_vars["VLLM_BATCH_INVARIANT"] == expected
    if expected == "1":
        assert env_vars["PYTHONHASHSEED"] == "17"
    else:
        assert "PYTHONHASHSEED" not in env_vars
