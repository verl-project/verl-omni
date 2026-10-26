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
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl_omni
from verl_omni.trainer import main_diffusion_v1
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


def test_diffusion_rollout_standalone_budget_overrides_without_struct_error():
    # The standalone replicas' memory budget must be a listed key of the diffusion
    # tree (it lives in DiffusionRolloutConfig), not verl's rollout yaml: a plain
    # override reaches the separate-async trainer's budget lookup instead of
    # failing Hydra struct mode.
    config = compose_diffusion_config(["actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.85"])
    assert config.actor_rollout_ref.rollout.standalone_gpu_memory_utilization == 0.85

    default = compose_diffusion_config()
    assert default.actor_rollout_ref.rollout.standalone_gpu_memory_utilization == 0.5


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


@pytest.mark.parametrize(
    "overrides",
    [[], ["+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS='8'"]],
    ids=["default", "user_env_vars"],
)
def test_run_diffusion_v1_force_enables_transfer_queue_before_ray_init(monkeypatch, overrides):
    import verl_omni.trainer.main_diffusion_v1 as main_v1

    config = compose_diffusion_config(overrides)
    assert config.transfer_queue.enable is False

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

    main_v1.run_diffusion_v1(config, task_runner_class=FakeRunner)

    assert config.transfer_queue.enable is True
    env_vars = captured["runtime_env"]["env_vars"]
    assert env_vars.get("TRANSFER_QUEUE_ENABLE") == "1"
    if overrides:
        assert env_vars.get("OMP_NUM_THREADS") == "8"




def test_default_task_runner_combines_resource_and_nsys_options(monkeypatch):
    config = OmegaConf.create(
        {
            "global_profiler": {"tool": "nsys", "steps": [1]},
            "transfer_queue": {"enable": False},
            "ray_kwargs": {},
        }
    )
    options = {"trace": "nvtx", "o": "controller-test"}
    build_options = Mock(return_value=options)
    runner_class = Mock()
    monkeypatch.setattr(main_diffusion_v1, "build_controller_nsight_launch_options", build_options)
    monkeypatch.setattr(main_diffusion_v1, "enable_rl_insight", Mock())
    monkeypatch.setattr(main_diffusion_v1, "DiffusionTaskRunnerV1", runner_class)
    monkeypatch.setattr(main_diffusion_v1, "is_cuda_available", True)
    monkeypatch.setattr(main_diffusion_v1.ray, "is_initialized", lambda: True)
    get = Mock()
    monkeypatch.setattr(main_diffusion_v1.ray, "get", get)
    monkeypatch.setattr("verl.utils.import_utils.is_nvtx_available", lambda: True)

    main_diffusion_v1.run_diffusion_v1(config)

    build_options.assert_called_once_with(config)
    runner_class.options.assert_called_once_with(num_cpus=1, runtime_env={"nsight": options})
    runner = runner_class.options.return_value.remote.return_value
    runner.run.remote.assert_called_once_with(config)
    get.assert_called_once_with(runner.run.remote.return_value)
