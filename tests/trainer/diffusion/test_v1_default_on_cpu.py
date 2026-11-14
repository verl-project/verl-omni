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

import os
import warnings
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

import verl_omni
from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls
from verl_omni.trainer.diffusion.v1.trainer_colocate_async import PolicyGradientDiffusionTrainerV1ColocateAsync
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


@pytest.mark.parametrize("overrides", [[], ["+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS='8'"]])
@pytest.mark.parametrize("entrypoint", ["v0", "v1"])
@pytest.mark.parametrize("reward_path", ["legacy", "named"])
@pytest.mark.parametrize("reward_enabled, full_determinism", [(False, True), (True, False), (True, True)])
def test_diffusion_ray_runtime_env_matches_reward_config(
    monkeypatch, entrypoint, reward_path, reward_enabled, full_determinism, overrides
):
    """User runtime overrides coexist with deterministic reward and TransferQueue flags."""
    import verl_omni.trainer.main_diffusion as main_v0
    import verl_omni.trainer.main_diffusion_v1 as main_v1

    config = compose_diffusion_config(overrides)
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
    if overrides:
        assert env_vars.get("OMP_NUM_THREADS") == "8"


@pytest.mark.parametrize("seed", [17, 2**32, 2**63 - 1])
@pytest.mark.parametrize("reward_path", ["legacy", "named"])
def test_reward_hash_seed_starts_python_without_changing_sampling_seed(seed, reward_path):
    """Large sampling seeds produce valid child-process hash seeds without mutation."""
    import subprocess
    import sys

    from verl_omni.workers.config.reward import reward_determinism_env

    config = compose_diffusion_config([])
    config.reward.reward_model.enable = reward_path == "legacy"
    model = config.reward.reward_model if reward_path == "legacy" else OmegaConf.create({"backend": "engine"})
    OmegaConf.update(model, "rollout", {"full_determinism": True, "seed": seed}, force_add=True)
    if reward_path == "named":
        OmegaConf.update(config, "reward.models.ocr", model, force_add=True)
    env = reward_determinism_env(config)
    assert env["PYTHONHASHSEED"] == str(seed % (2**32))
    assert model.rollout.seed == seed
    result = subprocess.run([sys.executable, "-c", "pass"], env={**os.environ, **env}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("seed", [None, "42", True])
def test_deterministic_reward_rejects_noninteger_seed(seed):
    from verl_omni.workers.config.reward import reward_determinism_env

    config = compose_diffusion_config([])
    config.reward.reward_model.enable = True
    OmegaConf.update(config, "reward.reward_model.rollout.full_determinism", True, force_add=True)
    OmegaConf.update(config, "reward.reward_model.rollout.seed", seed, force_add=True)
    with pytest.raises(ValueError, match="rollout.seed must be an integer"):
        reward_determinism_env(config)


def test_registered_trainer_uses_async_groups_and_exact_refill():
    trainer = get_diffusion_trainer_cls("colocate_async")(
        compose_diffusion_config(
            [
                "trainer.v1.trainer_mode=colocate_async",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                "+data.gen_batch_size=4",
            ]
        )
    )
    assert isinstance(trainer, PolicyGradientDiffusionTrainerV1ColocateAsync)
    assert isinstance(trainer.replay_buffer, ReplayBufferAsync)
    assert trainer.parameter_sync_step == 1
    assert trainer._generation_batch_size() == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("actor_rollout_ref.rollout.nnodes=1", "colocated rollout"),
        ("actor_rollout_ref.rollout.checkpoint_engine.backend=nccl", "backend=naive"),
        ("trainer.v1.colocate_async.num_warmup_batches=-1", "nonnegative integer"),
        ("trainer.v1.colocate_async.num_warmup_batches=1.5", "nonnegative integer"),
        ("+trainer.v1.colocate_async.parameter_sync_step=2", "one parameter sync"),
        ("actor_rollout_ref.model.algorithm=diffusion_nft", "FlowGRPO"),
        ("data.train_batch_size=0", "positive integer"),
        ("data.train_batch_size=-2", "positive integer"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=0", "positive integer"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=1.5", "positive integer"),
        ("data.train_batch_size=3", "divisible by ppo_mini_batch_size"),
    ],
)
def test_invalid_config_fails_before_workers_start(override, message):
    with pytest.raises(ValueError, match=message):
        PolicyGradientDiffusionTrainerV1ColocateAsync(
            compose_diffusion_config(
                [
                    "trainer.v1.trainer_mode=colocate_async",
                    "data.train_batch_size=4",
                    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                    override,
                ]
            )
        )
