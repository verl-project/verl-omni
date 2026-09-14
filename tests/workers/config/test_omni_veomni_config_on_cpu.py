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
"""Compose the public launcher and convert its actor schema without GPU imports."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import VeOmniActorConfig

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh"


@pytest.fixture
def actor_module():
    """Import real config classes while leaving optional rollout packages unloaded."""
    modules = {}
    for name in ("verl_omni", "verl_omni.workers", "verl_omni.workers.config", "verl_omni.workers.config.omni"):
        module = ModuleType(name)
        module.__path__ = [str(ROOT / name.replace(".", "/"))]
        modules[name] = module
    name = "verl_omni.workers.config.omni.actor"
    spec = importlib.util.spec_from_file_location(name, ROOT / "verl_omni/workers/config/omni/actor.py")
    module = importlib.util.module_from_spec(spec)
    modules[name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        package = modules["verl_omni.workers.config.omni"]
        for cls in (module.OmniVeOmniActorConfig, module.OmniActorConfig, module.OmniLossConfig):
            setattr(package, cls.__name__, cls)
        yield module


def _compose_launcher(tmp_path, extra=()):
    """Capture the actual shell argv, then compose it with the production YAML."""
    capture = tmp_path / "python3"
    capture.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    capture.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), *extra],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )
    argv = json.loads(result.stdout)
    assert argv[:2] == ["-m", "verl_omni.trainer.main_omni"]
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        return compose(config_name="omni_trainer", overrides=argv[2:])


def test_launcher_composes_actor_and_reference_as_veomni(tmp_path, actor_module):
    config = _compose_launcher(tmp_path)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor, actor_module.OmniVeOmniActorConfig)
    assert isinstance(actor, VeOmniActorConfig)
    assert actor.strategy == "veomni"
    assert actor.trainer_type == "policy_gradient"
    assert actor.engine.strategy == "veomni"
    assert actor.engine.expert_parallel_size == 8
    assert "fsdp_config" not in config.actor_rollout_ref.actor
    assert config.actor_rollout_ref.ref.strategy == "veomni"
    assert config.actor_rollout_ref.ref.veomni.expert_parallel_size == 8
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.actor_rollout_ref.model.use_fused_kernels
    assert config.actor_rollout_ref.rollout.layered_summon is False
    assert actor.optim.lr_scheduler_type == "constant"
    assert actor.optim.lr_warmup_steps_ratio == pytest.approx(0.05)
    assert config.actor_rollout_ref.rollout.val_kwargs.temperature == 0.0
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES == "verl_omni"


def test_launcher_user_overrides_take_precedence(tmp_path, actor_module):
    config = _compose_launcher(tmp_path, ("actor_rollout_ref.actor.optim.lr=1e-6", "trainer.total_training_steps=2"))
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.optim.lr == pytest.approx(1e-6)
    assert config.trainer.total_training_steps == 2


def test_invalid_omni_trainer_type_is_rejected(actor_module):
    with pytest.raises(ValueError, match="trainer_type"):
        actor_module.OmniVeOmniActorConfig(
            strategy="veomni", rollout_n=2, ppo_micro_batch_size_per_gpu=1, trainer_type="invalid"
        )
