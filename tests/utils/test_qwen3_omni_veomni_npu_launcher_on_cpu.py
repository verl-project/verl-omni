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

"""Capture real NPU launchers and compare dataset settings without training dependencies."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ROOT / "examples/gspo_trainer/qwen3_omni"


def _launch(tmp_path, dataset, backend, extra=(), exit_code=0, ep="1"):
    cann = tmp_path / "cann"
    atb = tmp_path / "nnal/atb"
    cann.mkdir(exist_ok=True)
    atb.mkdir(parents=True, exist_ok=True)
    (cann / "set_env.sh").write_text("# test CANN environment\n")
    (atb / "set_env.sh").write_text("# test ATB environment\n")
    prefix = "veomni_" if backend == "veomni" else ""
    script = LAUNCHERS / f"run_qwen3_omni_thinker_gspo_{prefix}npu_{dataset}_v1.sh"
    log = tmp_path / f"{dataset}-{backend}.log"
    env = {
        **os.environ,
        "ASCEND_HOME_PATH": cann.as_posix(),
        "LOG_FILE": log.as_posix(),
        "TRAINING_BACKEND": "fsdp2",
        "MODEL_PATH": "/models/path with spaces",
        "TRAIN_FILE": "/data/train with spaces.parquet",
        "VAL_FILE": "/data/validation.parquet",
        "OUTPUT_DIR": "/outputs/comparison",
        "EXPERIMENT_NAME": "comparison",
        "NUM_GPUS_ACTOR_ROLLOUT_REWARD": "16",
        "N_GPUS_PER_NODE": "16",
        "NNODES": "1",
        "ROLLOUT_TP": "2",
        "ACTOR_EP": ep,
        "TRAIN_BATCH_SIZE": "32",
        "ROLLOUT_N": "8",
        "TOTAL_TRAINING_STEPS": "500",
        "MULTI_STREAM_MEMORY_REUSE": "2",
        "STUB_EXIT": str(exit_code),
    }
    command = 'python3() { printf "%s\\0" "$@"; return "$STUB_EXIT"; }; export -f python3; bash "$@"'
    result = subprocess.run(
        [os.environ.get("BASH_EXECUTABLE", "bash"), "-c", command, "capture", script.as_posix(), *extra],
        cwd=ROOT,
        env=env,
        capture_output=True,
    )
    if result.returncode == 0 or exit_code:
        assert result.stdout == log.read_bytes()
    return result


def _values(result):
    assert result.returncode == 0, result.stderr.decode()
    argv = result.stdout.decode().rstrip("\0").split("\0")
    assert argv[:2] == ["-m", "verl_omni.trainer.main_omni"]
    return {item.key_or_group: item.value() for item in OverridesParser.create().parse_overrides(argv[2:])}


@pytest.mark.parametrize("dataset", ["nextqa"])
def test_veomni_preserves_dataset_training_and_rollout_settings(tmp_path, dataset):
    veomni = _values(_launch(tmp_path, dataset, "veomni"))
    assert veomni["data.custom_cls.name"] == "NextQARLHFDataset"
    assert veomni["data.train_batch_size"] == 32
    assert veomni["actor_rollout_ref.rollout.n"] == 8
    assert veomni["actor_rollout_ref.actor.optim.lr"] == 1e-6
    assert veomni["actor_rollout_ref.model.path"] == "/models/path with spaces"
    assert not any("fsdp_config" in key for key in veomni)
    assert veomni["model_engine"] == "veomni"
    assert veomni["actor_rollout_ref.actor.veomni.expert_parallel_size"] == 1
    for values in (veomni,):
        assert (
            values["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_mode"] == "PIECEWISE"
        )
        assert values["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode"] == 0
    assert not any(key.endswith(".veomni.model_dtype") for key in veomni)
    assert veomni["actor_rollout_ref.model.override_config.freeze_vision_tower"] is (dataset == "avqa")
    assert veomni["actor_rollout_ref.model.override_config.freeze_audio_tower"] is (dataset == "avqa")
    env = {key: value for key, value in veomni.items() if ".env_vars." in key}
    assert all(isinstance(value, str) for value in env.values())


@pytest.mark.parametrize("dataset", ["nextqa"])
def test_veomni_launcher_cli_overrides_and_exit_propagation(tmp_path, dataset):
    values = _values(_launch(tmp_path, dataset, "veomni", ("data.max_prompt_length=2048",)))
    assert values["data.max_prompt_length"] == 2048
    result = _launch(tmp_path, dataset, "veomni", exit_code=37)
    assert result.returncode == 37


@pytest.mark.parametrize("dataset", ["nextqa"])
def test_veomni_launcher_rejects_invalid_ep_before_training(tmp_path, dataset):
    result = _launch(tmp_path, dataset, "veomni", ep="3")
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize("dataset", ["nextqa"])
def test_veomni_launcher_composes_with_production_yaml(tmp_path, dataset):
    result = _launch(tmp_path, dataset, "veomni")
    assert result.returncode == 0, result.stderr.decode()
    argv = result.stdout.decode().rstrip("\0").split("\0")
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="omni_trainer", overrides=argv[2:])
    assert config.actor_rollout_ref.actor.strategy == "veomni"
    assert "fsdp_config" not in config.actor_rollout_ref.actor
    assert config.actor_rollout_ref.ref.veomni.moe_implementation == "fused_npu"
    env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True)
    assert all(isinstance(value, str) for value in env.values())
    assert config.actor_rollout_ref.actor.optim.lr == 1e-6
    assert config.actor_rollout_ref.actor.optim.weight_decay == 0.01



@pytest.mark.parametrize("dataset", ["nextqa"])
def test_veomni_launcher_instantiates_engine_configs(tmp_path, dataset):
    from hydra.utils import instantiate

    result = _launch(tmp_path, dataset, "veomni")
    assert result.returncode == 0, result.stderr.decode()
    argv = result.stdout.decode().rstrip("\0").split("\0")
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="omni_trainer", overrides=argv[2:])
    for role in ("actor", "ref"):
        engine = instantiate(config.actor_rollout_ref[role].veomni)
        assert engine.strategy == "veomni"
        assert engine.mixed_precision is True
