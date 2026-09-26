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

"""CPU checks for the three MiniMax H3 VeOmni launchers."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RECIPE = _ROOT / "examples/flowgrpo_trainer/minimax_h3"


@pytest.fixture
def recipe_env(tmp_path):
    model = tmp_path / "model with spaces"
    for partition in ("FL2VA", "Ref2VA"):
        (model / partition / "transformer").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARGV_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['TRAINER_EXIT_CODE']))\n"
    )
    stub.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MODEL_ROOT": str(model),
        "DATA_DIR": str(tmp_path / "data with spaces"),
        "OUTPUT_DIR": str(tmp_path / "output with spaces"),
        "NUM_GPUS": "8",
        "ROLLOUT_TP": "2",
        "TEXT_ENCODER_TP": "2",
        "ARGV_FILE": str(tmp_path / "argv.json"),
        "TRAINER_EXIT_CODE": "0",
    }


def _run(task, env, *overrides):
    env = dict(env)
    env["MODEL_PATH"] = env["MODEL_ROOT"] if task == "ref2va" else f"{env['MODEL_ROOT']}/FL2VA"
    return subprocess.run(
        ["bash", str(_RECIPE / f"run_minimax_h3_{task}_lora_veomni.sh"), *overrides],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("exit_code", [0, 23])
def test_veomni_launchers_preserve_paths_task_overrides_and_exit_status(recipe_env, task, exit_code):
    recipe_env["TRAINER_EXIT_CODE"] = str(exit_code)
    override = "trainer.total_training_steps=3"
    result = _run(task, recipe_env, override)
    assert result.returncode == exit_code, result.stderr
    argv = json.loads(Path(recipe_env["ARGV_FILE"]).read_text())
    assert argv[:2] == ["-m", "verl_omni.trainer.main_diffusion"]
    assert argv[-1] == override
    recipe_keys = [arg.lstrip("+").split("=", 1)[0] for arg in argv[2:-1]]
    assert len(recipe_keys) == len(set(recipe_keys))
    logs = list(Path(recipe_env["OUTPUT_DIR"]).glob("logs/*/*.log"))
    assert len(logs) == 1
    assert "verl_omni.trainer.main_diffusion" in logs[0].read_text()
    options = dict(arg.lstrip("+").split("=", 1) for arg in argv[2:])
    partition = "Ref2VA" if task == "ref2va" else "FL2VA"
    model = f"{recipe_env['MODEL_ROOT']}/{partition}"
    assert options["diffusion/model_engine"] == "veomni_diffusion"
    assert options["actor_rollout_ref.model.path"] == model
    assert options["actor_rollout_ref.model.config_path"] == f"{model}/transformer"
    assert options["actor_rollout_ref.model.transformer_subfolder"] == "transformer"
    assert options["actor_rollout_ref.model.target_modules"] == "['qkv_proj','out_proj','fc1','fc2']"
    assert options["actor_rollout_ref.actor.veomni_config.attn_implementation"] == "flash_attention_3_hub"
    assert options["actor_rollout_ref.ref.veomni_config.attn_implementation"] == "flash_attention_3_hub"
    assert options["data.train_files"] == f"{recipe_env['DATA_DIR']}/train.parquet"
    assert options["data.val_files"] == f"{recipe_env['DATA_DIR']}/test.parquet"
    assert options["trainer.default_local_dir"] == f"{recipe_env['OUTPUT_DIR']}/checkpoints"
    assert options["trainer.experiment_name"] == f"minimax_h3_{task}_lora_veomni"
    assert options["actor_rollout_ref.rollout.agent.num_workers"] == "4"
    assert not any(".fsdp_config." in arg for arg in argv)
    if exit_code == 0:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(_ROOT / "verl_omni/trainer/config"), version_base=None):
            config = compose(config_name="diffusion_trainer", overrides=argv[2:])
        assert config.actor_rollout_ref.actor.strategy == "veomni"
        assert config.trainer.total_training_steps == 3
    if task != "t2va":
        for phase in ("pipeline", "val_kwargs.pipeline"):
            assert options[f"actor_rollout_ref.rollout.{phase}.task"] == task
            if task == "fl2va":
                assert options[f"actor_rollout_ref.rollout.{phase}.frame_indices"] == "[0]"
            else:
                assert options[f"actor_rollout_ref.rollout.{phase}.reference_image_short_edge"] == "512"
                assert options[f"actor_rollout_ref.rollout.{phase}.max_sequence_length"] == "12288"
        if task == "ref2va":
            assert options["actor_rollout_ref.rollout.max_prompt_embed_length"] == "12288"
            assert options["actor_rollout_ref.rollout.pipeline.video_flow_shift"] == "12.0"
            assert options["actor_rollout_ref.rollout.pipeline.num_frames"] == "96"


@pytest.mark.parametrize("val_edge", [None, "1024"])
def test_ref2va_launcher_preserves_reference_size_overrides(recipe_env, val_edge):
    recipe_env["REF_IMAGE_SHORT_EDGE"] = "768"
    recipe_env.pop("VAL_REF_IMAGE_SHORT_EDGE", None)
    if val_edge is not None:
        recipe_env["VAL_REF_IMAGE_SHORT_EDGE"] = val_edge
    result = _run("ref2va", recipe_env)
    assert result.returncode == 0, result.stderr
    argv = json.loads(Path(recipe_env["ARGV_FILE"]).read_text())
    options = dict(arg.lstrip("+").split("=", 1) for arg in argv[2:])
    assert options["actor_rollout_ref.rollout.pipeline.reference_image_short_edge"] == "768"
    assert options["actor_rollout_ref.rollout.val_kwargs.pipeline.reference_image_short_edge"] == (val_edge or "768")


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("name,value", [("NUM_GPUS", "3"), ("ROLLOUT_TP", "0"), ("TEXT_ENCODER_TP", "two")])
def test_veomni_launchers_reject_invalid_parallelism(recipe_env, task, name, value):
    recipe_env[name] = value
    assert _run(task, recipe_env).returncode != 0
    assert not Path(recipe_env["ARGV_FILE"]).exists()


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_veomni_launchers_require_fused_checkpoint_and_writable_output(recipe_env, task):
    Path(recipe_env["OUTPUT_DIR"]).write_text("not a directory")
    assert _run(task, recipe_env).returncode != 0
    assert not Path(recipe_env["ARGV_FILE"]).exists()
    Path(recipe_env["OUTPUT_DIR"]).unlink()
    partition = "Ref2VA" if task == "ref2va" else "FL2VA"
    (Path(recipe_env["MODEL_ROOT"]) / partition / "transformer").rmdir()
    assert _run(task, recipe_env).returncode != 0
    assert not Path(recipe_env["ARGV_FILE"]).exists()
