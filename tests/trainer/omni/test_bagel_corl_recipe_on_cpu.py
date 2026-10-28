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

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RECIPE = Path("examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh")
BASH = shutil.which("bash")
if os.name == "nt":
    BASH = shutil.which("bash", path="D:/Program Files/Git/bin")
pytestmark = pytest.mark.skipif(not BASH, reason="Bash is required for recipe tests")


def _shell_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


@pytest.fixture
def launch(tmp_path):
    script = tmp_path / RECIPE
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / RECIPE, script)
    python = tmp_path / ".venv/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-m" && "$2" == "verl_omni.trainer.main_omni" ]]; then\n'
        '  printf "%s\\0" "$@" > "$CAPTURE_ARGS"\n'
        '  exit "${TRAINER_EXIT:-0}"\n'
        "fi\n"
        'if [[ "$1" == "-m" ]]; then exit "${DATASET_EXIT:-0}"; fi\n'
    )
    python.chmod(0o755)
    captured = tmp_path / "args.bin"
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.parquet", "val.parquet"):
        (data / name).touch()
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        REBUILD_UNICOT="0",
        RUN_TS="fixed",
        JUDGE_SERVER="0",
        ENABLE_RM="1",
        CAPTURE_ARGS=_shell_path(captured),
        UNICOT_TRAIN=_shell_path(data / "train.parquet"),
        UNICOT_TEST=_shell_path(data / "val.parquet"),
        CUDA_HOME="/test/cuda",
        _LOCAL_VERL="/absent-verl",
    )

    def run(overrides=(), **updates):
        captured.unlink(missing_ok=True)
        result = subprocess.run(
            [BASH, _shell_path(script), *overrides],
            env=env | updates,
            capture_output=True,
            text=True,
            timeout=30,
        )
        args = captured.read_bytes().decode().rstrip("\0").split("\0") if captured.exists() else []
        return result, args

    return run


def test_defaults_and_caller_precedence(launch):
    override = "actor_rollout_ref.actor.optim.lr=7e-6"
    result, args = launch([override])
    assert result.returncode == 0, result.stderr
    assert args[-1] == override
    for expected in (
        "actor_rollout_ref.rollout.n=8",
        "data.train_batch_size=4",
        "actor_rollout_ref.actor.optim.lr=3e-5",
        "+actor_rollout_ref.model.lr_gen=3e-5",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "+actor_rollout_ref.actor.diffusion_loss.clip_ratio=0.02",
        "algorithm.rollout_correction.rollout_rs=seq_mean_k1",
        "algorithm.rollout_correction.rollout_rs_threshold=20",
        "agentic_image_gen.corl_reward_version=v2",
        "trainer.save_freq=25",
        "trainer.resume_mode=auto",
    ):
        assert expected in args


@pytest.mark.parametrize("devices, batch, steps", [("0", "1", "4"), ("0,1", "2", "15")])
def test_smaller_gpu_budgets_keep_reward_off(launch, devices, batch, steps):
    updates = {"CUDA_VISIBLE_DEVICES": devices, "ENABLE_RM": "0", "UND_N_GPUS": str(len(devices.split(",")))}
    result, args = launch(**updates)
    assert result.returncode == 0, result.stderr
    assert f"data.train_batch_size={batch}" in args
    assert f"+actor_rollout_ref.rollout.pipeline.num_inference_steps={steps}" in args
    assert "actor_rollout_ref.rollout.n=2" in args
    assert "agentic_image_gen.corl_reward_version=v1" in args
    assert "reward.num_workers=0" in args


def test_path_arguments_and_explicit_overrides(launch):
    result, args = launch(
        ["trainer.experiment_name=caller"],
        BAGEL_MODEL_PATH="/models/bagel checkpoint",
        REWARD_MODEL="/models/reward checkpoint",
        RUN_DIR="/runs/with spaces",
        DEFAULT_LOCAL_DIR="/checkpoints/with spaces",
        ACTOR_LR="2e-5",
        ENABLE_PEC="0",
    )
    assert result.returncode == 0, result.stderr
    assert "actor_rollout_ref.model.path=/models/bagel checkpoint" in args
    assert "reward.reward_model.model_path=/models/reward checkpoint" in args
    assert "hydra.run.dir=/runs/with spaces" in args
    assert "trainer.default_local_dir=/checkpoints/with spaces" in args
    assert "actor_rollout_ref.actor.optim.lr=2e-5" in args
    assert "+actor_rollout_ref.rollout.enable_prompt_embed_cache=False" in args
    assert "+actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity=False" in args
    assert args[-1] == "trainer.experiment_name=caller"


@pytest.mark.parametrize(
    "updates, expected_status",
    [
        ({"TRAINER_EXIT": "23"}, 23),
        ({"REBUILD_UNICOT": "1", "DATASET_EXIT": "19"}, 1),
    ],
)
def test_launch_failure_propagation(launch, updates, expected_status):
    result, args = launch(**updates)
    assert result.returncode == expected_status, result.stderr
    assert bool(args) == ("TRAINER_EXIT" in updates)


def test_rejects_zero_advantage_gen_groups(launch):
    result, args = launch(N="1", S="1")
    assert result.returncode == 1
    assert "every GEN advantage is zero" in result.stderr
    assert not args
