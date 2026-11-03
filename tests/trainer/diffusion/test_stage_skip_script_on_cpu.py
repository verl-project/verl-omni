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


@pytest.fixture
def launcher(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("The stage-skip launcher requires Bash")
    script = tmp_path / "examples/stage_skip/run_sd35_medium_ocr_replay.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).parents[3] / "examples/stage_skip" / script.name
    shutil.copyfile(source, script)
    recipe = tmp_path / "examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        'printf "launch\\n" >> "$CALLS_PATH"\n'
        'printf "%s\\n" "$@" > "$ARGV_PATH"\n'
        'printf "{}\\n" > "$VERL_FILE_LOGGER_PATH"\n'
        'exit "${RECIPE_EXIT_CODE:-0}"\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "STAGE_SKIP_RUN_DIR": (tmp_path / "run with spaces").as_posix(),
        "NUM_STEPS": "3",
        "STAGES": "reward old_log_prob",
        "CALLS_PATH": (tmp_path / "calls.txt").as_posix(),
        "ARGV_PATH": (tmp_path / "argv.txt").as_posix(),
        "RECIPE_EXIT_CODE": "0",
    }
    return bash, script, env


@pytest.mark.parametrize(
    ("stages", "steps", "overrides"),
    [
        ("reward old_log_prob", 3, []),
        (
            "reward",
            5,
            ["trainer.nnodes=1", "actor_rollout_ref.model.path=/models/with spaces", "trainer.test_freq=5"],
        ),
        ("reward old_log_prob", 1, []),
    ],
)
def test_launches_once_with_repeat_and_preserves_overrides(launcher, stages, steps, overrides):
    bash, script, env = launcher
    env["STAGES"] = stages
    env["NUM_STEPS"] = str(steps)
    result = subprocess.run(
        [bash, script.as_posix(), *overrides], env=env, capture_output=True, encoding="utf-8", timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert Path(env["CALLS_PATH"]).read_text().splitlines() == ["launch"]
    args = Path(env["ARGV_PATH"]).read_text().splitlines()
    if overrides:
        assert args[-len(overrides) :] == overrides
    assert f"trainer.total_training_steps={steps}" in args
    step_list = "[" + ",".join(str(step) for step in range(1, steps + 1)) + "]"
    for role in ["skip.rollout_tq", *(f"omni_skip.{stage}" for stage in stages.split())]:
        assert f"{role}.enable=True" in args
        assert f"{role}.steps={step_list}" in args
        assert f"{role}.action=repeat" in args
        assert f"{role}.dump_dir={env['STAGE_SKIP_RUN_DIR']}/dump" in args
    if stages == "reward":
        assert not any(arg.startswith("omni_skip.old_log_prob.") for arg in args)
    assert (Path(env["STAGE_SKIP_RUN_DIR"]) / "metrics/repeat.jsonl").is_file()


def test_recipe_failure_is_returned_without_retry(launcher):
    bash, script, env = launcher
    env["RECIPE_EXIT_CODE"] = "17"
    result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True, encoding="utf-8", timeout=30)
    assert result.returncode == 17
    assert Path(env["CALLS_PATH"]).read_text().splitlines() == ["launch"]


def test_existing_dump_is_rejected_before_launch(launcher):
    bash, script, env = launcher
    dump = Path(env["STAGE_SKIP_RUN_DIR"]) / "dump"
    dump.mkdir(parents=True)
    payload = dump / "existing.pt"
    payload.write_text("existing cache", encoding="utf-8")
    result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True, encoding="utf-8", timeout=30)
    assert result.returncode != 0
    assert "already holds a dump" in result.stderr
    assert not Path(env["CALLS_PATH"]).exists()
    assert payload.read_text() == "existing cache"


@pytest.mark.parametrize("steps", ["0", "-1", "invalid"])
def test_invalid_step_count_is_rejected_before_launch(launcher, steps):
    bash, script, env = launcher
    env["NUM_STEPS"] = steps
    result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True, encoding="utf-8", timeout=30)
    assert result.returncode != 0
    assert "NUM_STEPS must be a positive integer" in result.stderr
    assert not Path(env["CALLS_PATH"]).exists()
