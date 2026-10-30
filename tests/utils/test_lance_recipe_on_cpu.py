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
"""Exercise the Lance launcher's shell/Hydra/Jinja prompt boundary without a GPU."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from jinja2 import Template


@pytest.fixture(scope="module")
def recipe_overrides(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("lance_recipe")
    capture = directory / "argv"
    python = directory / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$LANCE_CAPTURE_ARGS"\n')
    python.chmod(0o755)
    env = os.environ | {
        "PATH": str(directory) + os.pathsep + os.environ["PATH"],
        "WORKSPACE": str(directory),
        "LANCE_CAPTURE_ARGS": str(capture),
    }
    subprocess.run(
        ["bash", "examples/flowgrpo_trainer/lance/run_lance_pickscore_lora.sh"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = capture.read_text().rstrip("\0").split("\0")
    assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion"]
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(args[2:])
    return {override.key_or_group: override.value() for override in overrides}


@pytest.mark.parametrize("caption", ["A red car.", " ", "一只猫\non a chair", r'A sign reading "\n"'])
def test_recipe_renders_real_chat_boundaries_without_changing_caption(recipe_overrides, caption):
    template = recipe_overrides["actor_rollout_ref.model.custom_chat_template"]
    rendered = Template(template).render(messages=[{"role": "user", "content": caption}])
    assert rendered == f"<|im_start|>user\n{caption}<|im_end|>\n"


def test_recipe_filters_overlong_prompts_using_the_rollout_template(recipe_overrides):
    assert recipe_overrides["data.filter_overlong_prompts"] is True
    assert recipe_overrides["data.max_prompt_length"] == 256
    assert (
        recipe_overrides["data.apply_chat_template_kwargs.chat_template"]
        == recipe_overrides["actor_rollout_ref.model.custom_chat_template"]
    )


@pytest.mark.parametrize(
    "mode,reward", [("i2v", "imagebind"), ("image_edit", "pickscore"), ("video_edit", "imagebind")]
)
def test_conditioned_recipes_preserve_paths_and_caller_overrides(tmp_path, mode, reward):
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path / "workspace with spaces"
    directory.mkdir()
    capture = directory / "argv"
    python = directory / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$LANCE_CAPTURE_ARGS"\n')
    python.chmod(0o755)
    env = os.environ | {
        "PATH": str(directory) + os.pathsep + os.environ["PATH"],
        "WORKSPACE": str(directory),
        "LANCE_CAPTURE_ARGS": str(capture),
    }
    result = subprocess.run(
        [
            "bash",
            f"examples/flowgrpo_trainer/lance/run_lance_{mode}_{reward}_lora.sh",
            "trainer.total_training_steps=1",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = capture.read_text().rstrip("\0").split("\0")
    parsed = OverridesParser.create().parse_overrides(args[2:])
    overrides = {override.key_or_group: override.value() for override in parsed}
    assert overrides["data.train_files"] == f"{directory}/data/lance_{mode}/lance/train.parquet"
    assert overrides["actor_rollout_ref.model.tokenizer_path"] == f"{directory}/data/lance_{mode}/lance/tokenizer"
    assert overrides["actor_rollout_ref.rollout.agent.default_agent_loop"] == "lance_diffusion_single_turn_agent"
    assert overrides["trainer.total_training_steps"] == 1
    if mode == "video_edit":
        assert overrides["actor_rollout_ref.rollout.pipeline.frame_rate"] == 12
        assert overrides["actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate"] == 12
