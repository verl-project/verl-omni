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
"""Artifact binding must not inherit a previous run's configuration."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.agentic import artifacts, config


@pytest.mark.parametrize(
    "incoming, expected",
    [
        ({}, "outputs/e2e/new"),
        ({"agentic_image_gen": {}}, "outputs/e2e/new"),
        ({"agentic_image_gen": {"run_dir": None}}, "outputs/e2e/new"),
        ({"agentic_image_gen": {"e2e_root": "new-root"}}, "new-root/new"),
        ({"agentic_image_gen": {"run_dir": "new-run"}}, "new-run"),
    ],
)
def test_rebinding_artifacts_uses_only_the_incoming_config(tmp_path, monkeypatch, incoming, expected):
    tmp_path = tmp_path.resolve()
    monkeypatch.setenv("VERLOMNI_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    previous = OmegaConf.create(
        {
            "trainer": {"experiment_name": "old"},
            "agentic_image_gen": {"run_dir": str(tmp_path / "old-run"), "e2e_root": str(tmp_path / "old-root")},
        }
    )
    artifacts.clear_run_artifacts()
    config.clear_agentic_image_gen()
    try:
        config.bind_agentic_image_gen(previous)
        artifacts.bind_run_artifacts(previous)
        assert artifacts.resolve_run_dir() == tmp_path / "old-run"

        current = OmegaConf.create({"trainer": {"experiment_name": "new"}, **incoming})
        artifacts.bind_run_artifacts(current)

        assert artifacts.resolve_run_dir() == tmp_path / expected
        assert artifacts.resolve_rollout_images_root() == tmp_path / expected / "rollout_images"
    finally:
        artifacts.clear_run_artifacts()
        config.clear_agentic_image_gen()


@pytest.mark.parametrize("incoming", [None, {}, {"trainer": {}}])
def test_rebinding_without_experiment_name_drops_previous_name(tmp_path, monkeypatch, incoming):
    tmp_path = tmp_path.resolve()
    monkeypatch.setenv("VERLOMNI_ROOT", str(tmp_path))
    try:
        artifacts.bind_run_artifacts(OmegaConf.create({"trainer": {"experiment_name": "old"}}))
        artifacts.bind_run_artifacts(OmegaConf.create(incoming) if incoming is not None else None)
        assert artifacts.resolve_run_dir() == tmp_path / "outputs/e2e/agentic_run"
    finally:
        artifacts.clear_run_artifacts()


def test_artifact_binding_snapshots_config_and_resolves_only_artifact_fields(tmp_path):
    tmp_path = tmp_path.resolve()
    cfg = OmegaConf.create(
        {
            "trainer": {"experiment_name": "run"},
            "agentic_image_gen": {"run_dir": (tmp_path / "${trainer.experiment_name}").as_posix()},
            "unrelated": "${missing_model_path}",
        }
    )
    try:
        artifacts.bind_run_artifacts(cfg)
        cfg.trainer.experiment_name = "changed"
        assert artifacts.resolve_run_dir() == tmp_path / "run"
        assert artifacts.get_run_name() == "run"
    finally:
        artifacts.clear_run_artifacts()
