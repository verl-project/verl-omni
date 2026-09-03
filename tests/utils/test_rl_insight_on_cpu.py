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
from types import SimpleNamespace

import pytest

from verl_omni.utils.rl_insight import enable_rl_insight


@pytest.mark.parametrize("logger", ["rl_insight", ["console", "rl_insight"]])
def test_enable_rl_insight_accepts_string_or_list_logger(monkeypatch, logger):
    monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    config = SimpleNamespace(trainer={"logger": logger})

    enable_rl_insight(config)

    assert os.environ["VERL_RL_INSIGHT_ENABLE"] == "1"


@pytest.mark.parametrize("logger", [None, "console", ["console"]])
def test_enable_rl_insight_leaves_environment_unchanged_when_not_selected(monkeypatch, logger):
    monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    config = SimpleNamespace(trainer={"logger": logger})

    enable_rl_insight(config)

    assert "VERL_RL_INSIGHT_ENABLE" not in os.environ


@pytest.mark.parametrize("already_enabled", [False, True])
def test_enable_rl_insight_warns_when_ray_is_already_initialized(monkeypatch, caplog, already_enabled):
    if already_enabled:
        monkeypatch.setenv("VERL_RL_INSIGHT_ENABLE", "1")
    else:
        monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    monkeypatch.setattr("verl_omni.utils.rl_insight.ray.is_initialized", lambda: True)
    config = SimpleNamespace(trainer={"logger": ["rl_insight"]})

    with caplog.at_level("WARNING"):
        enable_rl_insight(config)

    assert "enabled after Ray initialization" in caplog.text
