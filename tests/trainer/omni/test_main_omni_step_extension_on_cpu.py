# Copyright 2026 verl-omni contributors
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

"""Reproduce an opt-in that is ignored outside its supported trainer."""

import pytest
import verl.trainer.main_ppo as upstream_entry
from omegaconf import OmegaConf

from verl_omni.trainer import main_omni


@pytest.mark.parametrize("mode", ["omni_sync", "sync", "separate_async"])
def test_unsupported_mode_must_not_silently_ignore_enabled_extension(mode, monkeypatch):
    started = []
    monkeypatch.setattr(main_omni, "enable_rl_insight", lambda _: None)
    monkeypatch.setattr(upstream_entry, "run_ppo", lambda *args, **kwargs: started.append(True))
    config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "disable",
                "use_v1": True,
                "v1": {"trainer_mode": mode, "reuse_proximal_actor_forward": True},
            }
        }
    )
    with pytest.raises(ValueError, match="omni_separate_async"):
        main_omni.run_omni(config)
    assert started == []


@pytest.mark.parametrize("enabled,mode", [(False, "omni_sync"), (True, "omni_separate_async")])
def test_supported_or_disabled_config_reaches_native_launcher(enabled, mode, monkeypatch):
    started = []
    monkeypatch.setattr(main_omni, "enable_rl_insight", lambda _: None)
    monkeypatch.setattr(upstream_entry, "run_ppo", lambda *args, **kwargs: started.append(True))
    config = OmegaConf.create(
        {"trainer": {"use_v1": True, "v1": {"trainer_mode": mode, "reuse_proximal_actor_forward": enabled}}}
    )
    main_omni.run_omni(config)
    assert started == [True]


def test_offline_trainer_rejects_enabled_extension_before_launch(monkeypatch):
    started = []
    monkeypatch.setattr(main_omni, "enable_rl_insight", lambda _: None)
    monkeypatch.setattr(main_omni, "launch_ray_task_runner", lambda *args: started.append(True))
    config = OmegaConf.create(
        {
            "algorithm": {"sample_source": "offline", "trainer_type": "direct_preference"},
            "trainer": {"v1": {"trainer_mode": "omni_separate_async", "reuse_proximal_actor_forward": True}},
        }
    )
    with pytest.raises(ValueError, match="omni_separate_async"):
        main_omni.run_omni(config, task_runner_class=object())
    assert started == []


def test_named_extension_rejects_unsupported_mode_before_launch(monkeypatch):
    started = []
    monkeypatch.setattr(main_omni, "enable_rl_insight", lambda _: None)
    monkeypatch.setattr(upstream_entry, "run_ppo", lambda *args, **kwargs: started.append(True))
    config = OmegaConf.create(
        {"trainer": {"use_v1": True, "v1": {"trainer_mode": "omni_sync", "step_extension": "proximal_actor_reuse"}}}
    )
    with pytest.raises(ValueError, match="omni_separate_async"):
        main_omni.run_omni(config)
    assert started == []


def test_named_extension_in_supported_mode_reaches_native_launcher(monkeypatch):
    started = []
    monkeypatch.setattr(main_omni, "enable_rl_insight", lambda _: None)
    monkeypatch.setattr(upstream_entry, "run_ppo", lambda *args, **kwargs: started.append(True))
    config = OmegaConf.create(
        {
            "trainer": {
                "use_v1": True,
                "v1": {"trainer_mode": "omni_separate_async", "step_extension": "proximal_actor_reuse"},
            }
        }
    )
    main_omni.run_omni(config)
    assert started == [True]
