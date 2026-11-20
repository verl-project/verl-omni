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
"""Launch-time deprecation notices for legacy reward paths."""

import os
import warnings
from unittest.mock import MagicMock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.workers.config.reward import warn_deprecated_reward_config

OCR_SCORER = {"path": "verl_omni/utils/reward_score/genrm_ocr.py", "name": "compute_score_ocr"}


def _config(mutate=None):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.reward_model.enable = False
    config.reward.models = OmegaConf.create({})
    if mutate is not None:
        mutate(config)
    return config


def _launch_notices(config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated_reward_config(config)
    return [str(item.message) for item in caught if item.category is DeprecationWarning]


def test_default_visual_manager_config_warns_once():
    notices = _launch_notices(_config())
    assert len(notices) == 1
    assert "VisualRewardManager is deprecated" in notices[0]


def test_legacy_single_engine_config_warns_for_manager_and_engine():
    def legacy(config):
        config.reward.reward_model.enable = True
        config.reward.custom_reward_function.path = OCR_SCORER["path"]
        config.reward.custom_reward_function.name = OCR_SCORER["name"]

    notices = _launch_notices(_config(legacy))
    assert len(notices) == 2
    assert any("VisualRewardManager is deprecated" in message for message in notices)
    assert any("single global reward model" in message for message in notices)


def test_use_accelerator_alias_warns_at_launch():
    def alias(config):
        config.reward.custom_reward_function.use_accelerator = True

    notices = _launch_notices(_config(alias))
    assert len(notices) == 2
    assert any("use_accelerator is deprecated" in message for message in notices)
    assert any("VisualRewardManager is deprecated" in message for message in notices)


def test_shared_router_hybrid_config_stays_silent():
    def hybrid(config):
        config.reward.reward_model.enable = True
        config.reward.reward_manager.name = "MultiVisualRewardManager"
        config.reward.reward_functions = OmegaConf.create({"ocr": OCR_SCORER})

    assert _launch_notices(_config(hybrid)) == []


def test_named_reward_config_stays_silent():
    def named(config):
        config.reward.reward_manager.name = "MultiVisualRewardManager"
        config.reward.reward_functions = OmegaConf.create(
            {"jpeg": {"path": "verl_omni/utils/reward_score/jpeg_compressibility.py", "name": "compute_score"}}
        )

    assert _launch_notices(_config(named)) == []


def test_text_manager_config_stays_silent():
    def text_manager(config):
        config.reward.reward_manager.name = "naive"

    assert _launch_notices(_config(text_manager)) == []


def test_audio_manager_with_legacy_engine_stays_silent():
    def audio_engine(config):
        config.reward.reward_manager.name = "AudioRewardManager"
        config.reward.reward_model.enable = True

    assert _launch_notices(_config(audio_engine)) == []


def test_null_custom_reward_function_does_not_crash():
    config = _config(lambda config: config.reward.__setattr__("custom_reward_function", None))
    notices = _launch_notices(config)
    assert any("VisualRewardManager is deprecated" in message for message in notices)


def test_config_without_reward_node_stays_silent():
    # Entrypoint unit tests call run_* with minimal configs lacking the reward group.
    assert _launch_notices(OmegaConf.create({})) == []


def test_manager_warning_shares_the_launch_notice_text():
    """VisualRewardManager's constructor warning and the launch notice are one constant."""
    from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager
    from verl_omni.workers.config.reward import DEPRECATED_VISUAL_MANAGER_MESSAGE

    config = _config()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VisualRewardManager(config, MagicMock(), compute_score=None)
    assert DEPRECATED_VISUAL_MANAGER_MESSAGE in [str(item.message) for item in caught]
    assert DEPRECATED_VISUAL_MANAGER_MESSAGE in _launch_notices(config)
