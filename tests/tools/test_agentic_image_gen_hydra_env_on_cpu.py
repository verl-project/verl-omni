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

"""CPU tests for Hydra ``agentic_image_gen`` process-local bind."""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl_omni
from verl_omni.tools.trajectory.hydra_env import (
    agentic_get,
    agentic_get_bool,
    agentic_get_str,
    bind_agentic_image_gen,
    clear_agentic_image_gen,
)
from verl_omni.utils.agentic_image_judge_parse import good_enough_threshold

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


def test_omni_trainer_composes_agentic_image_gen():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="omni_trainer")
    assert "agentic_image_gen" in cfg
    assert cfg.agentic_image_gen.max_generate_image_passes == 3
    assert cfg.agentic_image_gen.block_generate_after_yes is True
    assert cfg.agentic_image_gen.force_first_generate is False
    assert cfg.agentic_image_gen.force_reflection_after_judge is True
    assert cfg.agentic_image_gen.rewrite_judge_before_generate is True
    assert cfg.agentic_image_gen.e2e_root is None
    assert cfg.agentic_image_gen.good_enough_threshold == 0.80
    assert cfg.agentic_image_gen.qwen_image_height == 512
    assert cfg.agentic_image_gen.qwen_image_width == 512
    assert cfg.agentic_image_gen.qwen_image_steps == 20
    assert cfg.agentic_image_gen.qwen_image_true_cfg_scale == 4.0
    assert cfg.agentic_image_gen.qwen_image_seed is None
    assert cfg.agentic_image_gen.qwen_image_diversify_seed is True


def test_bind_agentic_image_gen_stores_diffusion_url():
    clear_agentic_image_gen()
    cfg = OmegaConf.create(
        {
            "agentic_image_gen": {
                "diffusion_tool_url": "http://127.0.0.1:9999/generate",
                "diffusion_tool_token": None,
                "block_generate_after_yes": False,
                "max_generate_image_passes": 5,
                "force_first_generate": True,
                "force_first_warmup_steps": 100,
                "force_first_end_step": 200,
                "force_reflection_after_judge": False,
            }
        }
    )
    bind_agentic_image_gen(cfg)

    assert agentic_get_str("diffusion_tool_url") == "http://127.0.0.1:9999/generate"
    assert agentic_get_bool("block_generate_after_yes") is False
    assert agentic_get("max_generate_image_passes") == 5
    assert agentic_get_bool("force_first_generate") is True
    assert agentic_get("force_first_warmup_steps") == 100
    assert agentic_get("force_first_end_step") == 200
    assert agentic_get_bool("force_reflection_after_judge") is False
    assert agentic_get("diffusion_tool_token") is None
    clear_agentic_image_gen()


def test_bind_stores_qwen_image_geometry_and_threshold():
    clear_agentic_image_gen()
    bind_agentic_image_gen(
        OmegaConf.create(
            {
                "agentic_image_gen": {
                    "qwen_image_height": 256,
                    "qwen_image_width": 384,
                    "qwen_image_steps": 8,
                    "qwen_image_true_cfg_scale": 2.5,
                    "qwen_image_seed": 7,
                    "qwen_image_diversify_seed": False,
                    "good_enough_threshold": 0.6,
                }
            }
        )
    )
    assert agentic_get("qwen_image_height") == 256
    assert agentic_get("qwen_image_width") == 384
    assert agentic_get("qwen_image_steps") == 8
    assert agentic_get("qwen_image_true_cfg_scale") == 2.5
    assert agentic_get("qwen_image_seed") == 7
    assert agentic_get_bool("qwen_image_diversify_seed") is False
    assert agentic_get("good_enough_threshold") == 0.6
    clear_agentic_image_gen()


def test_bind_clears_stale_cfg_when_agentic_image_gen_missing():
    clear_agentic_image_gen()
    bind_agentic_image_gen(
        OmegaConf.create(
            {
                "agentic_image_gen": {
                    "diffusion_tool_url": "stale-url",
                    "force_first_generate": True,
                }
            }
        )
    )
    assert agentic_get_str("diffusion_tool_url") == "stale-url"
    bind_agentic_image_gen(OmegaConf.create({"trainer": {}}))
    assert agentic_get_str("diffusion_tool_url") == ""
    assert agentic_get_bool("force_first_generate") is False
    bind_agentic_image_gen(None)
    assert agentic_get_str("diffusion_tool_url") == ""
    clear_agentic_image_gen()


def test_good_enough_threshold_hydra_fail_closed():
    clear_agentic_image_gen()
    with pytest.raises(RuntimeError, match="unbound"):
        good_enough_threshold()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {}}))
    assert good_enough_threshold() == 0.80
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"good_enough_threshold": 0.6}}))
    assert good_enough_threshold() == 0.6
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"good_enough_threshold": "garbage"}}))
    with pytest.raises(ValueError, match="good_enough_threshold"):
        good_enough_threshold()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"good_enough_threshold": 1.5}}))
    with pytest.raises(ValueError, match="good_enough_threshold"):
        good_enough_threshold()
    clear_agentic_image_gen()


def test_agentic_get_unbound_fails_loud():
    clear_agentic_image_gen()
    with pytest.raises(RuntimeError, match="unbound"):
        agentic_get("vllm_url")
    with pytest.raises(RuntimeError, match="unbound"):
        agentic_get_str("vllm_url")
    # Explicit defaults still allowed for test helpers.
    assert agentic_get("vllm_url", "") == ""
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {}}))
    assert agentic_get_str("vllm_url") == ""
    clear_agentic_image_gen()


def test_agentic_scorer_knobs_from_config_rejects_none():
    from verl_omni.tools.trajectory.hydra_env import agentic_scorer_knobs_from_config

    with pytest.raises(ValueError, match="composed Hydra config"):
        agentic_scorer_knobs_from_config(None)


def test_yaml_defaults_are_single_source_of_truth():
    from verl_omni.tools.trajectory.hydra_env import yaml_agentic_image_gen_defaults

    defaults = yaml_agentic_image_gen_defaults()
    assert defaults["max_generate_image_passes"] == 3
    assert defaults["good_enough_threshold"] == 0.80
    assert "vllm_url" in defaults


def test_merge_agentic_scorer_knobs_requires_stamp_or_config():
    from omegaconf import OmegaConf

    from verl_omni.tools.trajectory.hydra_env import SCORER_KNOB_KEYS, merge_agentic_scorer_knobs

    with pytest.raises(ValueError, match="missing from extra_info"):
        merge_agentic_scorer_knobs({"w_tool_call": 0.1}, None)

    stamped = {key: "x" for key in SCORER_KNOB_KEYS}
    stamped["w_tool_call"] = 0.1
    merged = merge_agentic_scorer_knobs(stamped, None)
    assert merged["w_tool_call"] == 0.1
    assert merged["good_enough_threshold"] == "x"

    cfg = OmegaConf.create({"agentic_image_gen": {"good_enough_threshold": 0.6, "vllm_url": "http://x"}})
    merged2 = merge_agentic_scorer_knobs({"w_tool_call": 0.1}, cfg)
    assert merged2["good_enough_threshold"] == 0.6
    assert merged2["vllm_url"] == "http://x"
    assert merged2["w_tool_call"] == 0.1
