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
"""CPU tests for verl_omni.utils.config validation."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.config import validate_config


def _config(**trainer):
    return OmegaConf.create({"trainer": {"resume_mode": "disable", **trainer}})


def test_validate_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="Available options"):
        validate_config(_config(resume_mode="resumee"))


def test_validate_config_requires_resume_path():
    with pytest.raises(ValueError, match="resume_from_path"):
        validate_config(_config(resume_mode="resume_path"))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("as_dict", [False, True])
def test_validate_config_timestep_staging(enabled, sp_size, as_dict):
    config = _config()
    config.actor_rollout_ref = {
        "actor": {"enable_timestep_staging": enabled, "fsdp_config": {"ulysses_sequence_parallel_size": sp_size}}
    }
    if as_dict:
        config = OmegaConf.to_container(config)
    if enabled and sp_size != 1:
        with pytest.raises(ValueError, match="sequence_parallel_size=1"):
            validate_config(config)
    else:
        validate_config(config)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni", "megatron"])
@pytest.mark.parametrize("enabled", [False, True])
def test_validate_config_no_sync_gradient_accumulation(strategy, enabled):
    config = _config()
    config.actor_rollout_ref = {"actor": {"strategy": strategy, "use_no_sync_for_gradient_accumulation": enabled}}
    if enabled and strategy not in ("fsdp", "fsdp2"):
        with pytest.raises(ValueError, match="fsdp or fsdp2"):
            validate_config(config)
    else:
        validate_config(config)


def test_dynamic_resource_scheduling_default_off_is_admitted():
    validate_config(_config())
    config = _config()
    config.async_training = {"use_dynamic_resource_scheduling": False}
    validate_config(config)


def test_dynamic_resource_scheduling_raises_on_v1_entrypoints():
    config = _config()
    config.async_training = {"use_dynamic_resource_scheduling": True}
    with pytest.raises(ValueError, match="hybrid_rollout.enable_switch"):
        validate_config(config)
