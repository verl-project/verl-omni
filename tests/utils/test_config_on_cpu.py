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


def _delta_config(**overrides):
    """A minimal omni_delta_sharded config that passes the gates; overrides merge on top."""
    base = {
        "trainer": {"resume_mode": "disable", "use_v1": True, "v1": {"trainer_mode": "separate_async"}},
        "actor_rollout_ref": {
            "model": {"lora": {"rank": 0}},
            "actor": {"fsdp_config": {"qat": {"enable": False}}},
            "rollout": {"checkpoint_engine": {"backend": "omni_delta_sharded"}},
        },
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def test_raw_delta_sharded_backend_is_rejected_with_guidance():
    """verl gates the raw delta_sharded name to sglang; point users at the omni name."""
    config = _delta_config(actor_rollout_ref={"rollout": {"checkpoint_engine": {"backend": "delta_sharded"}}})
    with pytest.raises(ValueError, match="omni_delta_sharded"):
        validate_config(config)


def test_delta_sharded_admitted_for_full_weight_separate_async():
    validate_config(_delta_config())  # diffusion v1 separate_async, full-weight

    # omni_separate_async is admitted without trainer.use_v1 (main_omni forces v1 later)
    config = _delta_config()
    OmegaConf.update(config, "trainer.v1.trainer_mode", "omni_separate_async")
    OmegaConf.update(config, "trainer.use_v1", False)
    validate_config(config)


@pytest.mark.parametrize("mode", ["sync", "omni_sync", "colocate_async"])
def test_delta_sharded_rejects_non_separate_async_modes(mode):
    with pytest.raises(ValueError, match="trainer_mode"):
        validate_config(_delta_config(trainer={"v1": {"trainer_mode": mode}}))


def test_delta_sharded_rejects_legacy_diffusion_trainer():
    # trainer.v1.trainer_mode=separate_async but the v0 runner was selected
    with pytest.raises(ValueError, match="use_v1"):
        validate_config(_delta_config(trainer={"use_v1": False}))


@pytest.mark.parametrize(
    "model_override",
    [
        {"lora_rank": 8},  # adapter sync (merge=false default)
        {"lora_rank": 8, "lora": {"merge": True}},  # merged full-weight export
        {"lora_rank": 16},  # legacy knob
        {"lora_adapter_path": "/tmp/adapter"},
    ],
)
def test_delta_sharded_rejects_lora(model_override):
    with pytest.raises(ValueError, match="full-weight"):
        validate_config(_delta_config(actor_rollout_ref={"model": model_override}))


def test_delta_sharded_rejects_conflicting_lora_spellings():
    """A resolver ValueError means LoRA was requested: the delta gate re-raises
    its own full-weight message instead of the resolver's flat-key hint."""
    with pytest.raises(ValueError, match="full-weight training"):
        validate_config(_delta_config(actor_rollout_ref={"model": {"lora": {"rank": 8}, "lora_rank": 16}}))


def test_delta_sharded_rejects_nested_only_rank():
    """Same for a nested-only rank: the flat-key hint would dead-end at this gate."""
    with pytest.raises(ValueError, match="full-weight training"):
        validate_config(_delta_config(actor_rollout_ref={"model": {"lora": {"rank": 8}}}))


@pytest.mark.parametrize(
    "actor_override",
    [
        {"fsdp_config": {"qat": {"enable": True}}},
        {"megatron": {"qat": {"enable": True}}},
    ],
)
def test_delta_sharded_rejects_qat(actor_override):
    with pytest.raises(ValueError, match="QAT"):
        validate_config(_delta_config(actor_rollout_ref={"actor": actor_override}))


def test_delta_gates_do_not_fire_for_other_backends():
    config = _delta_config(
        trainer={"use_v1": False, "v1": {"trainer_mode": "sync"}},
        actor_rollout_ref={
            "model": {"lora": {"rank": 8}},
            "rollout": {"checkpoint_engine": {"backend": "nccl"}},
        },
    )
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


def _rollout_config(**overrides):
    """A minimal config that passes the gates; rollout keys are overridable."""
    rollout = {
        "name": "vllm_omni",
        "expert_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "data_parallel_size": 1,
    }
    rollout.update(overrides)
    return OmegaConf.create({"trainer": {"resume_mode": "disable"}, "actor_rollout_ref": {"rollout": rollout}})


def test_rollout_ep_requires_tp_times_dp():
    """vLLM derives the EP degree from TP x DP; a mismatched size cannot be honored."""
    config = _rollout_config(expert_parallel_size=2, tensor_model_parallel_size=1)
    with pytest.raises(ValueError, match=r"tensor_model_parallel_size \* data_parallel_size"):
        validate_config(config)


def test_rollout_ep_matching_tp_times_dp_is_admitted():
    validate_config(_rollout_config(expert_parallel_size=2, tensor_model_parallel_size=2))
    validate_config(_rollout_config(expert_parallel_size=4, tensor_model_parallel_size=4))


def test_rollout_ep_one_is_admitted_for_any_sizes():
    validate_config(_rollout_config(expert_parallel_size=1, tensor_model_parallel_size=8, data_parallel_size=4))


def test_rollout_ep_gate_accepts_a_plain_dict_config():
    config = OmegaConf.to_container(_rollout_config(expert_parallel_size=2, tensor_model_parallel_size=2))
    validate_config(config)


def test_diffusion_rollout_ep_is_rejected():
    """No diffusion stage reads the knob, so EP>1 there cannot be honored."""
    config = _rollout_config(expert_parallel_size=2, tensor_model_parallel_size=2)
    config.actor_rollout_ref.rollout.rollout_attn_backend = "FLASH_ATTN_3_HUB"
    with pytest.raises(ValueError, match="diffusion rollout"):
        validate_config(config)


def test_rollout_parallel_config_escape_hatch_is_rejected():
    """engine_kwargs.vllm_omni.parallel_config bypasses the size accounting."""
    config = _rollout_config(engine_kwargs={"vllm_omni": {"parallel_config": {"enable_expert_parallel": True}}})
    with pytest.raises(ValueError, match="bypasses"):
        validate_config(config)


def test_rollout_flat_enable_expert_parallel_is_rejected():
    """The flat spelling is never honored on AR and unvalidated on diffusion."""
    config = _rollout_config(engine_kwargs={"vllm_omni": {"enable_expert_parallel": True}})
    with pytest.raises(ValueError, match="not a supported spelling"):
        validate_config(config)


def test_rollout_stage_override_ep_is_rejected():
    """The vLLM-Omni resolver applies stage_overrides over the deploy config."""
    config = _rollout_config(engine_kwargs={"vllm_omni": {"stage_overrides": {"0": {"enable_expert_parallel": True}}}})
    with pytest.raises(ValueError, match="stage_overrides"):
        validate_config(config)


def test_rollout_ep_with_dp_greater_than_one_is_rejected():
    """DP>1 cannot express EP in the deploy config, so accepting it would lie."""
    config = _rollout_config(expert_parallel_size=4, tensor_model_parallel_size=2, data_parallel_size=2)
    with pytest.raises(ValueError, match=r"data_parallel_size=2 is not wired"):
        validate_config(config)


def test_rollout_non_positive_ep_size_is_rejected():
    """0 and negative sizes are not meaningful degrees; refuse them loudly."""
    for bad in (0, -2):
        with pytest.raises(ValueError, match="positive integer"):
            validate_config(_rollout_config(expert_parallel_size=bad))
