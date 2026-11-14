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
"""CPU tests for rollout expert-parallel honor-or-reject semantics.

The contract: ``actor_rollout_ref.rollout.expert_parallel_size`` either configures
expert parallelism or fails loudly. vLLM-Omni takes EP as a per-stage flag over the
ranks a stage already spans, so the rollout forwards it only for pipelines that opt
a stage in, and refuses the knob everywhere else.
"""

import types

import pytest
import yaml

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_omni.omni_rollout_adapter import Qwen3OmniRolloutAdapter
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

QWEN3_OMNI_PIPELINE = "qwen3_omni_moe"


def _ar_strategy(*, ep=1, tp=2, dp=1, pipeline_name=QWEN3_OMNI_PIPELINE, pipeline_mode="thinker_only"):
    config = types.SimpleNamespace(
        max_model_len=None,
        prompt_length=512,
        response_length=256,
        expert_parallel_size=ep,
        tensor_model_parallel_size=tp,
        data_parallel_size=dp,
        engine_kwargs={"vllm_omni": {"pipeline_name": pipeline_name, "pipeline_mode": pipeline_mode}},
    )
    return ARStrategy(types.SimpleNamespace(config=config))


def _diffusion_strategy(*, ep=1, parallel_config=None, flat_flag=False, stage_overrides=None):
    omni_kwargs: dict = {}
    if parallel_config is not None:
        omni_kwargs["parallel_config"] = parallel_config
    if flat_flag:
        omni_kwargs["enable_expert_parallel"] = True
    if stage_overrides is not None:
        omni_kwargs["stage_overrides"] = stage_overrides
    config = types.SimpleNamespace(expert_parallel_size=ep, engine_kwargs={"vllm_omni": omni_kwargs})
    return DiffusionStrategy(types.SimpleNamespace(config=config))


# ---------------------------------------------------------------------------
# Qwen3-Omni opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline_mode", ["thinker_only", "thinker_talker", "full"])
def test_qwen3_omni_opts_its_policy_stage_in(pipeline_mode):
    """The Thinker is the policy stage and the only stage declared EP-capable."""
    assert Qwen3OmniRolloutAdapter.expert_parallel_stage_ids(pipeline_mode) == frozenset({0})
    assert Qwen3OmniRolloutAdapter.policy_stage_id(pipeline_mode=pipeline_mode) == 0


def test_qwen3_omni_is_listed_as_an_ep_pipeline():
    """The refusal message names the supported combinations from the registry."""
    assert QWEN3_OMNI_PIPELINE in OmniRolloutPipelineBase.expert_parallel_pipelines()


def test_qwen3_omni_rejects_an_unknown_pipeline_mode():
    with pytest.raises(ValueError, match="Unknown pipeline_mode"):
        Qwen3OmniRolloutAdapter.expert_parallel_stage_ids("thinker_only_typo")


# ---------------------------------------------------------------------------
# AR rollout honor-or-reject
# ---------------------------------------------------------------------------


def test_ep_one_is_admitted_without_touching_the_deploy_config():
    strategy = _ar_strategy(ep=1, tp=4)
    strategy.validate_configs()
    assert strategy._expert_parallel_stage_ids == frozenset()


def test_matching_ep_is_honored_on_the_policy_stage():
    strategy = _ar_strategy(ep=2, tp=2, dp=1)
    strategy.validate_configs()
    assert strategy._expert_parallel_stage_ids == frozenset({0})


def test_ep_must_equal_tp_times_dp():
    """vLLM derives the EP degree from TP x DP; a mismatched size cannot be honored."""
    strategy = _ar_strategy(ep=4, tp=2, dp=1)
    with pytest.raises(ValueError, match=r"tensor_model_parallel_size \* data_parallel_size"):
        strategy.validate_configs()


def test_ep_with_dp_greater_than_one_is_refused():
    """The deploy config cannot express DP, so EP=TP*DP would silently deliver EP=TP."""
    strategy = _ar_strategy(ep=4, tp=2, dp=2)
    with pytest.raises(ValueError, match=r"data_parallel_size=2 is not wired"):
        strategy.validate_configs()


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {"enable_expert_parallel": True},
        {"parallel_config": {"enable_expert_parallel": True}},
        {"stage_overrides": {"0": {"enable_expert_parallel": True}}},
    ],
)
def test_replica_rejects_engine_side_ep_overrides(engine_kwargs):
    """The replica is validated without the driver gate, so it must refuse EP=1 too."""
    strategy = _ar_strategy(ep=1, tp=2)
    strategy.server.config.engine_kwargs["vllm_omni"].update(engine_kwargs)
    with pytest.raises(ValueError, match="expert_parallel"):
        strategy.validate_configs()


def test_replica_normalizes_an_unset_size_before_launch():
    """A null size would crash the launcher, so the replica normalizes it to 1."""
    strategy = _ar_strategy(ep=None, tp=2)
    strategy.validate_configs()
    assert strategy.server.config.expert_parallel_size == 1


@pytest.mark.parametrize("bad_ep", [2.9, True])
def test_replica_rejects_non_integer_sizes(bad_ep):
    """Truncating 2.9 to 2 (or True to 1) would silently change the EP degree."""
    strategy = _ar_strategy(ep=bad_ep, tp=2)
    with pytest.raises(ValueError, match="expert_parallel_size"):
        strategy.validate_configs()


def test_validate_configs_then_deploy_config_carries_the_flag(monkeypatch):
    """The validated stage set is what the generated YAML is built from."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    strategy = _ar_strategy(ep=2, tp=2)
    strategy.validate_configs()
    engine_kwargs = {"pipeline_name": QWEN3_OMNI_PIPELINE, "pipeline_mode": "thinker_only"}
    strategy.preprocess_engine_kwargs(engine_kwargs)
    with open(engine_kwargs["deploy-config"]) as handle:
        deploy = yaml.safe_load(handle)
    assert deploy["stages"][0]["enable_expert_parallel"] is True

    accepted = _ar_strategy(ep=1, tp=2)
    accepted.validate_configs()
    engine_kwargs = {"pipeline_name": QWEN3_OMNI_PIPELINE, "pipeline_mode": "thinker_only"}
    accepted.preprocess_engine_kwargs(engine_kwargs)
    with open(engine_kwargs["deploy-config"]) as handle:
        deploy = yaml.safe_load(handle)
    assert all("enable_expert_parallel" not in stage for stage in deploy["stages"])


def test_pipeline_without_an_ep_opt_in_is_refused():
    strategy = _ar_strategy(ep=2, tp=2, pipeline_name="no_such_pipeline")
    with pytest.raises(ValueError, match="not supported by the selected rollout pipeline"):
        strategy.validate_configs()


def test_prepare_engine_args_drops_the_global_ep_flag():
    """The per-stage deploy config is the only channel; the global arg must not survive."""
    strategy = _ar_strategy(ep=2, tp=2)
    strategy._rollout_output_modalities = None
    engine_args = {"enable_expert_parallel": True, "model": "qwen3-omni"}
    strategy.prepare_engine_args(engine_args, types.SimpleNamespace())
    assert "enable_expert_parallel" not in engine_args


def test_refusal_names_the_supported_pipelines():
    strategy = _ar_strategy(ep=2, tp=2, pipeline_name="no_such_pipeline")
    with pytest.raises(ValueError, match=QWEN3_OMNI_PIPELINE):
        strategy.validate_configs()


def test_pipeline_that_opts_out_of_ep_is_refused(monkeypatch):
    """A registered pipeline whose hook stays empty must not silently accept EP."""

    class DenseAdapter(OmniRolloutPipelineBase):
        pass

    monkeypatch.setattr(OmniRolloutPipelineBase, "get_class", classmethod(lambda cls, model_type: DenseAdapter))
    strategy = _ar_strategy(ep=2, tp=2)
    with pytest.raises(ValueError, match="not supported by the selected rollout pipeline"):
        strategy.validate_configs()


# ---------------------------------------------------------------------------
# Diffusion rollout: dead knob, refused
# ---------------------------------------------------------------------------


def test_diffusion_ep_is_refused():
    strategy = _diffusion_strategy(ep=2)
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_parallel_config_escape_hatch_is_refused():
    """engine_kwargs.vllm_omni.parallel_config is copied verbatim; refuse it too."""
    strategy = _diffusion_strategy(ep=1, parallel_config={"enable_expert_parallel": True})
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_flat_enable_expert_parallel_is_refused():
    """The diffusion ingress forwards the flat flag unvalidated, so it is a live hatch."""
    strategy = _diffusion_strategy(ep=1, flat_flag=True)
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_stage_override_ep_is_refused():
    """stage_overrides entries are applied over the deploy config, so they are a hatch too."""
    strategy = _diffusion_strategy(ep=1, stage_overrides={"0": {"enable_expert_parallel": True}})
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_ep_one_is_admitted():
    _diffusion_strategy(ep=1).validate_configs()
    _diffusion_strategy(ep=1, parallel_config={"enable_expert_parallel": False}).validate_configs()


@pytest.mark.parametrize("bad_ep", [0, -2])
def test_diffusion_non_positive_ep_is_refused(bad_ep):
    """The replica is reachable without the driver gate, so it must reject 0/negative."""
    strategy = _diffusion_strategy(ep=bad_ep)
    with pytest.raises(ValueError, match="positive integer"):
        strategy.validate_configs()


def test_diffusion_normalizes_an_unset_size_before_launch():
    """verl's launch_server compares ``expert_parallel_size > 1``, so None must become 1."""
    strategy = _diffusion_strategy(ep=None)
    strategy.validate_configs()
    assert strategy.server.config.expert_parallel_size == 1
