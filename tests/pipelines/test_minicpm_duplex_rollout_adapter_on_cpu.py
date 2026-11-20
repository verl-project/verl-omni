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
"""CPU tests for the MiniCPM-o 4.5 duplex rollout adapter."""

import pytest

from verl_omni.pipelines.minicpm.duplex_rollout_adapter import (
    MINICPMO_4_5_DUPLEX_PIPELINE,
    MiniCPMODuplexRolloutAdapter,
)
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase


def test_adapter_registers_under_its_own_pipeline_name():
    adapter = OmniRolloutPipelineBase.get_class("minicpmo_4_5_duplex")
    assert adapter is MiniCPMODuplexRolloutAdapter
    # The parent's thinker-only registration stays intact (subclass registration).
    assert OmniRolloutPipelineBase.get_class("minicpmo_4_5") is not MiniCPMODuplexRolloutAdapter


def test_stage_configs_keep_the_native_duplex_runtime():
    stages = MiniCPMODuplexRolloutAdapter.build_stage_configs("duplex")
    (stage,) = stages
    assert stage.stage_id == 0
    assert getattr(stage, "supports_duplex_mrv2", False)
    assert getattr(stage, "supports_native_mrv2_data_plane", False)
    assert MINICPMO_4_5_DUPLEX_PIPELINE.duplex_plugin  # the engine's duplex plugin rides along
    assert MiniCPMODuplexRolloutAdapter.get_pipeline_id("duplex") == "minicpmo_4_5_duplex"


def test_only_the_duplex_mode_is_accepted():
    with pytest.raises(ValueError, match="duplex only"):
        MiniCPMODuplexRolloutAdapter.build_stage_configs("thinker_only")
    with pytest.raises(ValueError, match="no stage 1"):
        MiniCPMODuplexRolloutAdapter.get_stage_engine_extras(1, "duplex")


def test_deploy_config_declares_duplex_session_mode():
    assert MiniCPMODuplexRolloutAdapter.deploy_config_overrides("duplex") == {"session_mode": "duplex"}


def test_thinker_only_stage_extras_stay_clean():
    # The duplex stage must not inherit the thinker-only LLM-arch/async overrides:
    # the duplex runtime owns the model class.
    assert MiniCPMODuplexRolloutAdapter.get_stage_engine_extras(0, "duplex") == {}
    assert MiniCPMODuplexRolloutAdapter.get_engine_hf_overrides("duplex") == {}


def _fake_output(payload):
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    return AgentLoopOutput(
        prompt_ids=[0],
        response_ids=[],
        response_mask=[],
        num_turns=2,
        metrics={},
        extra_fields={"duplex_replay_payload": payload},
    )


class _Tokenizer:
    pass


def test_postprocess_maps_the_replay_onto_the_policy_sequence():
    from tests.utils.dataset.duplex.test_replay_on_cpu import FakeTokenizer, _trajectory
    from verl_omni.utils.dataset.duplex.replay import (
        DUPLEX_REPLAY_KEY,
        DuplexReplay,
        build_replay,
        resolve_duplex_vocabulary,
    )

    replay = build_replay(_trajectory(), resolve_duplex_vocabulary(FakeTokenizer()))
    output = MiniCPMODuplexRolloutAdapter.postprocess_agent_loop_output(
        _fake_output(replay), tokenizer=_Tokenizer(), response_length=256
    )
    assert output.prompt_ids == replay.prompt_ids
    assert output.response_ids == replay.response_ids
    assert output.response_mask == replay.response_mask
    assert output.response_logprobs == replay.response_logprobs
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert "unit_pcm" not in payload  # the actor micro-batch drops the raw PCM
    restored = DuplexReplay.from_extra_fields(payload)
    assert restored.response_ids == replay.response_ids
    assert "duplex_replay_payload" not in output.extra_fields


def test_postprocess_rejects_missing_and_malformed_payloads():
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    empty = AgentLoopOutput(prompt_ids=[0], response_ids=[], response_mask=[], num_turns=2, metrics={}, extra_fields={})
    with pytest.raises(RuntimeError, match="did not return a replay payload"):
        MiniCPMODuplexRolloutAdapter.postprocess_agent_loop_output(empty, tokenizer=_Tokenizer(), response_length=8)
    malformed = AgentLoopOutput(
        prompt_ids=[0],
        response_ids=[],
        response_mask=[],
        num_turns=2,
        metrics={},
        extra_fields={"duplex_replay_payload": {"prompt_ids": []}},
    )
    with pytest.raises(TypeError, match="DuplexReplay"):
        MiniCPMODuplexRolloutAdapter.postprocess_agent_loop_output(malformed, tokenizer=_Tokenizer(), response_length=8)


def test_build_duplex_replay_attaches_streaming_mels():
    from tests.utils.dataset.duplex.test_replay_on_cpu import FakeStreamingProcessor, FakeTokenizer, _trajectory
    from verl_omni.utils.dataset.duplex.replay import resolve_duplex_vocabulary

    replay = MiniCPMODuplexRolloutAdapter.build_duplex_replay(_trajectory(), processor=FakeStreamingProcessor())
    assert replay.unit_mels is not None and len(replay.unit_mels) == len(replay.unit_layouts)
    assert replay.unit_mel_lens == [10] * len(replay.unit_layouts)
    payload = replay.to_extra_fields(include_pcm=False)
    assert payload["unit_mels"] == replay.unit_mels
    resolve_duplex_vocabulary(FakeTokenizer())  # vocabulary resolution stays cheap and cached-safe


def test_require_duplex_vllm_omni_is_loud_without_the_capture_surface():
    import vllm_omni.entrypoints.duplex_omni as entry

    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

    capable = callable(getattr(entry.DuplexSessionHandle, "collect_action_records", None))
    if capable:
        pytest.skip("capture-capable vLLM-Omni installed; the loud-failure path is not reachable")
    with pytest.raises(RuntimeError, match="action-capture"):
        require_duplex_vllm_omni()


def test_capture_surface_probe_accepts_a_capable_build(monkeypatch):
    import vllm_omni.entrypoints.duplex_omni as entry

    capable = type("CapableHandle", (), {"collect_action_records": staticmethod(lambda *args, **kwargs: None)})
    monkeypatch.setattr(entry, "DuplexSessionHandle", capable, raising=True)
    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

    assert require_duplex_vllm_omni() is entry.DuplexOmni
