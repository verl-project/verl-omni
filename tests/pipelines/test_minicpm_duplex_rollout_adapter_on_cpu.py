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
    # Engine-captured logprobs stay in the replay payload; the output column
    # stays unset so verl's rollout-correction never mixes them with the
    # trainer-recomputed old_log_probs.
    assert output.response_logprobs is None
    payload = output.extra_fields[DUPLEX_REPLAY_KEY]
    assert payload["response_logprobs"] == replay.response_logprobs
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
    from verl_omni.utils.dataset.duplex.replay import audio_tokens_for_mel_frames, resolve_duplex_vocabulary

    replay = MiniCPMODuplexRolloutAdapter.build_duplex_replay(_trajectory(), processor=FakeStreamingProcessor())
    assert replay.unit_mels is not None and len(replay.unit_mels) == len(replay.unit_layouts)
    # Mel frames per unit window: the 1035 ms pre-roll ingestion then 1000 ms chunks.
    assert replay.unit_mel_lens == [104, 101, 101, 101]
    assert all(
        audio_tokens_for_mel_frames(mel_frames) == layout.num_audio_tokens
        for mel_frames, layout in zip(replay.unit_mel_lens, replay.unit_layouts, strict=True)
    )
    payload = replay.to_extra_fields(include_pcm=False)
    assert payload["unit_mels"] == replay.unit_mels
    resolve_duplex_vocabulary(FakeTokenizer())  # vocabulary resolution stays cheap and cached-safe


def test_build_duplex_replay_resolves_the_bundled_ref_audio_from_the_model_path(tmp_path):
    # The engine embeds <model snapshot>/assets/HT_ref_audio.wav in every
    # duplex session; threading the actor checkpoint path through
    # build_duplex_replay must reproduce the 70-id context (8 prefix + 60
    # reference rows + 2 suffix) with the reference block as the first span.
    import numpy as np
    import soundfile as sf

    from tests.utils.dataset.duplex.test_replay_on_cpu import FakeStreamingProcessor, _trajectory
    from verl_omni.utils.dataset.duplex.replay import audio_tokens_for_samples

    samples = 96256  # the bundled asset: 6.016 s at 16 kHz
    assert audio_tokens_for_samples(samples) == 60
    snapshot = tmp_path / "snapshot"
    (snapshot / "assets").mkdir(parents=True)
    sf.write(str(snapshot / "assets" / "HT_ref_audio.wav"), np.zeros(samples, dtype=np.float32), 16000)
    replay = MiniCPMODuplexRolloutAdapter.build_duplex_replay(
        _trajectory(), processor=FakeStreamingProcessor(), model_path=str(snapshot)
    )
    assert len(replay.prompt_ids) == 70
    assert replay.ref_audio is not None and replay.ref_audio.num_audio_tokens == 60
    assert replay.audio_bounds[0] == [8, 68]
    assert replay.unit_mels is not None and len(replay.unit_mels) == 5  # reference clip + four units
    assert replay.unit_mel_lens[0] == 602  # the offline-tower frames of the bundled clip


def test_replay_dump_is_env_gated_and_mel_free(tmp_path, monkeypatch):
    import json

    from tests.utils.dataset.duplex.test_replay_on_cpu import FakeStreamingProcessor, _trajectory
    from verl_omni.agent_loop.omni_duplex_agent_loop import REPLAY_DUMP_DIR_ENV, dump_duplex_replay

    replay = MiniCPMODuplexRolloutAdapter.build_duplex_replay(_trajectory(), processor=FakeStreamingProcessor())
    output = MiniCPMODuplexRolloutAdapter.postprocess_agent_loop_output(
        _fake_output(replay), tokenizer=_Tokenizer(), response_length=256
    )
    # Unset: nothing is written.
    monkeypatch.delenv(REPLAY_DUMP_DIR_ENV, raising=False)
    assert dump_duplex_replay(output.extra_fields, episode_id="ep-1") is False
    assert not (tmp_path / "duplex_replay.jsonl").exists()
    # Set: one line-buffered JSON line per episode, PCM- and mel-free (the
    # FDB input).
    dump_dir = tmp_path / "dump"
    monkeypatch.setenv(REPLAY_DUMP_DIR_ENV, str(dump_dir))
    assert dump_duplex_replay(output.extra_fields, episode_id="ep-1") is True
    lines = (dump_dir / "duplex_replay.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["episode_id"] == "ep-1"
    assert "unit_mels" not in entry and "unit_mel_lens" not in entry and "unit_pcm" not in entry
    assert entry["unit_layouts"] and entry["event"]["axis"] == "turn"


def test_require_duplex_vllm_omni_is_loud_without_the_capture_surface(monkeypatch):
    import vllm_omni.entrypoints.duplex_omni as entry

    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

    # The probe reads the post-close surface the strategy drains
    # (DuplexOmni.collect_action_records); a build without it fails loudly.
    incapable = type("IncapableOmni", (), {})
    monkeypatch.setattr(entry, "DuplexOmni", incapable, raising=True)
    with pytest.raises(RuntimeError, match="action-capture"):
        require_duplex_vllm_omni()


def test_capture_surface_probe_accepts_a_capable_build(monkeypatch):
    import vllm_omni.entrypoints.duplex_omni as entry

    capable = type("CapableOmni", (), {"collect_action_records": staticmethod(lambda *args, **kwargs: None)})
    monkeypatch.setattr(entry, "DuplexOmni", capable, raising=True)
    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

    assert require_duplex_vllm_omni() is entry.DuplexOmni
