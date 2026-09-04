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
"""CPU contracts for Qwen3-TTS's multi-stage rollout integration."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("verl")
pytest.importorskip("vllm_omni")

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from vllm import SamplingParams

from verl_omni.agent_loop.single_turn_agent_loop import OmniSingleTurnAgentLoop
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_tts import omni_rollout_adapter
from verl_omni.pipelines.qwen3_tts.omni_rollout_adapter import Qwen3TTSRolloutAdapter
from verl_omni.pipelines.qwen3_tts.rollout_utils import QWEN3_TTS_REPLAY_KEY
from verl_omni.pipelines.qwen3_tts.talker_training_adapter import Qwen3TTSTalkerAdapter
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        return "first text" if token_ids == [1] else "other text"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_external_module_import_registers_omni_agent_loop():
    code = """
import vllm_omni.platforms as platforms
from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

platforms._current_omni_platform = UnspecifiedOmniPlatform()

import verl_omni
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

target = _agent_loop_registry[\"omni_single_turn_agent\"][\"_target_\"]
assert target == \"verl_omni.agent_loop.single_turn_agent_loop.OmniSingleTurnAgentLoop\"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_optional_rollout_hooks_preserve_existing_ar_defaults():
    first, final = object(), object()

    assert OmniRolloutPipelineBase.supports_async_chunk is True
    assert OmniRolloutPipelineBase.weight_sync_stage_ids() is None
    assert OmniRolloutPipelineBase.prepare_engine_prompt([], None, {}) is None
    assert (
        OmniRolloutPipelineBase.postprocess_agent_loop_output(
            final,
            tokenizer=None,
            response_length=8,
        )
        is final
    )
    assert OmniRolloutPipelineBase.combine_engine_outputs([final], {}) == (final, {})
    with pytest.raises(NotImplementedError, match="multiple final outputs"):
        OmniRolloutPipelineBase.combine_engine_outputs([first, final], {})
    with pytest.raises(RuntimeError, match="no outputs"):
        OmniRolloutPipelineBase.combine_engine_outputs([], {})


def test_omni_single_turn_agent_resolves_registered_pipeline_adapter():
    rollout_config = SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "qwen3_tts_rl"}})

    assert OmniSingleTurnAgentLoop._resolve_rollout_adapter(rollout_config) is Qwen3TTSRolloutAdapter

    missing_config = SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "missing"}})
    with pytest.raises(ValueError, match="requires a registered"):
        OmniSingleTurnAgentLoop._resolve_rollout_adapter(missing_config)


def test_rollout_pipeline_registers_upstream_talker(monkeypatch):
    registered_pipelines = []
    monkeypatch.setattr(
        omni_rollout_adapter,
        "register_pipeline",
        lambda pipeline: registered_pipelines.append(pipeline),
    )

    Qwen3TTSRolloutAdapter.ensure_pipeline_registered()

    assert registered_pipelines == [omni_rollout_adapter.QWEN3_TTS_RL_PIPELINE]
    assert registered_pipelines[0].model_arch == omni_rollout_adapter.QWEN3_TTS_PIPELINE.model_arch


def test_rollout_adapter_builds_unique_prompt_and_scopes_weight_sync(tmp_path):
    speaker = tmp_path / "speaker.json"
    speaker.write_text("[0.0, 1.0]")
    model_config = SimpleNamespace(
        tokenizer=_Tokenizer(),
        override_config={"tts_spk_embed_path": str(speaker), "tts_language": "Auto"},
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2150)),
    )

    first = Qwen3TTSRolloutAdapter.prepare_engine_prompt([1], model_config, {})
    second = Qwen3TTSRolloutAdapter.prepare_engine_prompt([2], model_config, {})

    assert first["additional_information"]["text"] == ["first text"]
    assert first["cache_salt"] != second["cache_salt"]
    assert Qwen3TTSRolloutAdapter.weight_sync_stage_ids("full") == [0]
    assert [
        stage.final_output_type for stage in Qwen3TTSRolloutAdapter.build_stage_configs("full") if stage.final_output
    ] == ["latent", "audio"]


def test_ar_strategy_resolves_qwen3_tts_adapter_and_scopes_weight_sync(monkeypatch):
    server = SimpleNamespace(_rollout_flags={})
    strategy = ARStrategy(server)
    deploy_calls = []
    monkeypatch.setattr(
        strategy,
        "_write_deploy_config",
        lambda engine_kwargs, pipeline_name, adapter_cls, pipeline_mode: deploy_calls.append(
            (pipeline_name, adapter_cls, pipeline_mode)
        ),
    )
    engine_kwargs = {
        "output_mode": "ar",
        "pipeline_name": "qwen3_tts_rl",
        "pipeline_mode": "full",
        "async_chunk": False,
    }

    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert deploy_calls == [("qwen3_tts_rl", Qwen3TTSRolloutAdapter, "full")]
    assert strategy._rollout_adapter is Qwen3TTSRolloutAdapter
    assert strategy._weight_sync_stage_ids == [0]
    assert engine_kwargs == {"async-chunk": False}


@pytest.mark.parametrize("async_chunk", [None, True])
def test_ar_strategy_requires_non_chunked_qwen3_tts_rollout(async_chunk):
    strategy = ARStrategy(SimpleNamespace(_rollout_flags={}))
    engine_kwargs = {
        "pipeline_name": "qwen3_tts_rl",
        "pipeline_mode": "full",
    }
    if async_chunk is not None:
        engine_kwargs["async_chunk"] = async_chunk

    with pytest.raises(ValueError, match="requires async_chunk=false"):
        strategy.preprocess_engine_kwargs(engine_kwargs)


def test_rollout_adapter_requires_speaker_embedding():
    model_config = SimpleNamespace(
        tokenizer=_Tokenizer(),
        override_config={"tts_language": "Auto"},
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2150)),
    )

    with pytest.raises(ValueError, match="requires tts_spk_embed_path"):
        Qwen3TTSRolloutAdapter.prepare_engine_prompt([1], model_config, {})


def test_qwen3_tts_adapters_require_explicit_language(tmp_path):
    speaker = tmp_path / "speaker.json"
    speaker.write_text("[0.0, 1.0]")
    model_config = SimpleNamespace(
        tokenizer=_Tokenizer(),
        override_config={"tts_spk_embed_path": str(speaker)},
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2150)),
    )

    with pytest.raises(ValueError, match="supports only tts_language=Auto"):
        Qwen3TTSRolloutAdapter.prepare_engine_prompt([1], model_config, {})
    with pytest.raises(ValueError, match="supports only tts_language=Auto"):
        Qwen3TTSTalkerAdapter.configure_model(SimpleNamespace(config=SimpleNamespace()), model_config)


def test_talker_adapter_rejects_remove_padding_before_model_configuration():
    model_config = SimpleNamespace(use_remove_padding=True)

    with pytest.raises(ValueError, match="use_remove_padding=false"):
        Qwen3TTSTalkerAdapter.configure_model(SimpleNamespace(), model_config)


def test_talker_adapter_pads_exact_rollout_fields_for_actor_forward():
    model_inputs = {"input_ids": torch.zeros(2, 6, dtype=torch.long)}
    payloads = [
        {"text_ids": [1, 2, 6], "audio_codes": torch.ones(3, 16, dtype=torch.long)},
        {"text_ids": [3, 4, 5], "audio_codes": torch.full((2, 16), 2, dtype=torch.long)},
    ]
    micro_batch = list_of_dict_to_tensordict(
        [
            AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
                metrics=AgentLoopMetrics(),
                extra_fields={QWEN3_TTS_REPLAY_KEY: item},
            ).as_dict()
            for item in payloads
        ]
    )

    prepared = Qwen3TTSTalkerAdapter.prepare_model_inputs(model_inputs, micro_batch, None)

    assert prepared["tts_text_ids"].shape == (2, 3)
    assert prepared["tts_audio_codes"].shape == (2, 3, 16)
    assert prepared["text_len"].tolist() == [3, 3]
    assert prepared["response_len"].tolist() == [3, 2]
    assert not prepared["tts_audio_codes"][1, 2].any()


def test_talker_adapter_requires_namespaced_replay_payload():
    model_inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    micro_batch = TensorDict({}, batch_size=[1])

    with pytest.raises(RuntimeError, match=QWEN3_TTS_REPLAY_KEY):
        Qwen3TTSTalkerAdapter.prepare_model_inputs(model_inputs, micro_batch, None)


def test_rollout_adapter_combines_policy_codes_and_waveform():
    token_ids = [101, 102, 2150]
    generated = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    generated[:, 0] = torch.tensor(token_ids)
    policy = SimpleNamespace(
        stage_id=0,
        outputs=[SimpleNamespace(token_ids=token_ids)],
        multimodal_output={"codes": {"audio": torch.cat((torch.zeros(12, 16), generated))}},
    )
    decoder = SimpleNamespace(
        stage_id=1,
        outputs=[],
        multimodal_output={"audio": torch.ones(2400), "sr": 24_000},
    )
    prompt = {"additional_information": {"text": ["first text"]}}

    selected, fields = Qwen3TTSRolloutAdapter.combine_engine_outputs([policy, decoder], prompt)

    assert selected is policy
    torch.testing.assert_close(fields["tts_audio_codes"], generated.long())
    torch.testing.assert_close(fields["audio"], torch.ones(2400))
    assert fields["audio_sample_rate"] == 24_000
    assert fields["tts_text"] == "first text"


def test_rollout_adapter_prepares_actor_policy_sequence():
    codes = torch.arange(5 * 16, dtype=torch.long).reshape(5, 16)
    output = SimpleNamespace(
        prompt_ids=[9, 8],
        response_ids=[7, 6, 5, 4, 3],
        response_mask=[1] * 5,
        response_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
        extra_fields={
            "tts_audio_codes": codes,
            "tts_text": "first text",
            "audio": torch.ones(2400),
            "audio_sample_rate": 24_000,
        },
    )

    result = Qwen3TTSRolloutAdapter.postprocess_agent_loop_output(
        output,
        tokenizer=_Tokenizer(),
        response_length=3,
    )

    assert result is output
    assert result.prompt_ids == [0]
    assert result.response_ids == codes[:3, 0].tolist()
    assert result.response_mask == [1, 1, 1]
    assert result.response_logprobs == [-0.1, -0.2, -0.3]
    assert "tts_audio_codes" not in result.extra_fields
    assert "tts_text" not in result.extra_fields
    replay = result.extra_fields[QWEN3_TTS_REPLAY_KEY]
    torch.testing.assert_close(replay["audio_codes"], codes[:3])
    assert replay["text_ids"]
    assert result.extra_fields["audio_sample_rate"] == 24_000


def test_ar_strategy_prepares_stage_specific_sampling_params():
    class Adapter:
        @staticmethod
        def prepare_engine_prompt(**kwargs):
            return {
                "prompt_token_ids": [1, 1, 1, 1],
                "additional_information": {"text": ["hello"]},
            }

    server = SimpleNamespace(
        model_config=SimpleNamespace(),
        config=SimpleNamespace(
            max_model_len=64,
            prompt_length=16,
            response_length=8,
            repetition_penalty=1.0,
        ),
        engine=SimpleNamespace(default_sampling_params_list=[SamplingParams(), SimpleNamespace(stage="decoder")]),
    )
    strategy = ARStrategy(server)
    strategy._rollout_adapter = Adapter
    strategy._rollout_output_modalities = ["latent", "audio"]
    strategy._stage_sampling_constraints = {0: {}}

    prompt, params = strategy.preprocess_input(
        [5, 6],
        {"temperature": 0.8, "logprobs": True},
        {},
        None,
        None,
    )

    assert prompt["additional_information"]["max_new_tokens"] == [8]
    assert len(params) == 2
    assert params[0].max_tokens == 8
    assert params[0].temperature == pytest.approx(0.8)
    assert params[0].logprobs == 0
    assert params[1].stage == "decoder"


@pytest.mark.parametrize(
    ("adapter_prompt", "message"),
    [
        ({"additional_information": {"text": ["hello"]}}, "must contain prompt_token_ids"),
        ({"prompt_token_ids": "1,2"}, "list of integers"),
        ([1, 2], "must return a dict or None"),
    ],
)
def test_ar_strategy_rejects_invalid_adapter_prompt(adapter_prompt, message):
    class Adapter:
        @staticmethod
        def prepare_engine_prompt(**kwargs):
            return adapter_prompt

    server = SimpleNamespace(
        model_config=SimpleNamespace(),
        config=SimpleNamespace(max_model_len=64, prompt_length=16, response_length=8),
    )
    strategy = ARStrategy(server)
    strategy._rollout_adapter = Adapter

    with pytest.raises((RuntimeError, TypeError), match=message):
        strategy.preprocess_input([5, 6], {}, {}, None, None)


@pytest.mark.asyncio
async def test_ar_strategy_retains_requested_stage_outputs_and_targets_weight_sync():
    policy = SimpleNamespace(outputs=[])

    class Engine:
        def __init__(self):
            self.generate_kwargs = None
            self.rpc_kwargs = None

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield policy

        async def collective_rpc(self, **kwargs):
            self.rpc_kwargs = kwargs
            return "rpc-result"

    class Adapter:
        @staticmethod
        def combine_engine_outputs(outputs, prompt):
            assert outputs == [policy]
            return policy, {"audio_sample_rate": 24_000}

    server = object.__new__(vLLMOmniHttpServer)
    server.engine = Engine()
    strategy = ARStrategy(server)
    strategy._rollout_output_modalities = ["latent", "audio"]
    strategy._rollout_adapter = Adapter
    strategy._weight_sync_stage_ids = [0]
    server._generate_strategy = strategy

    result = await strategy.run_generation({"prompt_token_ids": [1]}, SamplingParams(), "request-0", None, 0)
    rpc_result = await server.collective_rpc("update_weights_from_ipc", kwargs={"base_sync_done": True})

    assert result is policy
    assert result._verl_omni_rollout_fields == {"audio_sample_rate": 24_000}
    assert server.engine.generate_kwargs["output_modalities"] == ["latent", "audio"]
    assert server.engine.rpc_kwargs["stage_ids"] == [0]
    assert rpc_result == "rpc-result"
