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
"""MiniCPM simplex token, processor, and actor replay contracts without weights."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

from verl_omni.pipelines.minicpm import MiniCPMRolloutAdapter, MiniCPMThinkerAdapter
from verl_omni.pipelines.minicpm.agent_loop import MiniCPMAgentLoopWorker
from verl_omni.pipelines.minicpm.omni_rollout_adapter import (
    _MINICPM_PROCESSED_PROMPT_KEY,
    MINICPM_PROMPT_KEY,
)
from verl_omni.pipelines.minicpm.processor import (
    _load_audio,
    clone_minicpmo_actor_inputs,
    prepare_minicpmo_inputs,
    render_minicpmo_messages,
    split_minicpmo_actor_inputs,
)
from verl_omni.pipelines.minicpm.thinker_training_adapter import (
    _MiniCPMAutoModel,
    _minicpmo_forward,
    _minicpmo_get_vllm_embedding,
    _minicpmo_whisper_attention_forward,
)
from verl_omni.pipelines.model_base import OmniModelBase, OmniRolloutPipelineBase


def test_registry_and_unsupported_stage():
    assert OmniModelBase.get_class_by_name("MiniCPMO", "thinker") is MiniCPMThinkerAdapter
    assert OmniRolloutPipelineBase.get_class("minicpmo_4_5") is MiniCPMRolloutAdapter
    with pytest.raises(NotImplementedError):
        OmniModelBase.get_class_by_name("MiniCPMO", "talker")


def test_native_topology_has_no_duplex_or_talker():
    pytest.importorskip("vllm_omni")
    pipeline = MiniCPMRolloutAdapter._pipeline("thinker_only")
    assert len(pipeline.stages) == 1
    assert pipeline.stages[0].model_stage == "llm"
    assert not pipeline.duplex_control_enabled
    assert pipeline.duplex_runtime_extension is None
    assert MiniCPMRolloutAdapter.weight_sync_stage_ids() == [0]
    assert not MiniCPMRolloutAdapter.supports_async_chunk
    with pytest.raises(ValueError, match="thinker_only"):
        MiniCPMRolloutAdapter.build_stage_configs("full")


def test_teacher_prompt_preserves_response_ids_and_processor_options():
    config = MagicMock()
    config.tokenizer.decode.side_effect = AssertionError("Do not decode the student sequence")
    replay = {"source_ids": [1, 8, 3], "expanded_ids": [1, 4, 4, 4, 3]}
    kwargs = {MINICPM_PROMPT_KEY: replay, "max_slice_nums": 1}
    result = MiniCPMRolloutAdapter.prepare_engine_prompt([1, 4, 4, 4, 3, 19, 20], config, {"image": [object()]}, kwargs)
    assert result == {
        "prompt_token_ids": [1, 4, 4, 4, 3, 19, 20],
        "mm_processor_kwargs": {"max_slice_nums": 1, _MINICPM_PROCESSED_PROMPT_KEY: replay},
    }
    assert MINICPM_PROMPT_KEY in kwargs
    with pytest.raises(ValueError, match="prefix differs"):
        MiniCPMRolloutAdapter.prepare_engine_prompt([1, 7, 3], config, {}, kwargs)
    with pytest.raises(ValueError, match="prompt contract"):
        MiniCPMRolloutAdapter.prepare_engine_prompt([1, 2], config, {"image": [object()]})


def test_text_prompt_never_requires_decoding():
    assert MiniCPMRolloutAdapter.prepare_engine_prompt([1, 2], None, {})["prompt_token_ids"] == [1, 2]


def test_native_rendering_preserves_modality_order():
    processor = MagicMock()
    messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "look"}, {"type": "image"}]}]
    render_minicpmo_messages(processor, messages, add_generation_prompt=True)
    rendered = processor.tokenizer.apply_chat_template.call_args.args[0]
    assert rendered[0]["content"] == "(<audio>./</audio>)look(<image>./</image>)"
    assert isinstance(messages[0]["content"], list)
    render_minicpmo_messages(processor, messages, tokenize=True)
    assert processor.tokenizer.apply_chat_template.call_args.kwargs["tokenize"] is True
    with pytest.raises(ValueError, match="content type"):
        render_minicpmo_messages(processor, [{"role": "user", "content": [{"type": "video"}]}])


def test_processor_batches_media_once_and_checks_counts():
    processor = MagicMock()
    image = object()
    audio = np.zeros(160, dtype=np.float32)
    prepare_minicpmo_inputs(
        processor,
        "(<image>./</image>)(<audio>./</audio>)",
        images=[image],
        audios=[audio],
        mm_processor_kwargs={"sampling_rate": 16000, "max_slice_nums": 1},
    )
    kwargs = processor.call_args.kwargs
    assert kwargs["text"] == ["<image>./</image><audio>./</audio>"]
    assert kwargs["images"] == [[image]]
    assert kwargs["audios"][0][0] is audio
    assert "sampling_rate" not in kwargs
    with pytest.raises(ValueError, match="exactly once"):
        prepare_minicpmo_inputs(processor, "text", images=[image])
    with pytest.raises(NotImplementedError, match="one audio"):
        prepare_minicpmo_inputs(processor, "text", audios=[audio, audio])


def test_native_batch_feature_is_not_reconverted():
    class NativeFeature(dict):
        def convert_to_tensors(self, *args):
            raise AssertionError("Native tensor conversion is not idempotent")

    feature = NativeFeature(input_ids=torch.tensor([[1, 2]]), image_bound=[torch.tensor([[1, 2]])])
    ids, inputs = split_minicpmo_actor_inputs(feature)
    assert ids == [1, 2]
    assert inputs["image_bound"].shape == (1, 2)
    snapshot = clone_minicpmo_actor_inputs(inputs)
    inputs["image_bound"].zero_()
    assert snapshot["image_bound"].tolist() == [[1, 2]]


def test_audio_resampling_and_channel_validation():
    assert _load_audio((np.ones(800, dtype=np.float32), 8000)).shape == (1600,)
    with pytest.raises(ValueError, match="mono"):
        _load_audio(np.ones((2, 800), dtype=np.float32))


def test_actor_media_bounds_account_for_padding_without_mutating_snapshot():
    original = MagicMock()
    model = SimpleNamespace(_verl_minicpmo_original_forward=original)
    bounds = [torch.tensor([[1, 3]]), torch.tensor([[1, 3]])]
    _minicpmo_forward(
        model,
        input_ids=torch.ones(2, 6, dtype=torch.long),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
        image_bound=bounds,
        pixel_values=[[torch.ones(3, 2, 2)], [torch.ones(3, 2, 2)]],
    )
    data = original.call_args.args[0]
    assert data["image_bound"][0].tolist() == [[3, 5]]
    assert data["image_bound"][1].tolist() == [[1, 3]]
    assert bounds[0].tolist() == [[1, 3]]
    assert data["position_ids"].tolist()[0] == [0, 0, 0, 1, 2, 3]


def test_actor_mixed_text_audio_batch_keeps_sample_alignment():
    model = SimpleNamespace(_verl_minicpmo_original_forward=MagicMock())
    _minicpmo_forward(
        model,
        torch.ones(2, 6, dtype=torch.long),
        audio_features=[[], torch.ones(80, 11)],
        audio_feature_lens=[[], torch.tensor([11])],
        audio_bounds=[torch.empty(0, 2, dtype=torch.long), torch.tensor([[1, 3]])],
    )
    data = model._verl_minicpmo_original_forward.call_args.args[0]
    assert data["audio_features"].shape == (2, 80, 11)
    torch.testing.assert_close(data["audio_features"][0], torch.zeros(80, 11))
    torch.testing.assert_close(data["audio_features"][1], torch.ones(80, 11))


def test_actor_text_padding_avoids_dummy_media_encoding():
    model = SimpleNamespace(_verl_minicpmo_original_forward=MagicMock())
    _minicpmo_forward(model, torch.tensor([[1, 2]]))
    data = model._verl_minicpmo_original_forward.call_args.args[0]
    assert data["vision_hidden_states"] == [[]]
    assert data["audio_features"] == []
    with pytest.raises(ValueError, match="remove_padding"):
        MiniCPMThinkerAdapter.prepare_model_inputs({}, None, SimpleNamespace(use_remove_padding=True))


def test_vision_injection_preserves_leaf_gradients():
    embedding = torch.randn(1, 4, 3, requires_grad=True)
    model = SimpleNamespace(
        llm=SimpleNamespace(config=SimpleNamespace(), model=SimpleNamespace(embed_tokens=lambda ids: embedding)),
        get_vision_embedding=lambda data: [torch.ones(1, 2, 3)],
        training=True,
    )
    result, _ = _minicpmo_get_vllm_embedding(
        model, {"input_ids": torch.tensor([[1, 2, 3, 4]]), "image_bound": [torch.tensor([[1, 3]])]}
    )
    result.sum().backward()
    torch.testing.assert_close(embedding.grad[0, 1:3], torch.zeros(2, 3))
    torch.testing.assert_close(embedding.grad[0, 0], torch.ones(3))


def test_whisper_cache_name_and_return_arity():
    original = MagicMock(return_value=(torch.ones(1), None))
    cache = object()
    result = _minicpmo_whisper_attention_forward(
        SimpleNamespace(_verl_minicpmo_original_forward=original), past_key_value=cache
    )
    assert len(result) == 3 and result[2] is cache
    assert original.call_args.kwargs == {"past_key_values": cache}


def test_frozen_whisper_positions_keep_checkpoint_names_without_sharding():
    model = torch.nn.Module()
    model.llm = torch.nn.Module()
    model.llm.embedding = torch.nn.Embedding(8, 4)
    model.llm.get_input_embeddings = lambda: model.llm.embedding
    model.llm.set_input_embeddings = lambda value: None
    model.llm.prepare_inputs_for_generation = lambda **kwargs: kwargs
    model.config = SimpleNamespace()
    model.vpm = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 4))
    image_positions = model.vpm[0].weight.detach().clone()
    model.apm = torch.nn.Module()
    model.apm.embed_positions = torch.nn.Embedding(8, 4)
    model.apm.layers = torch.nn.ModuleList()
    model.tts = torch.nn.Linear(4, 4)
    positions = model.apm.embed_positions.weight.detach().clone()
    MiniCPMThinkerAdapter.configure_model(model, SimpleNamespace())
    assert not hasattr(model, "tts")
    assert not any(parameter.requires_grad for parameter in model.vpm.parameters())
    assert not isinstance(model.apm.embed_positions.weight, torch.nn.Parameter)
    assert not isinstance(model.vpm[0].weight, torch.nn.Parameter)
    torch.testing.assert_close(model.state_dict()["vpm.0.weight"], image_positions)
    torch.testing.assert_close(model.state_dict()["apm.embed_positions.weight"], positions)
    assert "apm.embed_positions.weight" not in dict(model.named_parameters())


def test_remote_loader_initializes_transformers_metadata_once(monkeypatch):
    import transformers.dynamic_module_utils as dynamic

    class NativeModel:
        def __init__(self, config):
            self.initializations = 0

        def post_init(self):
            self.initializations += 1
            self.all_tied_weights_keys = {}

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls(kwargs["config"])

    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", lambda *args: NativeModel)
    config = SimpleNamespace(auto_map={"AutoModel": "model.Native"}, init_tts=True)
    for _ in range(2):
        model = _MiniCPMAutoModel.from_pretrained("model", config=config, trust_remote_code=True)
        assert model.initializations == 1
        assert model.all_tied_weights_keys == {}
    assert config.init_tts is False
    with pytest.raises(ValueError, match="trust_remote_code"):
        _MiniCPMAutoModel.from_pretrained("model", config=config, trust_remote_code=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_token,nonfinite", [(False, False), (True, False), (False, True)])
async def test_replay_worker_checks_shifted_teacher_labels(monkeypatch, wrong_token, nonfinite):
    cls = MiniCPMAgentLoopWorker.__ray_metadata__.modified_class
    monkeypatch.setattr(AgentLoopWorkerTQ.__ray_metadata__.modified_class, "_compute_teacher_logprobs", AsyncMock())
    worker = object.__new__(cls)
    worker.distillation_enabled = True
    output = SimpleNamespace(
        extra_fields={
            "teacher_ids": torch.tensor([[2], [4 if wrong_token else 3], [0]]),
            "teacher_logprobs": torch.tensor([[-0.5], [float("nan") if nonfinite else -0.2], [0.0]]),
        }
    )
    if wrong_token or nonfinite:
        with pytest.raises(ValueError, match="token sequence|non-finite"):
            await worker._compute_teacher_logprobs(output, [1, 2], [3], False)
    else:
        await worker._compute_teacher_logprobs(output, [1, 2], [3], False)


@pytest.mark.asyncio
@pytest.mark.parametrize("validate,task_rewards", [(False, False), (False, True), (True, False)])
async def test_pure_opd_skips_training_task_rewards_but_keeps_validation(monkeypatch, validate, task_rewards):
    parent = AsyncMock()
    monkeypatch.setattr(AgentLoopWorkerTQ.__ray_metadata__.modified_class, "_agent_loop_postprocess", parent)
    worker = object.__new__(MiniCPMAgentLoopWorker.__ray_metadata__.modified_class)
    worker.distillation_enabled = True
    worker.config = SimpleNamespace(
        distillation=SimpleNamespace(distillation_loss=SimpleNamespace(use_task_rewards=task_rewards))
    )
    output = SimpleNamespace(reward_score=None, extra_fields={})
    await worker._agent_loop_postprocess(output, validate, uid="sample")
    parent.assert_awaited_once_with(output, validate, uid="sample")
    if not validate and not task_rewards:
        assert output.reward_score == 0.0
        assert output.extra_fields["reward_extra_info"] == {}
    else:
        assert output.reward_score is None


def test_replay_worker_requires_snapshot():
    worker = object.__new__(MiniCPMAgentLoopWorker.__ray_metadata__.modified_class)
    with pytest.raises(ValueError, match="processor outputs"):
        worker._compute_multi_modal_inputs(SimpleNamespace(), None)
    output = SimpleNamespace(_minicpm_actor_inputs={"image_bound": torch.tensor([[1, 3]])})
    actual = worker._compute_multi_modal_inputs(output, None)
    actual["image_bound"].zero_()
    assert output._minicpm_actor_inputs["image_bound"].tolist() == [[1, 3]]
