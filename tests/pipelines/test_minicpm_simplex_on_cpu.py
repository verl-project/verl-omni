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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_actor_model_cast_preserves_rotary_frequencies(monkeypatch, dtype):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    import verl_omni.pipelines.minicpm.thinker_training_adapter as adapter

    # The stub has no media towers for the remote-code patches to wrap.
    monkeypatch.setattr(adapter, "_apply_remote_code_patches", lambda module: None)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=1000000.0,
    )
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.llm = Qwen3ForCausalLM(config)
    rotary = model.llm.model.rotary_emb
    frequencies = rotary.inv_freq.clone()
    x = torch.zeros(1, 3, 64, dtype=dtype)
    positions = torch.tensor([[0, 1023, 4095]])
    expected = rotary(x, positions)

    MiniCPMThinkerAdapter.configure_model(model, SimpleNamespace())
    model.to(dtype=dtype)

    assert model.llm.model.embed_tokens.weight.dtype == dtype
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.original_inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, frequencies, atol=0, rtol=0)
    torch.testing.assert_close(rotary.original_inv_freq, frequencies, atol=0, rtol=0)
    for actual, reference in zip(rotary(x, positions), expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert not any("inv_freq" in key for key in model.state_dict())


def test_remote_loader_restores_nonpersistent_resampler_positions(monkeypatch, tmp_path):
    from transformers import AutoModel, PretrainedConfig, PreTrainedModel

    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMO

    class Resampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.max_size = (2, 2)
            self._set_2d_pos_cache(self.max_size)

        def _set_2d_pos_cache(self, size, device="cpu"):
            positions = torch.arange(size[0] * size[1], dtype=torch.float32, device=device).cos()
            self.register_buffer("pos_embed", positions, persistent=False)

    class NativeModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.resampler = Resampler()
            self.post_init()

    config = PretrainedConfig()
    source = NativeModel(config)
    source.resampler._set_2d_pos_cache(source.resampler.max_size)
    expected = source.resampler.pos_embed.clone()
    source.save_pretrained(tmp_path)
    monkeypatch.setattr("verl_omni.models.transformers.minicpm_o.patch_minicpm_auto_model_init", lambda *a, **k: None)
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.patch_minicpm_siglip_flash_attn_support", lambda *a, **k: None
    )
    monkeypatch.setattr(
        AutoModel, "from_pretrained", lambda path, **kwargs: NativeModel.from_pretrained(path, config=kwargs["config"])
    )

    loaded = MiniCPMO.from_pretrained(str(tmp_path), config=config, trust_remote_code=True)

    torch.testing.assert_close(loaded.resampler.pos_embed, expected, atol=0, rtol=0)
    assert "resampler.pos_embed" not in loaded.state_dict()
    torch.testing.assert_close(loaded.resampler.proj.weight, source.resampler.proj.weight)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("response_length", [3, 4])
async def test_agent_exposes_token_budget_to_evaluation(monkeypatch, response_length):
    import asyncio

    import verl_omni.pipelines.minicpm.agent_loop as module

    agent = object.__new__(module.MiniCPMSimplexAgentLoop)
    agent.loop = asyncio.get_running_loop()
    agent.processor = object()
    agent.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    agent.apply_chat_template_kwargs = {}
    agent.rollout_config = SimpleNamespace(prompt_length=16)
    agent.response_length = 4
    agent.process_multi_modal_info = AsyncMock(return_value={})
    agent._get_mm_processor_kwargs = lambda audios: {}
    response = SimpleNamespace(
        token_ids=[3] * response_length,
        log_probs=[0.0] * response_length,
        extra_fields={"rollout_prompt_ids": [1, 2]},
        num_preempted=0,
    )
    agent.server_manager = SimpleNamespace(generate=AsyncMock(return_value=response))
    monkeypatch.setattr(module, "render_minicpmo_messages", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(module, "prepare_minicpmo_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "split_minicpmo_actor_inputs", lambda inputs: ([1, 2], {}))

    output = await agent.run({}, raw_prompt=[{"role": "user", "content": "prompt"}])

    assert output.extra_fields["response_at_token_limit"] == (response_length == 4)
