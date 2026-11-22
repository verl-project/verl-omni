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
"""CPU parity tests for SD3 token-id-native prompt encoding.

Necessity: the SD3 FlowGRPO rollout adapter consumes per-text-encoder token ids
produced once by the agent loop instead of decoding ids back to text inside the
pipeline. These tests verify that ``SD3TokenIdPromptMixin.encode_prompt_from_token_ids``
reproduces the vLLM-Omni text-based ``encode_prompt`` exactly, including
per-encoder padding, truncation, and CLIP/T5 embedding concatenation.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "special_e2e"))

from build_sd3_tiny_random import get_dummy_sd3_components  # noqa: E402

from verl_omni.pipelines.sd3_flow_grpo.common import (  # noqa: E402
    SD3TokenIdPromptMixin,
    pad_token_id_batch,
)

_vllm_omni_sd3 = pytest.importorskip(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3",
    reason="vllm-omni is required for the text-path reference implementation",
)

T5_MAX_SEQUENCE_LENGTH = 16


class _ParityPipeline(SD3TokenIdPromptMixin):
    """Tiny SD3 components + the real vLLM-Omni text encode path as reference."""

    # Borrow the unbound text-path methods so the reference cannot drift from
    # the actual rollout pipeline implementation.
    encode_prompt = _vllm_omni_sd3.StableDiffusion3Pipeline.encode_prompt
    _get_clip_prompt_embeds = _vllm_omni_sd3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    _get_t5_prompt_embeds = _vllm_omni_sd3.StableDiffusion3Pipeline._get_t5_prompt_embeds

    def __init__(self):
        components = get_dummy_sd3_components()
        self.tokenizer = components["tokenizer"]
        self.tokenizer_2 = components["tokenizer_2"]
        self.tokenizer_3 = components["tokenizer_3"]
        # Use a non-default pad token on CLIP-G so per-encoder padding differs from CLIP-L.
        self.tokenizer_2.pad_token = "<|startoftext|>"
        self.text_encoder = components["text_encoder"].eval()
        self.text_encoder_2 = components["text_encoder_2"].eval()
        self.text_encoder_3 = components["text_encoder_3"].eval()
        self.transformer = components["transformer"]
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.device = torch.device("cpu")
        self.od_config = SimpleNamespace(dtype=torch.float32)


def _agent_loop_tokenize(tokenizer, texts: list[str], max_length: int) -> list[list[int]]:
    """Mirror ``DiffusionSingleTurnAgentLoop._tokenize_per_encoder`` for one tokenizer."""
    return [
        tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"] for text in texts
    ]


@pytest.fixture(scope="module")
def pipeline() -> _ParityPipeline:
    return _ParityPipeline()


def _encode_both_paths(pipeline: _ParityPipeline, prompts: list[str], num_images_per_prompt: int = 1):
    clip_ids = _agent_loop_tokenize(pipeline.tokenizer, prompts, pipeline.tokenizer_max_length)
    t5_ids = _agent_loop_tokenize(pipeline.tokenizer_3, prompts, T5_MAX_SEQUENCE_LENGTH)

    with torch.no_grad():
        ref_embeds, ref_pooled = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            max_sequence_length=T5_MAX_SEQUENCE_LENGTH,
            num_images_per_prompt=num_images_per_prompt,
        )
        embeds, pooled = pipeline.encode_prompt_from_token_ids(
            clip_prompt_ids=clip_ids,
            t5_prompt_ids=t5_ids,
            max_sequence_length=T5_MAX_SEQUENCE_LENGTH,
            num_images_per_prompt=num_images_per_prompt,
        )
    return (ref_embeds, ref_pooled), (embeds, pooled)


@pytest.mark.parametrize("num_images_per_prompt", [1, 2])
def test_token_id_encode_matches_text_encode(pipeline, num_images_per_prompt):
    prompts = ["a red circle on a white background", "a blue square"]
    (ref_embeds, ref_pooled), (embeds, pooled) = _encode_both_paths(pipeline, prompts, num_images_per_prompt)

    assert embeds.shape == ref_embeds.shape
    assert embeds.shape[1] == pipeline.tokenizer_max_length + T5_MAX_SEQUENCE_LENGTH
    assert pooled.shape == ref_pooled.shape
    torch.testing.assert_close(embeds, ref_embeds, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pooled, ref_pooled, rtol=0.0, atol=0.0)


def test_empty_prompt_parity(pipeline):
    (ref_embeds, ref_pooled), (embeds, pooled) = _encode_both_paths(pipeline, [""])

    torch.testing.assert_close(embeds, ref_embeds, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pooled, ref_pooled, rtol=0.0, atol=0.0)


def test_long_prompt_truncation_parity(pipeline):
    long_prompt = " ".join(["a red circle on a white background"] * 40)
    (ref_embeds, ref_pooled), (embeds, pooled) = _encode_both_paths(pipeline, [long_prompt])

    torch.testing.assert_close(embeds, ref_embeds, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pooled, ref_pooled, rtol=0.0, atol=0.0)


def test_pad_token_id_batch_pads_and_truncates():
    padded = pad_token_id_batch([[1, 2], [3, 4, 5, 6, 7]], max_length=4, pad_token_id=0, device=torch.device("cpu"))

    assert padded.shape == (2, 4)
    assert padded.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6]]


@pytest.mark.parametrize("pause_at", [1, 3, 5])
@pytest.mark.parametrize("sde_type", ["sde", "cps"])
def test_sd3_sde_snapshot_preserves_training_prefix_and_rng(pause_at, sde_type):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.sd3.step_state import restore_step_state, step_execution_profile
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

    def prepare():
        generator = torch.Generator().manual_seed(17)
        scheduler = FlowMatchSDEDiscreteScheduler()
        scheduler.set_timesteps(6)
        scheduler.set_begin_index(0)
        return StepRequestState(
            request_id="prompt-A:sample-2",
            sampling=OmniDiffusionSamplingParams(seed=17, generator=generator),
            prompt={"prompt": "a sign", "extra_args": {"uid": "prompt-A"}},
            latents=torch.randn(1, 4, 2, 2, generator=generator),
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            extra={
                "sde_windows": [[1, 4]],
                "noise_level": 0.7,
                "sde_type": sde_type,
                "logprobs": True,
                "all_latents": [[]],
                "all_log_probs": [[]],
                "all_timesteps": [[]],
            },
        )

    def advance(state):
        prediction = state.latents * 0.125 + state.current_timestep * 0.0001
        adapter.step_scheduler(state, prediction)

    adapter = object.__new__(StableDiffusion3PipelineWithLogProb)
    torch.nn.Module.__init__(adapter)
    adapter.device = torch.device("cpu")
    adapter.od_config = OmniDiffusionConfig(dtype=torch.float32)
    adapter._interrupt = False
    adapter.predict_noise_maybe_with_cfg = lambda _cfg, _scale, positive, _negative, _normalize: (
        positive["hidden_states"] * 0.125 + positive["timestep"].view(-1, 1, 1, 1) * 0.0001
    )
    reference, interrupted = prepare(), prepare()
    adapter.scheduler = reference.scheduler
    expected = adapter.diffuse(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 4),
        None,
        None,
        reference.latents,
        reference.timesteps,
        False,
        0.0,
        0.7,
        (1, 4),
        sde_type,
        reference.sampling.generator,
        True,
    )
    for _ in range(pause_at):
        advance(interrupted)
    payload = adapter.export_step_state(interrupted, policy_ref="base-A/adapter-v7", max_bytes=100_000)
    with pytest.raises(ValueError, match="scheduler"):
        restore_step_state(
            payload,
            policy_ref="base-A/adapter-v7",
            device=adapter.device,
            max_bytes=100_000,
            execution_profile=step_execution_profile(adapter.od_config, adapter.device),
        )
    with pytest.raises(ValueError, match="policy"):
        adapter.restore_step_state(payload, policy_ref="base-A/adapter-v8", max_bytes=100_000)
    if pause_at == 3 and sde_type == "sde":
        adapter.od_config.dtype = torch.bfloat16
        with pytest.raises(ValueError, match="execution profile"):
            adapter.restore_step_state(payload, policy_ref="base-A/adapter-v7", max_bytes=100_000)
        adapter.od_config.dtype = torch.float32
    restored = adapter.restore_step_state(payload, policy_ref="base-A/adapter-v7", max_bytes=100_000)
    assert restored.request_id == interrupted.request_id and restored.prompt == interrupted.prompt
    for _ in range(pause_at, 6):
        advance(restored)
    assert torch.equal(expected[0], restored.latents)
    assert torch.equal(reference.sampling.generator.get_state(), restored.sampling.generator.get_state())
    for field, expected_tensor in zip(("all_latents", "all_log_probs", "all_timesteps"), expected[1:], strict=True):
        actual = torch.stack([torch.stack(row) for row in restored.extra[field]])
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
        assert torch.equal(actual, expected_tensor)


def test_sd3_sde_snapshot_rejects_process_global_rng():
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

    scheduler = FlowMatchSDEDiscreteScheduler()
    scheduler.set_timesteps(6)
    state = StepRequestState(
        request_id="unseeded-request",
        sampling=OmniDiffusionSamplingParams(),
        latents=torch.ones(1, 4, 2, 2),
        timesteps=scheduler.timesteps,
        scheduler=scheduler,
    )
    adapter = object.__new__(StableDiffusion3PipelineWithLogProb)
    torch.nn.Module.__init__(adapter)
    with pytest.raises(ValueError, match="request-owned generator"):
        adapter.export_step_state(state, policy_ref="base-A/adapter-v7", max_bytes=100_000)


def _make_flowgrpo_step_pipeline():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

    result = object.__new__(StableDiffusion3PipelineWithLogProb)
    torch.nn.Module.__init__(result)
    components = get_dummy_sd3_components()
    for name, value in components.items():
        if name != "scheduler":
            setattr(result, name, value.eval() if isinstance(value, torch.nn.Module) else value)
    result.scheduler = FlowMatchSDEDiscreteScheduler.from_config(components["scheduler"].config)
    result.device = torch.device("cpu")
    result.od_config = OmniDiffusionConfig(dtype=torch.float32)
    result.tokenizer_max_length = result.tokenizer.model_max_length
    result.vae_scale_factor = 8
    result.default_sample_size = 32
    result._interrupt = False
    return result


@pytest.mark.parametrize(
    "guidance, outputs, preview, warmup", [(0.0, 1, False, False), (4.0, 2, True, False), (0.0, 1, True, True)]
)
def test_flowgrpo_encode_step_decode_and_restore_match_full_output(guidance, outputs, preview, warmup):
    from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from verl_omni.pipelines.sd3_flow_grpo.common import SD3_CLIP_TOKENS_KEY, SD3_T5_TOKENS_KEY

    pipeline = _make_flowgrpo_step_pipeline()
    tokens = {
        SD3_CLIP_TOKENS_KEY: _agent_loop_tokenize(pipeline.tokenizer, ["a sign"], 77)[0],
        SD3_T5_TOKENS_KEY: _agent_loop_tokenize(pipeline.tokenizer_3, ["a sign"], T5_MAX_SEQUENCE_LENGTH)[0],
    }
    request = OmniDiffusionRequest(
        request_id=DUMMY_DIFFUSION_REQUEST_ID if warmup else "prompt-A",
        prompt={"prompt": "dummy run"} if warmup else {"extra_args": {"extra_prompt_ids": tokens, "uid": "prompt-A"}},
        sampling_params=OmniDiffusionSamplingParams(
            seed=17,
            height=16,
            width=16,
            num_outputs_per_prompt=outputs,
            num_inference_steps=2 if warmup else 6,
            guidance_scale=guidance,
            max_sequence_length=T5_MAX_SEQUENCE_LENGTH,
            output_type=None if warmup else "latent",
            extra_args={
                **({} if warmup else {"sde_window_size": 3, "sde_window_range": [1, 5]}),
                "requested_outputs": ["image_preview"] if preview else [],
            },
        ),
    )
    with torch.inference_mode():
        expected = pipeline.forward(DiffusionRequestBatch(requests=[copy.deepcopy(request)]))[0]
        state = StepRequestState(request_id=request.request_id, prompt=request.prompt, sampling=request.sampling_params)
        pipeline.prepare_encode(state)
        for _ in range(1 if warmup else 3):
            pipeline.step_scheduler(state, pipeline.denoise_step(InputBatch.make_batch([state])))
        payload = pipeline.export_step_state(state, policy_ref="fixed-policy", max_bytes=1 << 20)
        state = pipeline.restore_step_state(payload, policy_ref="fixed-policy", max_bytes=1 << 20)
        assert state.request_id == request.request_id and state.prompt == request.prompt
        assert state.sampling.seed == 17 and state.sampling.extra_args == request.sampling_params.extra_args
        if outputs == 2:
            assert state.sampling.generator[0] is state.sampling.generator[1]
        while not state.denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(InputBatch.make_batch([state])))
        actual = pipeline.post_decode(state)
    for field in ("trajectory_latents", "trajectory_log_probs", "trajectory_timesteps"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    header = actual.output["metadata"]["media_artifacts"]
    assert header == expected.output["metadata"]["media_artifacts"]
    assert header["primary"] == ("image_preview" if warmup else "image_latent")
    assert header["preview"] == ("image_preview" if preview else None)
    for row, reference in zip(actual.output["payload"]["image"], expected.output["payload"]["image"], strict=True):
        assert row.keys() == reference.keys()
        for name in row:
            torch.testing.assert_close(row[name], reference[name], rtol=0, atol=0)
    embeddings = expected.output["metadata"]["prompt_embeddings"]
    for name, value in actual.output["metadata"]["prompt_embeddings"].items():
        if value is None:
            assert embeddings[name] is None
        else:
            torch.testing.assert_close(value, embeddings[name], rtol=0, atol=0)


def test_sd3_step_execution_is_available_only_for_flowgrpo():
    from vllm_omni.diffusion.models.interface import supports_step_execution

    from verl_omni.pipelines.model_base import VllmOmniPipelineBase
    from verl_omni.pipelines.sd3_dgpo import StableDiffusion3DGPOPipeline
    from verl_omni.pipelines.sd3_flow_grpo import StableDiffusion3PipelineWithLogProb

    flowgrpo = VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "flow_grpo")
    dgpo = VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "dgpo")
    assert flowgrpo is StableDiffusion3PipelineWithLogProb
    assert dgpo is StableDiffusion3DGPOPipeline
    assert supports_step_execution(object.__new__(flowgrpo))
    assert not supports_step_execution(object.__new__(dgpo))
