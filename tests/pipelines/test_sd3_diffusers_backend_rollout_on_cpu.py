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

"""CPU tests for the SD3 FlowGRPO rollout on vLLM-Omni's diffusers backend.

The pipeline samples with the diffusers transformer the trainer optimizes, so
the log-probs it returns must equal what ``StableDiffusion3FlowGRPO``
recomputes from the same trajectory.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import SD3Transformer2DModel
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "special_e2e"))

from build_sd3_tiny_random import ensure_tiny_sd3_checkpoint  # noqa: E402

from verl_omni.pipelines.model_base import VllmOmniDiffusersPipelineBase, VllmOmniPipelineBase  # noqa: E402
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler  # noqa: E402
from verl_omni.pipelines.sd3_flow_grpo import vllm_omni_diffusers_rollout_adapter as adapter_module  # noqa: E402
from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import StableDiffusion3FlowGRPO  # noqa: E402
from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_diffusers_rollout_adapter import (  # noqa: E402
    StableDiffusion3DiffusersPipelineWithLogProb,
)
from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import (  # noqa: E402
    StableDiffusion3PipelineWithLogProb,
)

PROMPTS = ["a red circle", "a blue square"]
SEEDS = [11, 12]
STEPS = 4
SIZE = 64
NOISE_LEVEL = 0.7


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> str:
    return ensure_tiny_sd3_checkpoint(
        str(tmp_path_factory.mktemp("sd3_tiny")), hidden_size=16, seed=0, dtype=torch.float32
    )


@pytest.fixture(scope="module")
def pipeline(model_path) -> StableDiffusion3DiffusersPipelineWithLogProb:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter_module, "get_local_device", lambda: torch.device("cpu"))
        od_config = OmniDiffusionConfig(model=model_path, dtype=torch.float32, diffusion_load_format="diffusers")
        return StableDiffusion3DiffusersPipelineWithLogProb(od_config=od_config)


def _request(pipe, index: int, text: str, seed: int, *, guidance_scale: float | None = None) -> OmniDiffusionRequest:
    def ids(value: str) -> dict[str, list[int]]:
        return {
            "clip": pipe.tokenizer(value, truncation=True, max_length=77)["input_ids"],
            "t5": pipe.tokenizer_3(value, truncation=True, max_length=16)["input_ids"],
        }

    extra = {"extra_prompt_ids": ids(text)}
    if guidance_scale is not None:
        extra["negative_extra_prompt_ids"] = ids("blurry")
    sampling = OmniDiffusionSamplingParams(
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        max_sequence_length=16,
        seed=seed,
        extra_args={"noise_level": NOISE_LEVEL, "sde_window_size": 2, "sde_window_range": [0, 3], "logprobs": True},
    )
    if guidance_scale is not None:
        sampling.guidance_scale = guidance_scale
        sampling.guidance_scale_provided = True
    prompt = {"prompt": text, "extra_args": extra}
    return OmniDiffusionRequest(request_id=f"req-{index}", prompt=prompt, sampling_params=sampling)


def _run(pipe, requests: list[OmniDiffusionRequest]):
    return pipe.forward(DiffusionRequestBatch(requests=requests))


def test_registered_for_the_diffusers_backend_only():
    cls = VllmOmniDiffusersPipelineBase.get_class("StableDiffusion3Pipeline", "flow_grpo")
    assert cls is StableDiffusion3DiffusersPipelineWithLogProb
    assert issubclass(cls, DiffusersAdapterPipeline)
    native = VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "flow_grpo")
    assert native is StableDiffusion3PipelineWithLogProb


def test_pipeline_holds_the_diffusers_modules(pipeline):
    assert isinstance(pipeline.transformer, SD3Transformer2DModel)
    assert pipeline.transformer is pipeline._pipeline.transformer
    assert dict(pipeline.named_children()).keys() >= {"transformer", "vae", "text_encoder", "text_encoder_3"}
    assert isinstance(pipeline.scheduler, FlowMatchSDEDiscreteScheduler)


@pytest.mark.parametrize("guidance_scale", [None, 4.5])
def test_log_probs_match_training_recompute(pipeline, model_path, guidance_scale):
    output = _run(pipeline, [_request(pipeline, 0, PROMPTS[0], SEEDS[0], guidance_scale=guidance_scale)])[0]
    latents = output.trajectory_latents  # (1, W + 1, C, h, w)
    timesteps = output.trajectory_timesteps  # (1, W)
    log_probs = output.trajectory_log_probs[0]  # (W,)
    assert latents.shape == (1, 3, 16, SIZE // pipeline.vae_scale_factor, SIZE // pipeline.vae_scale_factor)
    assert log_probs.shape == (2,)

    module = SD3Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float32)
    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(STEPS)
    model_config = SimpleNamespace(
        pipeline=SimpleNamespace(guidance_scale=guidance_scale),
        algo=SimpleNamespace(noise_level=NOISE_LEVEL, sde_type="sde"),
    )
    embeds = output.output["metadata"]["prompt_embeddings"]

    def inputs(step, prefix=""):
        return StableDiffusion3FlowGRPO.build_transformer_inputs(
            latents=latents[:, step],
            timesteps=timesteps[:, step],
            prompt_embeds=embeds[f"{prefix}prompt_embeds"],
            pooled_prompt_embeds=embeds[f"{prefix}pooled_prompt_embeds"],
        )

    recomputed = []
    with torch.no_grad():
        for step in range(timesteps.shape[1]):
            log_prob, *_ = StableDiffusion3FlowGRPO.forward_and_sample_previous_step(
                module,
                scheduler,
                model_config,
                inputs(step),
                inputs(step, "negative_") if guidance_scale is not None else None,
                {"all_latents": latents, "all_timesteps": timesteps},
                step,
            )
            recomputed.append(log_prob[0])
    torch.testing.assert_close(torch.stack(recomputed), log_probs, rtol=0.0, atol=1e-5)


def test_batch_matches_per_request(pipeline):
    # A request keeps the generator its seed creates, so each run gets fresh requests.
    cases = list(enumerate(zip(PROMPTS, SEEDS, strict=True)))
    batched = _run(pipeline, [_request(pipeline, i, text, seed) for i, (text, seed) in cases])
    single = [_run(pipeline, [_request(pipeline, i, text, seed)])[0] for i, (text, seed) in cases]
    for b, s in zip(batched, single, strict=True):
        torch.testing.assert_close(b.trajectory_timesteps, s.trajectory_timesteps, rtol=0.0, atol=0.0)
        torch.testing.assert_close(b.trajectory_latents, s.trajectory_latents, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b.trajectory_log_probs, s.trajectory_log_probs, rtol=1e-5, atol=1e-4)
    assert not torch.equal(batched[0].trajectory_latents, batched[1].trajectory_latents)


def test_load_weights_copies_synced_tensors(pipeline):
    state = {name: tensor.clone() for name, tensor in pipeline.transformer.state_dict().items()}

    def trajectory():
        return _run(pipeline, [_request(pipeline, 0, PROMPTS[0], SEEDS[0])])[0].trajectory_latents

    before = trajectory()
    try:
        pipeline.load_weights((f"transformer.{name}", tensor + 0.01) for name, tensor in state.items())
        for name, tensor in pipeline.transformer.state_dict().items():
            torch.testing.assert_close(tensor, state[name] + 0.01)
        assert not torch.allclose(trajectory(), before)
    finally:
        pipeline.load_weights((f"transformer.{name}", tensor) for name, tensor in state.items())
    torch.testing.assert_close(trajectory(), before, rtol=0.0, atol=0.0)


def test_load_weights_rejects_unknown_or_misshaped_tensors(pipeline):
    name, tensor = next(iter(pipeline.transformer.state_dict().items()))
    before = tensor.clone()
    with pytest.raises(KeyError, match="no matching transformer parameter"):
        pipeline.load_weights([(name, tensor)])
    with pytest.raises(KeyError, match="no matching transformer parameter"):
        pipeline.load_weights([(f"transformer.{name}", tensor + 1), ("transformer.not_a_param.weight", tensor)])
    with pytest.raises(ValueError, match="expected"):
        pipeline.load_weights([(f"transformer.{name}", tensor + 1), (f"transformer.{name}", tensor.flatten()[:1])])
    torch.testing.assert_close(pipeline.transformer.state_dict()[name], before, rtol=0.0, atol=0.0)


def test_text_prompt_requests_still_generate(pipeline):
    request = OmniDiffusionRequest(
        request_id="warmup",
        prompt="a red circle",
        sampling_params=OmniDiffusionSamplingParams(height=SIZE, width=SIZE, num_inference_steps=2, seed=0),
    )
    output = _run(pipeline, [request])[0]
    assert output.trajectory_latents is not None
    assert output.output["payload"]["image"] is not None
