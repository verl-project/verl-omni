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
"""The LTX-2.3 DiffusionART rollout must keep the base adapter's named media contract and only change the sampler.

Necessity: the audio-video scorers select ``video_preview`` (fps), ``audio`` (sample rate) and the latents by name
(``docs/advanced/diffusion_media_artifacts.md``); a subclass that re-wrapped the output or re-introduced the legacy
``fps`` / ``audio`` metadata would break reward selection silently. The ART subclass therefore overrides only the
FlowGRPO configuration (ODE, all transitions, no log-probs) and must inherit ``forward`` unchanged.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("vllm_omni")

from vllm_omni.diffusion.data import DiffusionOutput  # noqa: E402
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline  # noqa: E402
from vllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: E402
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402

from verl_omni.pipelines.ltx2_diffusion_art.vllm_omni_rollout_adapter import LTX23DiffusionARTPipeline  # noqa: E402
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb  # noqa: E402
from verl_omni.pipelines.model_base import VllmOmniPipelineBase  # noqa: E402


def test_art_pipeline_changes_only_the_sampler_configuration():
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "diffusion_art") is LTX23DiffusionARTPipeline
    assert issubclass(LTX23DiffusionARTPipeline, LTX23PipelineWithLogProb)
    assert LTX23DiffusionARTPipeline.supports_request_batch is False
    # Output construction (named artifacts, trajectory, prompt embeddings) is the base adapter's; the ART class
    # must not re-wrap it.
    for name in ("forward", "run_phase", "_decode_output", "diffusion_io_spec"):
        assert name not in LTX23DiffusionARTPipeline.__dict__, name
    assert set(LTX23DiffusionARTPipeline.diffusion_io_spec.artifacts) == {
        "video_preview",
        "audio",
        "video_latent",
        "audio_latent",
    }


def test_configure_flow_grpo_sets_a_pure_ode_with_all_transitions_and_per_modality_cfg():
    pipeline = object.__new__(LTX23DiffusionARTPipeline)
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(output_type="pt", extra_args={"video_cfg_scale": 1.5}, guidance_scale=3.0)
    )
    pipeline._configure_flow_grpo(request)
    assert pipeline._flow_grpo_noise_level == 0.0
    assert pipeline._flow_grpo_logprobs is False
    assert pipeline._flow_grpo_window_size is None
    assert request.sampling_params.extra_args["video_cfg_scale"] == 1.5
    assert request.sampling_params.extra_args["audio_cfg_scale"] == 3.0  # falls back to guidance_scale
    assert pipeline._select_sde_steps(20, torch.device("cpu")) == list(range(20))
    with pytest.raises(NotImplementedError, match="text-to-audio-video"):
        pipeline._configure_flow_grpo(
            SimpleNamespace(sampling_params=SimpleNamespace(output_type="pt", extra_args={"task": "ti2va"}))
        )


@pytest.mark.parametrize(
    "output_type,requested", [("pt", None), ("latent", ["video_preview", "audio"]), ("latent", None)]
)
def test_art_forward_inherits_the_named_audio_video_artifacts(monkeypatch, output_type, requested):
    """Same mocked forward as the base adapter's named-output test: the ART subclass must emit the same contract."""
    pipeline = object.__new__(LTX23DiffusionARTPipeline)
    pipeline._inject_precomputed_prompt_embeds = MagicMock()
    pipeline.distributed_video_decode = False
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
    pipeline._flow_grpo_prompt_context = SimpleNamespace(
        **{
            f"{polarity}_connector_{name}": torch.ones(1, 2, 3)
            for polarity in ("positive", "negative")
            for name in ("prompt_embeds", "audio_prompt_embeds", "attention_mask")
        }
    )
    latent = torch.ones(1, 4, 2, 4, 6)
    audio_latent = torch.ones(1, 2, 3, 4)
    trajectory = torch.ones(1, 2, 4, 8)
    pipeline._flow_grpo_trajectory = {"all_latents": trajectory, "video_seq_len": torch.tensor([3])}

    def decode(self, *, latents, audio_latents, output_type, **kwargs):
        if output_type == "latent":
            return DiffusionOutput(output=(latents, audio_latents))
        return DiffusionOutput(output=(torch.zeros(1, 5, 3, 8, 12), torch.ones(1, 2, 320)))

    def forward(self, request, **kwargs):
        return self._decode_output(latents=latent, audio_latents=audio_latent, output_type=output_type)

    monkeypatch.setattr(LTX2Pipeline, "_decode_output", decode)
    monkeypatch.setattr(LTX2Pipeline, "forward", forward)
    extra_args = {"requested_outputs": requested} if requested is not None else {}
    request = OmniDiffusionRequest(
        request_id="ltx-art",
        prompt={"prompt_ids": [1, 2]},
        sampling_params=OmniDiffusionSamplingParams(output_type=output_type, extra_args=extra_args, fps=24),
    )
    output = pipeline.forward(DiffusionRequestBatch(requests=[request]))

    # ODE configuration happened on the real request object.
    assert pipeline._flow_grpo_noise_level == 0.0 and request.sampling_params.extra_args["video_cfg_scale"] == 1.0
    torch.testing.assert_close(output.trajectory_latents, trajectory)
    header = output.output["metadata"]["media_artifacts"]
    decoded = output_type != "latent" or bool(requested)
    assert header["primary"] == ("video_latent" if output_type == "latent" else "video_preview")
    assert set(header["specs"]) >= {"video_latent", "audio_latent"}
    assert ("video_preview" in header["specs"]) == decoded and ("audio" in header["specs"]) == decoded
    if decoded:
        assert header["specs"]["video_preview"]["fps"] == pytest.approx(24.0)
        assert header["specs"]["audio"]["sample_rate"] == 48000
        assert header["preview"] == "video_preview" and header["audio"] == "audio"
    # The ART subclass adds nothing to the training payload: no legacy ``fps`` scorer key (scorers select the
    # ``video_preview`` artifact and compare its fps with the configured one), only the base adapter's fields.
    rl = output.output["metadata"]["rl"]
    assert "fps" not in rl
    assert rl["video_seq_len"].tolist() == [3]
