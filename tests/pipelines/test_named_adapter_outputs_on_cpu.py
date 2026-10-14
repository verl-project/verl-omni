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
"""Real postprocessors, request batching and step-decode paths share one artifact contract."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.diffusion_rollout_output import quantize_pixels, rollout_output, with_visual_artifacts
from verl_omni.pipelines.request_batch import requested_outputs_for_batch, split_diffusion_output_by_request
from verl_omni.pipelines.rollout_media import (
    ArtifactContractError,
    artifact_fields,
    artifacts_from_fields,
    select_artifact,
)
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


def _format(native, architecture="QwenImagePipeline", request_id="request"):
    return format_diffusion_outputs(
        request=SimpleNamespace(
            request_id=request_id, prompt={"prompt_ids": [1]}, sampling_params=OmniDiffusionSamplingParams()
        ),
        od_config=SimpleNamespace(model_class_name=architecture),
        diffusion_output=native,
        output_data=native.output,
        postprocess_output=normalize_diffusion_postprocess_output(native.output),
    )[0]


@pytest.mark.parametrize(
    "architecture,kind,latent_layout",
    [
        ("QwenImagePipeline", "image", "LC"),
        ("QwenImageEditPlusPipeline", "image", "LC"),
        ("StableDiffusion3Pipeline", "image", "CHW"),
        ("FluxPipeline", "image", "LC"),
        ("BooguImagePipeline", "image", "CHW"),
        ("WanPipeline", "video", "CTHW"),
        ("LTX2Pipeline", "video", "CTHW"),
    ],
)
@pytest.mark.parametrize("output_type", ["pt", "latent"])
def test_native_postprocessor_and_formatter_preserve_named_streams(
    tmp_path, architecture, kind, latent_layout, output_type
):
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae/config.json").write_text(
        json.dumps({"temporal_downsample": [False, True, True], "block_out_channels": [8, 8, 8, 8]})
    )
    (tmp_path / "vocoder").mkdir()
    (tmp_path / "vocoder/config.json").write_text('{"output_sampling_rate": 48000}')
    from vllm_omni.diffusion.registry import get_diffusion_post_process_func

    processor = get_diffusion_post_process_func(
        SimpleNamespace(model=str(tmp_path), model_class_name=architecture, output_type="image")
    )
    pixels = (
        torch.linspace(-1, 1, 72).reshape(1, 3, 3, 2, 4)
        if kind == "video"
        else torch.linspace(-1, 1, 24).reshape(1, 3, 2, 4)
    )
    latent_shape = {"LC": (1, 2, 16), "CHW": (1, 16, 2, 2), "CTHW": (1, 16, 2, 2, 2)}[latent_layout]
    latents = torch.randn(latent_shape, dtype=torch.bfloat16)
    native = with_visual_artifacts(
        rollout_output(media=None, rl={"sentinel": torch.tensor([13])}),
        decoded=pixels,
        latents=latents,
        modality=kind,
        decoded_layout="CTHW" if kind == "video" else "CHW",
        latent_layout=latent_layout,
        output_type=output_type,
        fps=24 if kind == "video" else None,
        context="adapter request",
    )
    assert processor(native.output) is native.output
    output = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(
        _format(native, architecture), None, {"output_type": output_type}
    )
    assert set(output.artifacts) == {f"{kind}_latent", f"{kind}_preview"}
    assert output.extra_fields["sentinel"] == 13
    torch.testing.assert_close(output.artifacts[f"{kind}_latent"].data, latents[0])
    assert output.artifacts[f"{kind}_preview"].spec.layout == ("TCHW" if kind == "video" else "CHW")


@pytest.mark.parametrize("output_type,request_preview", [("image", False), ("latent", False), ("latent", True)])
def test_flux_forward_emits_named_packed_latents_and_optional_batch_preview(output_type, request_preview):
    from verl_omni.pipelines.flux_dance_grpo.vllm_omni_rollout_adapter import FluxDanceGRPOPipelineWithLogProb
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    pipeline = object.__new__(FluxDanceGRPOPipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.transformer = SimpleNamespace(in_channels=64)
    pipeline.encode_prompt_from_token_ids = MagicMock(
        return_value=(torch.zeros(2, 4, 8), torch.zeros(2, 8), torch.zeros(4, 3))
    )
    pipeline._set_timesteps = MagicMock(return_value=torch.tensor([1000.0, 500.0, 100.0]))
    native = torch.arange(2 * 16 * 4 * 6, dtype=torch.float32).reshape(2, 16, 4, 6)
    packed = pipeline._pack_latents(native, 2, 16, 4, 6)
    trajectory = packed.unsqueeze(1).expand(-1, 3, -1, -1).clone()
    pipeline.diffuse = MagicMock(return_value=(trajectory, trajectory + 1, torch.zeros(2, 3), torch.ones(2, 3), packed))
    pixels = torch.linspace(-1, 1, 2 * 3 * 32 * 48).reshape(2, 3, 32, 48)
    pipeline.vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=2.0, shift_factor=0.5),
        dtype=torch.float32,
        decode=MagicMock(return_value=(pixels,)),
    )
    requests = [
        OmniDiffusionRequest(
            request_id=f"flux-{index}",
            prompt=OmniRolloutRequest.from_generate_kwargs(
                prompt_ids=[index], extra_prompt_ids={"clip": [index], "t5": [index, 2]}
            ).to_diffusion_prompt(),
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=48,
                num_inference_steps=3,
                output_type=output_type,
                seed=index,
                extra_args={
                    "timestep_sample_strategy": "continuous",
                    "drop_last_transition": False,
                    "requested_outputs": ["image_preview"] if request_preview and index == 1 else [],
                },
            ),
        )
        for index in range(2)
    ]
    outputs = pipeline.forward(DiffusionRequestBatch(requests=requests))
    decode = output_type == "image" or request_preview
    assert pipeline.vae.decode.call_count == int(decode)
    if decode:
        torch.testing.assert_close(pipeline.vae.decode.call_args.args[0], native / 2 + 0.5)
    for index, output in enumerate(outputs):
        converted = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(
            _format(output, "FluxPipeline", requests[index].request_id), None, {"output_type": output_type}
        )
        artifact = converted.artifacts["image_latent"]
        assert artifact.spec.layout == "LC"
        assert artifact.data.dtype == packed.dtype
        torch.testing.assert_close(artifact.data, packed[index])
        torch.testing.assert_close(output.trajectory_latents, trajectory[index : index + 1])
        torch.testing.assert_close(converted.extra_fields["all_next_latents"], trajectory[index] + 1)
        assert ("image_preview" in converted.artifacts) == decode
        if decode:
            preview = converted.artifacts["image_preview"]
            assert preview.spec.layout == "CHW"
            expected = quantize_pixels(pixels[index], "minus_one_one", context="test")
            torch.testing.assert_close(preview.data, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quantization_matches_real_vae_image_processor(dtype):
    from diffusers.image_processor import VaeImageProcessor

    raw = torch.linspace(-2, 2, 48).reshape(2, 3, 2, 4).to(dtype)
    expected = VaeImageProcessor().postprocess(raw, output_type="pt").float().mul(255).round().to(torch.uint8)
    torch.testing.assert_close(quantize_pixels(raw, "minus_one_one", context="test"), expected)


@pytest.mark.parametrize("outputs_per_prompt", [1, 2])
def test_request_split_keeps_named_samples_and_training_batch_axes(outputs_per_prompt):
    batch_size = 2 * outputs_per_prompt
    latents = torch.arange(batch_size * 32, dtype=torch.float32).reshape(batch_size, 2, 16)
    trajectory = torch.arange(batch_size * 3 * 32, dtype=torch.float32).reshape(batch_size, 3, 2, 16)
    native = with_visual_artifacts(
        rollout_output(media=None, trajectory_latents=trajectory),
        decoded=torch.zeros(batch_size, 3, 2, 4),
        latents=latents,
        latent_layout="LC",
        output_type="latent",
        context="batch",
    )
    outputs = split_diffusion_output_by_request(
        native, SimpleNamespace(num_reqs=2), num_outputs_per_prompt=outputs_per_prompt
    )
    for index, output in enumerate(outputs):
        rows = output.output["payload"]["image"]
        assert len(rows) == outputs_per_prompt
        start = index * outputs_per_prompt
        torch.testing.assert_close(rows[0]["image_latent"], latents[start])
        torch.testing.assert_close(output.trajectory_latents, trajectory[start : start + outputs_per_prompt])
        assert output.output["metadata"]["media_artifacts"]["specs"]["image_latent"]["layout"] == "LC"


def test_batch_preview_request_is_not_lost_when_only_later_request_needs_it():
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                request_id="latent-only",
                prompt={"prompt_ids": [1]},
                sampling_params=OmniDiffusionSamplingParams(output_type="latent"),
            ),
            OmniDiffusionRequest(
                request_id="needs-preview",
                prompt={"prompt_ids": [2]},
                sampling_params=OmniDiffusionSamplingParams(
                    output_type="latent", extra_args={"requested_outputs": ["image_preview"]}
                ),
            ),
        ]
    )
    assert requested_outputs_for_batch(batch) == ["image_preview"]
    assert batch.requests[0].sampling_params.extra_args.get("requested_outputs") is None


@pytest.mark.parametrize(
    "module,cls_name",
    [
        ("qwen_image_flow_grpo", "QwenImagePipelineWithLogProb"),
        ("qwen_image_diffusion_nft", "QwenImageDiffusionNFTPipeline"),
        ("qwen_image_dpo", "QwenImageDPOPipeline"),
    ],
)
@pytest.mark.parametrize(
    "output_type,requested,decode_type",
    [
        ("pt", [], "pil"),
        ("latent", [], "latent"),
        ("latent", ["image_preview"], "pil"),
    ],
)
def test_real_step_decode_hooks_emit_explicit_latent_and_optional_preview(
    monkeypatch, module, cls_name, output_type, requested, decode_type
):
    from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline

    calls = []
    pixels = torch.ones(1, 3, 2, 4)

    def decode(self, latents, height, width, output_type):
        calls.append(output_type)
        return DiffusionOutput(output=latents if output_type == "latent" else pixels)

    monkeypatch.setattr(QwenImagePipeline, "_decode_latents", decode)
    cls = getattr(importlib.import_module(f"verl_omni.pipelines.{module}.vllm_omni_rollout_adapter"), cls_name)
    pipeline = object.__new__(cls)
    state = StepRequestState(
        request_id="step",
        sampling=OmniDiffusionSamplingParams(
            height=16, width=32, output_type=output_type, extra_args={"requested_outputs": requested}
        ),
    )
    state.latents = torch.ones(1, 2, 16, dtype=torch.bfloat16)
    state.timesteps = torch.tensor([1.0])
    state.extra.update(height=16, width=32)
    state.all_latents = [state.latents]
    state.all_log_probs = [torch.zeros(1)]
    state.all_timesteps = [torch.tensor(1.0)]
    output = pipeline.post_decode(state)
    assert calls == [decode_type]
    header = output.output["metadata"]["media_artifacts"]
    row = output.output["payload"]["image"][0]
    assert header["primary"] == ("image_latent" if output_type == "latent" else "image_preview")
    assert ("image_preview" in row) == (decode_type == "pil")
    torch.testing.assert_close(row["image_latent"], state.latents[0])
    if "rl" in output.output["metadata"]:
        assert "latents_clean" in output.output["metadata"]["rl"]


@pytest.mark.parametrize(
    "module,cls_name",
    [
        ("qwen_image_flow_grpo", "QwenImagePipelineWithLogProb"),
        ("qwen_image_diffusion_nft", "QwenImageDiffusionNFTPipeline"),
        ("qwen_image_dpo", "QwenImageDPOPipeline"),
    ],
)
def test_step_warmup_media_is_discarded_but_real_nonfinite_pixels_fail(monkeypatch, module, cls_name):
    from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
    from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID

    calls = []

    def decode(self, *args):
        calls.append(args)
        return DiffusionOutput(output=torch.full((1, 3, 2, 4), float("nan")))

    monkeypatch.setattr(QwenImagePipeline, "_decode_latents", decode)
    cls = getattr(importlib.import_module(f"verl_omni.pipelines.{module}.vllm_omni_rollout_adapter"), cls_name)
    pipeline = object.__new__(cls)
    state = StepRequestState(
        request_id=DUMMY_DIFFUSION_REQUEST_ID, sampling=OmniDiffusionSamplingParams(height=16, width=32)
    )
    state.latents = torch.ones(1, 2, 16)
    state.timesteps = torch.tensor([1.0])
    state.extra.update(height=16, width=32)
    state.all_latents = [state.latents]
    state.all_log_probs = [torch.zeros(1)]
    state.all_timesteps = [torch.tensor(1.0)]
    output = pipeline.post_decode(state)
    assert output.output is None and len(calls) == 1
    state.request_id = "real-request"
    with pytest.raises(ValueError, match="real-request.*nonfinite decoded pixels"):
        pipeline.post_decode(state)


@pytest.mark.parametrize("conflict", ["values", "dtype"])
def test_complementary_engine_sources_are_merged_and_conflicts_fail(conflict):
    native = with_visual_artifacts(
        rollout_output(media=None),
        decoded=torch.ones(1, 3, 2, 4),
        latents=torch.ones(1, 2, 16),
        latent_layout="LC",
        output_type="pt",
        context="adapter",
    )
    final = _format(native, request_id="merge-test")
    row = final.images[0]
    final.multimodal_output["image_latent"] = row["image_latent"]
    final.images = [row["image_preview"]]
    strategy = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None))
    output = strategy.process_output(final, None, {})
    assert set(output.artifacts) == {"image_latent", "image_preview"}
    final.images = [row]
    final.multimodal_output["image_latent"] = (
        row["image_latent"] + 1 if conflict == "values" else row["image_latent"].half()
    )
    with pytest.raises(ValueError, match="merge-test.*conflicting artifact='image_latent'"):
        strategy.process_output(final, None, {})


@pytest.mark.parametrize("requested", ["image_preview", ["image_preview", "image_preview"], ["missing_artifact"]])
def test_request_artifact_selection_fails_before_engine_generation(monkeypatch, requested):
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    engine = SimpleNamespace(
        default_sampling_params_list=[None],
        engine=SimpleNamespace(get_stage_metadata=lambda index: SimpleNamespace(stage_type="diffusion")),
    )
    strategy = DiffusionStrategy(SimpleNamespace(engine=engine))
    monkeypatch.setattr(strategy, "_diffusion_io_spec", lambda: QwenImagePipelineWithLogProb.diffusion_io_spec)
    with pytest.raises(ValueError, match="requested_outputs|undeclared artifacts"):
        strategy.preprocess_input(
            OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1]), {"requested_outputs": requested}, None
        )


def test_batch_split_never_infers_static_metadata_batch_axes():
    native = with_visual_artifacts(
        rollout_output(media=None),
        decoded=torch.zeros(2, 3, 2, 4),
        latents=torch.ones(2, 2, 16),
        latent_layout="LC",
        output_type="pt",
        context="batch",
    )
    shared = torch.arange(2)  # Coincidentally the same length as the request batch.
    native.output["metadata"]["shared_schedule"] = shared
    split = split_diffusion_output_by_request(native, SimpleNamespace(num_reqs=2), num_outputs_per_prompt=1)
    for item in split:
        torch.testing.assert_close(item.output["metadata"]["shared_schedule"], shared)
        assert item.output["metadata"]["shared_schedule"].data_ptr() == shared.data_ptr()
    native.trajectory_latents = torch.zeros(3, 4, 2, 16)
    with pytest.raises(ValueError, match="Expected rollout batch size 2"):
        split_diffusion_output_by_request(native, SimpleNamespace(num_reqs=2), num_outputs_per_prompt=1)


@pytest.mark.parametrize("field", ["trajectory", "rl"])
def test_unbatch_never_silently_discards_training_rows(field):
    native = with_visual_artifacts(
        rollout_output(media=None),
        decoded=torch.zeros(1, 3, 2, 4),
        latents=torch.ones(1, 2, 16),
        latent_layout="LC",
        output_type="pt",
        context="batch",
    )
    final = _format(native, request_id="bad-meta")
    if field == "trajectory":
        final.trajectory_latents = torch.zeros(3, 2, 16)
    else:
        final.multimodal_output["metadata"]["rl"] = {"latents_clean": torch.zeros(3, 2, 16)}
    with pytest.raises(ValueError, match="bad-meta.*expected batch size 1"):
        DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(final, None, {})


@pytest.mark.parametrize("output_type", ["unknown", "", False])
def test_unknown_representation_is_not_treated_as_pixels(output_type):
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    engine = SimpleNamespace(
        default_sampling_params_list=[None],
        engine=SimpleNamespace(get_stage_metadata=lambda index: SimpleNamespace(stage_type="diffusion")),
    )
    with pytest.raises(ValueError, match="Unsupported diffusion output_type"):
        DiffusionStrategy(SimpleNamespace(engine=engine)).preprocess_input(
            OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1]),
            {"output_type": output_type},
            None,
        )


def test_ltx_declares_single_request_state_ownership():
    from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb

    assert LTX23PipelineWithLogProb.supports_request_batch is False


def test_error_provenance_survives_tensor_transport():
    native = with_visual_artifacts(
        rollout_output(media=None),
        decoded=None,
        latents=torch.ones(1, 2, 16),
        latent_layout="LC",
        output_type="latent",
        context="adapter",
    )
    output = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(
        _format(native, request_id="trace-me"), None, {"output_type": "latent"}
    )
    fields = artifact_fields(output.artifacts, output.primary_artifact, output.preview_artifact)
    restored = artifacts_from_fields(fields, context="reward")
    with pytest.raises(ArtifactContractError, match="trace-me.*image_preview"):
        select_artifact(restored, name="image_preview", modality="image", representation="decoded")


@pytest.mark.parametrize("output_type,requested", [("pil", []), ("latent", []), ("latent", ["image_preview"])])
def test_bagel_forward_emits_native_packed_latents_and_canonical_preview(monkeypatch, output_type, requested):
    from PIL import Image
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline

    from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import (
        BagelPipelineWithLogProb,
        _BagelSchedulerAdapter,
    )
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    pipeline = object.__new__(BagelPipelineWithLogProb)
    pipeline.scheduler = _BagelSchedulerAdapter(FlowMatchSDEDiscreteScheduler())
    latent = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    trajectory = torch.stack([latent, latent + 1, latent + 2])
    pixels = torch.arange(24, dtype=torch.uint8).reshape(2, 4, 3)
    image = Image.fromarray(pixels.numpy())

    monkeypatch.setattr(BagelPipeline, "_decode_image_from_latent", lambda self, *args: image)

    def forward(self, request):
        decoded = self._decode_image_from_latent(None, None, latent, (2, 4))
        return DiffusionOutput(
            output={
                "payload": {
                    "image": decoded,
                    "trajectory": {
                        "latents": trajectory,
                        "timesteps": torch.tensor([1.0, 0.5]),
                        "log_probs": torch.tensor([0.2, 0.3]),
                        "decoded": [image],
                    },
                },
                "metadata": {"sentinel": 13},
            }
        )

    monkeypatch.setattr(BagelPipeline, "forward", forward)
    request = SimpleNamespace(
        request_id="bagel-named",
        prompts=[{"prompt": "image prompt"}],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=3, output_type=output_type, extra_args={"requested_outputs": requested}
        ),
    )
    output = pipeline.forward(request)
    header = output.output["metadata"]["media_artifacts"]
    row = output.output["payload"]["image"][0]
    assert header["primary"] == ("image_latent" if output_type == "latent" else "image_preview")
    assert header["specs"]["image_latent"]["layout"] == "LC"
    assert header["specs"]["image_preview"]["layout"] == "CHW"
    assert row["image_latent"].dtype == torch.bfloat16
    assert row["image_preview"].dtype == torch.uint8
    torch.testing.assert_close(row["image_latent"], latent)
    torch.testing.assert_close(row["image_preview"], pixels.permute(2, 0, 1))
    torch.testing.assert_close(output.trajectory_latents, trajectory.unsqueeze(0))
    torch.testing.assert_close(output.trajectory_timesteps, torch.tensor([[1.0, 0.5]]))
    torch.testing.assert_close(output.trajectory_log_probs, torch.tensor([[0.2, 0.3]]))
    assert output.trajectory_decoded == [image]
    assert output.output["metadata"]["sentinel"] == 13
    request.sampling_params.extra_args["requested_outputs"] = ["missing"]
    with pytest.raises(ValueError, match="bagel-named.*requested artifacts absent.*missing"):
        pipeline.forward(request)


def _wan_forward_preview(monkeypatch, sampling_kwargs, interpolation):
    import verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter as adapter

    pipeline = object.__new__(adapter.Wan22DanceGRPOPipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2), in_channels=4)
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae_scale_factor_temporal = 4
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.transformer_2 = None
    pipeline.boundary_ratio = None
    pipeline.expand_timesteps = False
    pipeline.scheduler = SimpleNamespace(
        set_timesteps=MagicMock(),
        timesteps=torch.tensor([900.0, 500.0]),
        config=SimpleNamespace(num_train_timesteps=1000),
    )
    pipeline.encode_prompt = MagicMock(return_value=(torch.zeros(1, 4, 8), torch.ones(1, 4)))
    latent = torch.zeros(1, 4, 2, 4, 6)
    trajectory = latent.unsqueeze(1).expand(-1, 2, -1, -1, -1, -1)
    pipeline.prepare_latents = MagicMock(return_value=latent)
    pipeline.diffuse = MagicMock(return_value=(latent, trajectory, torch.zeros(1, 2), torch.ones(1, 2)))
    pixels = torch.zeros(1, 3, 5, 8, 12)
    pipeline.vae = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(latents_mean=[0] * 4, latents_std=[1] * 4, z_dim=4),
        decode=MagicMock(return_value=(pixels,)),
    )
    interpolate = MagicMock(return_value=(pixels.repeat_interleave(2, dim=2), 2))
    monkeypatch.setattr(adapter.pipeline_wan2_2, "interpolate_video_tensor", interpolate)
    monkeypatch.setattr(adapter, "current_omni_platform", SimpleNamespace(is_available=lambda: False))
    sampling = OmniDiffusionSamplingParams(
        height=64,
        width=96,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        output_type="pt",
        enable_frame_interpolation=interpolation,
        **sampling_kwargs,
    )
    request = OmniDiffusionRequest(request_id="wan-fps", prompt={"prompt_ids": [1, 2]}, sampling_params=sampling)
    output = pipeline.forward(DiffusionRequestBatch(requests=[request]), init_same_noise=False)
    assert interpolate.call_count == int(interpolation)
    torch.testing.assert_close(output.trajectory_latents, trajectory)
    return output


def _ltx_forward_preview(monkeypatch, sampling_kwargs, output_type="pt", requested=None):
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

    from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb

    pipeline = object.__new__(LTX23PipelineWithLogProb)
    pipeline._configure_flow_grpo = MagicMock()
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
    pipeline._flow_grpo_trajectory = {"all_latents": trajectory}

    decode_types = []

    def decode(self, *, latents, audio_latents, output_type, **kwargs):
        decode_types.append(output_type)
        if output_type == "latent":
            return DiffusionOutput(output=(latents, audio_latents))
        return DiffusionOutput(output=(torch.zeros(1, 5, 3, 8, 12), torch.ones(1, 2, 320)))

    def forward(self, request, **kwargs):
        return self._decode_output(latents=latent, audio_latents=audio_latent, output_type=output_type)

    monkeypatch.setattr(LTX2Pipeline, "_decode_output", decode)
    monkeypatch.setattr(LTX2Pipeline, "forward", forward)
    request = OmniDiffusionRequest(
        request_id="ltx-fps",
        prompt={"prompt_ids": [1, 2]},
        sampling_params=OmniDiffusionSamplingParams(
            output_type=output_type, extra_args={"requested_outputs": requested or []}, **sampling_kwargs
        ),
    )
    output = pipeline.forward(DiffusionRequestBatch(requests=[request]))
    expected_decode = output_type != "latent" or bool(set(requested or ()) & {"video_preview", "audio"})
    assert decode_types == ["pt" if expected_decode else "latent"]
    torch.testing.assert_close(output.trajectory_latents, trajectory)
    return output


@pytest.mark.parametrize("adapter", ["wan", "ltx"])
@pytest.mark.parametrize(
    "sampling_kwargs,expected",
    [
        ({"fps": 30}, 30),
        ({"fps": 30, "frame_rate": 23.976}, 23.976),
        ({"fps": 29.97}, 29.97),
        ({"fps": None, "frame_rate": None}, None),
    ],
)
def test_video_forward_emits_resolved_frame_rate(monkeypatch, adapter, sampling_kwargs, expected):
    forward = _wan_forward_preview if adapter == "wan" else _ltx_forward_preview
    args = (monkeypatch, sampling_kwargs, False) if adapter == "wan" else (monkeypatch, sampling_kwargs)
    if expected is None:
        with pytest.raises(ArtifactContractError, match="decoded video requires explicit fps"):
            forward(*args)
        return
    output = forward(*args)
    actual = output.output["metadata"]["media_artifacts"]["specs"]["video_preview"]["fps"]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "sampling_kwargs,expected",
    [
        ({"fps": 30}, 60),
        ({"fps": 30, "frame_rate": 23.976}, 47.952),
        ({"fps": 29.97}, 59.94),
        ({"fps": None, "frame_rate": None}, None),
    ],
)
def test_wan_interpolation_scales_resolved_frame_rate_once(monkeypatch, sampling_kwargs, expected):
    if expected is None:
        with pytest.raises(ArtifactContractError, match="decoded video requires explicit fps"):
            _wan_forward_preview(monkeypatch, sampling_kwargs, True)
        return
    output = _wan_forward_preview(monkeypatch, sampling_kwargs, True)
    header = output.output["metadata"]["media_artifacts"]
    actual = header["specs"]["video_preview"]["fps"]
    assert actual == pytest.approx(expected)
    assert output.output["payload"]["video"][0]["video_preview"].shape[0] == 10


@pytest.mark.parametrize("requested,decode", [([], False), (["audio"], True), (["video_preview"], True)])
def test_ltx_latent_forward_preserves_independent_audio_decode(monkeypatch, requested, decode):
    output = _ltx_forward_preview(monkeypatch, {"fps": 29.97}, "latent", requested)
    header = output.output["metadata"]["media_artifacts"]
    assert header["primary"] == "video_latent"
    assert ("audio" in header["specs"]) == decode
    assert ("video_preview" in header["specs"]) == decode


@pytest.mark.parametrize("adapter", ["qwen", "sd3"])
@pytest.mark.parametrize("request_preview", [False, True])
def test_image_forward_keeps_preview_request_union_from_later_request(adapter, request_preview):
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

    cls = QwenImagePipelineWithLogProb if adapter == "qwen" else StableDiffusion3PipelineWithLogProb
    pipeline = object.__new__(cls)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 4
    pipeline.transformer = SimpleNamespace(in_channels=16, guidance_embeds=False)
    latents = torch.arange(64, dtype=torch.float32).reshape(2, 2, 16) if adapter == "qwen" else torch.ones(2, 4, 2, 4)
    trajectory = latents.unsqueeze(1).expand(-1, 3, *([-1] * (latents.ndim - 1)))
    pipeline.encode_prompt = MagicMock(
        return_value=(
            torch.zeros(2, 4, 8),
            torch.ones(2, 4, dtype=torch.long) if adapter == "qwen" else torch.zeros(2, 8),
        )
    )
    pipeline.prepare_latents = MagicMock(return_value=latents)
    pipeline.prepare_timesteps = MagicMock(return_value=(torch.tensor([900.0, 500.0, 100.0]), 3))
    pipeline.diffuse = MagicMock(return_value=(latents, trajectory, torch.zeros(2, 2), torch.ones(2, 2)))
    pixels = torch.zeros(2, 3, 2, 4)
    pipeline.vae = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(
            latents_mean=[0] * 4, latents_std=[1] * 4, z_dim=4, scaling_factor=2.0, shift_factor=0.5
        ),
        decode=MagicMock(return_value=(pixels.unsqueeze(2) if adapter == "qwen" else pixels,)),
    )
    if adapter == "qwen":
        pipeline._unpack_latents = MagicMock(return_value=torch.ones(2, 4, 1, 2, 4))
    requests = [
        OmniDiffusionRequest(
            request_id=f"{adapter}-{index}",
            prompt={"prompt_ids": [index, 2]} if adapter == "qwen" else {"prompt": f"image {index}"},
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=32,
                num_inference_steps=3,
                output_type="latent",
                extra_args={"requested_outputs": ["image_preview"] if request_preview and index == 1 else []},
            ),
        )
        for index in range(2)
    ]
    outputs = pipeline.forward(DiffusionRequestBatch(requests=requests))
    assert pipeline.vae.decode.call_count == int(request_preview)
    for index, output in enumerate(outputs):
        header = output.output["metadata"]["media_artifacts"]
        row = output.output["payload"]["image"][0]
        assert header["primary"] == "image_latent"
        assert ("image_preview" in row) == request_preview
        torch.testing.assert_close(row["image_latent"], latents[index])
        torch.testing.assert_close(output.trajectory_latents, trajectory[index : index + 1])
