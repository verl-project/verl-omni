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
from verl_omni.pipelines.rollout_artifacts import (
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
    "module,factory,architecture,kind,latent_layout",
    [
        ("qwen_image_flow_grpo", "get_rollout_post_process_func", "QwenImagePipeline", "image", "LC"),
        ("qwen_image_edit_flow_grpo", "get_rollout_post_process_func", "QwenImageEditPlusPipeline", "image", "LC"),
        ("sd3_flow_grpo", "get_latent_post_process_func", "StableDiffusion3Pipeline", "image", "CHW"),
        ("flux_dance_grpo", "get_rollout_post_process_func", "FluxPipeline", "image", "LC"),
        ("boogu_image_flow_grpo", "get_rollout_post_process_func", "BooguImagePipeline", "image", "CHW"),
        ("wan22_dance_grpo", "get_rollout_post_process_func", "WanPipeline", "video", "CTHW"),
        ("ltx2_flow_grpo", "get_rollout_post_process_func", "LTX2Pipeline", "video", "CTHW"),
    ],
)
@pytest.mark.parametrize("output_type", ["pt", "latent"])
def test_native_postprocessor_and_formatter_preserve_named_streams(
    tmp_path, module, factory, architecture, kind, latent_layout, output_type
):
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae/config.json").write_text(
        json.dumps({"temporal_downsample": [False, True, True], "block_out_channels": [8, 8, 8, 8]})
    )
    (tmp_path / "vocoder").mkdir()
    (tmp_path / "vocoder/config.json").write_text('{"output_sampling_rate": 48000}')
    adapter = importlib.import_module(f"verl_omni.pipelines.{module}.vllm_omni_rollout_adapter")
    processor = getattr(adapter, factory)(SimpleNamespace(model=str(tmp_path), output_type="image"))
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
    output = DiffusionStrategy(SimpleNamespace(global_steps=1)).process_output(
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
        converted = DiffusionStrategy(SimpleNamespace(global_steps=1)).process_output(
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
    strategy = DiffusionStrategy(SimpleNamespace(global_steps=1))
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
        DiffusionStrategy(SimpleNamespace(global_steps=1)).process_output(final, None, {})


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
    output = DiffusionStrategy(SimpleNamespace(global_steps=1)).process_output(
        _format(native, request_id="trace-me"), None, {"output_type": "latent"}
    )
    fields = artifact_fields(output.artifacts, output.primary_artifact, output.preview_artifact)
    restored = artifacts_from_fields(fields, context="reward")
    with pytest.raises(ArtifactContractError, match="trace-me.*image_preview"):
        select_artifact(restored, name="image_preview", modality="image", representation="decoded")
