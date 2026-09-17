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
"""Named media is shape-declared, lossless and independent of tuple position."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.diffusion_rollout_output import with_media_artifacts, wrap_rollout_postprocessor
from verl_omni.pipelines.rollout_artifacts import (
    MediaArtifact,
    artifact_fields,
    artifacts_from_fields,
    select_artifact,
    validate_artifacts,
)
from verl_omni.pipelines.rollout_media import MediaSpec
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


@pytest.mark.parametrize("layout", ["TCHW", "CTHW", "THWC"])
@pytest.mark.parametrize("frames,channels", [(3, 3), (3, 4), (4, 1)])
def test_decoded_video_axes_do_not_depend_on_channel_count(layout, frames, channels):
    canonical = torch.arange(frames * channels * 2 * 4, dtype=torch.uint8).reshape(frames, channels, 2, 4)
    raw = canonical.permute(*("TCHW".index(axis) for axis in layout))
    result = MediaArtifact(MediaSpec("video", "decoded", layout, fps=24), raw).normalized(
        context="test", name="preview"
    )
    assert result.spec.layout == "TCHW"
    torch.testing.assert_close(result.data, canonical)


@pytest.mark.parametrize("layout,shape", [("CTHW", (16, 3, 2, 2)), ("LC", (13, 64)), ("CLT", (2, 32, 8))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_latents_keep_dtype_axes_and_storage(layout, shape, dtype):
    data = torch.randn(shape).to(dtype)
    artifact = MediaArtifact(MediaSpec("video", "latent", layout), data)
    assert artifact.normalized(context="native", name="latent") is artifact
    assert artifact.data.data_ptr() == data.data_ptr()


@pytest.mark.parametrize(
    "spec,data,error",
    [
        (MediaSpec("video", "decoded", "TCHW"), torch.zeros(3, 3, 2, 2), "uint8"),
        (MediaSpec("video", "latent", "CTHW"), torch.zeros(3, 3, 2, 2, dtype=torch.uint8), "floating latent"),
        (MediaSpec("audio", "decoded", "CT"), torch.zeros(2, 8), "sample_rate"),
        (MediaSpec("audio", "decoded", "CT", sample_rate=0), torch.zeros(2, 8), "sample_rate"),
        (MediaSpec("audio", "decoded", "CT", sample_rate=True), torch.zeros(2, 8), "sample_rate"),
        (MediaSpec("video", "decoded", "CHW"), torch.zeros(3, 2, 2, dtype=torch.uint8), "layout"),
        (MediaSpec("video", "latent", "CTHW"), torch.zeros(3, 2, 2), "shape"),
        (MediaSpec("image", "decoded", "CHW", fps=24), torch.zeros(3, 2, 2, dtype=torch.uint8), "fps"),
        (MediaSpec("depth", "decoded", "CHW"), torch.zeros(3, 2, 2, dtype=torch.uint8), "modality"),
    ],
)
def test_invalid_artifact_fails_with_context(spec, data, error):
    with pytest.raises(ValueError, match=error) as caught:
        MediaArtifact(spec, data).validate(context="pipeline=test, request_id=req", name="broken")
    assert "pipeline=test" in str(caught.value) and "artifact='broken'" in str(caught.value)
    assert "request_id=req" in str(caught.value)


def _artifacts():
    return {
        "video_preview": MediaArtifact(
            MediaSpec("video", "decoded", "THWC", fps=12), torch.zeros(3, 4, 5, 3, dtype=torch.uint8)
        ),
        "video_latent": MediaArtifact(
            MediaSpec("video", "latent", "CTHW"), torch.zeros(16, 2, 2, 2, dtype=torch.bfloat16)
        ),
        "audio": MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=32000), torch.zeros(2, 160)),
    }


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "requested", "spec", "primary"])
def test_name_and_declaration_mismatches_are_errors(mode):
    items = list(_artifacts().items())
    specs = {name: artifact.spec for name, artifact in items}
    primary, requested = "video_preview", None
    if mode == "missing":
        items.pop()
    elif mode == "extra":
        items.append(("extra", items[0][1]))
    elif mode == "duplicate":
        items.append(items[0])
    elif mode == "requested":
        requested = ["missing_preview"]
    elif mode == "spec":
        specs["video_preview"] = replace(specs["video_preview"], fps=30)
    else:
        primary = "missing"
    with pytest.raises(ValueError):
        validate_artifacts(items, specs, primary=primary, requested=requested, context="test")


@pytest.mark.parametrize("primary,output_type", [("video_preview", "pt"), ("video_latent", "latent"), ("audio", "pt")])
def test_named_media_survives_real_upstream_formatter_and_strategy(primary, output_type):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _minimax_h3_post_process
    from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    artifacts = _artifacts()
    native = with_media_artifacts(
        DiffusionOutput(output={"payload": {"video": torch.zeros(1)}, "metadata": {"rl": {"sentinel": 7}}}),
        artifacts=list(artifacts.items()),
        specs={name: item.spec for name, item in artifacts.items()},
        primary=primary,
        preview="video_preview",
        audio="audio",
        context="pipeline=test, request_id=req",
    )
    processed = _minimax_h3_post_process(native.output)
    normalized = normalize_diffusion_postprocess_output(processed)
    request = SimpleNamespace(
        request_id="req", prompt={"prompt_ids": [1]}, sampling_params=OmniDiffusionSamplingParams()
    )
    final = format_diffusion_outputs(
        request=request,
        od_config=SimpleNamespace(model_class_name="MiniMaxH3Pipeline"),
        diffusion_output=native,
        output_data=processed,
        postprocess_output=normalized,
    )[0]
    result = DiffusionStrategy(SimpleNamespace(global_steps=1)).process_output(
        final, None, {"output_type": output_type}
    )
    assert set(result.artifacts) == set(artifacts)
    assert result.primary_artifact == primary and result.preview_artifact == "video_preview"
    assert result.artifacts["video_preview"].spec.layout == "TCHW"
    assert result.artifacts["video_latent"].data.dtype == torch.bfloat16
    assert result.artifacts["audio"].spec.sample_rate == 32000
    assert result.extra_fields["sentinel"] == 7
    assert result.extra_fields["audio"].shape == (2, 160)
    fields = artifact_fields(result.artifacts, primary, result.preview_artifact)
    restored = artifacts_from_fields(fields, context="transport")
    assert set(restored) == set(artifacts)
    torch.testing.assert_close(restored[primary].data, result.diffusion_output)


def test_artifact_transport_rejects_dropped_declarations_and_data():
    artifacts = _artifacts()
    fields = artifact_fields(artifacts, "video_latent", "video_preview")
    del fields["media_artifact__audio"]
    with pytest.raises(ValueError, match="declared artifacts missing"):
        artifacts_from_fields(fields, context="test")
    del fields["media_artifact_specs"]
    with pytest.raises(ValueError, match="no declarations"):
        artifacts_from_fields(fields, context="test")


def test_named_empty_payload_is_not_misreported_as_abort():
    final = SimpleNamespace(
        images=[],
        request_id="missing-output",
        multimodal_output={
            "metadata": {
                "media_artifacts": {
                    "primary": "video_preview",
                    "preview": "video_preview",
                    "audio": None,
                    "specs": {"video_preview": asdict(MediaSpec("video", "decoded", "TCHW", fps=24))},
                }
            }
        },
    )
    with pytest.raises(ValueError, match="missing-output.*named artifact payload"):
        DiffusionStrategy(SimpleNamespace(global_steps=1)).process_output(final, None, {})


def test_media_only_postprocessor_does_not_touch_named_payloads():
    wrapped = wrap_rollout_postprocessor(
        lambda *args, **kwargs: pytest.fail("already-normalized media was reprocessed")
    )
    data = {"payload": {"video": {}}, "metadata": {"media_artifacts": {"specs": {}}}}
    assert wrapped(data) is data
    with pytest.raises(ValueError, match="multiple payload"):
        wrapped({"payload": {"video": torch.zeros(1), "audio": torch.zeros(1)}})


@pytest.mark.parametrize("nested", [False, True])
def test_requested_outputs_survive_sampling_lowering(nested):
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    strategy = DiffusionStrategy(
        SimpleNamespace(
            engine=SimpleNamespace(
                default_sampling_params_list=[None],
                engine=SimpleNamespace(get_stage_metadata=lambda index: SimpleNamespace(stage_type="diffusion")),
            )
        )
    )
    sampling = {"requested_outputs": ["video_preview", "video_latent"]}
    _, params = strategy.preprocess_input(
        OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1]), {"extra_args": sampling} if nested else sampling, None
    )
    assert params[0].extra_args == sampling
    with pytest.raises(ValueError, match="Conflicting diffusion sampling"):
        strategy.preprocess_input(
            OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1]),
            {"requested_outputs": ["one"], "extra_args": {"requested_outputs": ["two"]}},
            None,
        )


def test_preview_selection_never_falls_back_to_latent():
    with pytest.raises(ValueError, match="absent"):
        select_artifact(
            {"video_latent": _artifacts()["video_latent"]},
            name="video_preview",
            modality="video",
            representation="decoded",
        )
