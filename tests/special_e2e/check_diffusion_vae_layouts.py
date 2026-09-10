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
"""GPU probes for actual checkpoint VAE implementations, independently of DiT quality.

Run with CUDA_VISIBLE_DEVICES selecting one device and --models-json pointing to
an explicit case-name -> local checkpoint-root mapping. Nothing is downloaded,
no Ray cluster is touched, and cases run sequentially. JSON records checkpoint
paths, actual module classes, parameter counts, and native/decoded shapes/dtypes.
This is VAE layout evidence, not a substitute for an end-to-end training smoke.
"""

import argparse
import gc
import json
import traceback
from pathlib import Path

import torch

from verl_omni.pipelines.diffusion_rollout_output import quantize_pixels
from verl_omni.pipelines.rollout_artifacts import MediaArtifact
from verl_omni.pipelines.rollout_media import MediaSpec


def _load_diffusers(root, subfolder, device):
    import diffusers

    config = json.loads((root / subfolder / "config.json").read_text())
    if subfolder == "vocoder":
        from diffusers.pipelines.ltx2 import vocoder

        cls = getattr(vocoder, config["_class_name"])
    else:
        cls = getattr(diffusers, config["_class_name"])
    return (
        cls.from_pretrained(str(root), subfolder=subfolder, local_files_only=True, torch_dtype=torch.float32)
        .eval()
        .to(device)
    )


@torch.inference_mode()
def probe(case, checkpoint, device):
    """Decode a small native latent using a real VAE, then check the declared axes."""
    root = Path(checkpoint)
    audio = case.endswith("audio")
    if case.startswith("h3-"):
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

        component = "audio_vae" if audio else "video_vae"
        cls = MiniMaxH3AudioVAE if audio else MiniMaxH3VideoVAE
        model = cls(str(root / component), device=device)
        channels = int(model.config_dict["latent_channels"])
        shape = (2, channels, 16) if audio else (1, channels, 7, 8, 12)
        latents = torch.zeros(shape, device=device)
        if audio:
            decoded = model.decode_latent(latents)
        else:
            # Match MiniMaxH3Pipeline.decode's FP16 decoder autocast contract.
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                decoded = model.decode_latent(latents)
        native_layout, decoded_layout = ("CLT", "NCT") if audio else ("NCTHW", "NCTHW")
        sample_rate = model.sample_rate if audio else None
    elif case == "bagel":
        from safetensors.torch import load_file
        from vllm_omni.diffusion.models.bagel.pipeline_bagel import AutoEncoder, default_ae_params

        model = AutoEncoder(default_ae_params())
        model.load_state_dict(load_file(str(root / "ae.safetensors")))
        model = model.eval().to(device)
        latents = torch.zeros(1, 16, 4, 6, device=device)
        decoded = model.decode(latents)
        native_layout, decoded_layout, sample_rate = "NCHW", "NCHW", None
    else:
        model = _load_diffusers(root, "audio_vae" if audio else "vae", device)
        channels = model.config.z_dim if case in ("qwen-image", "wan") else model.config.latent_channels
        if case == "qwen-image":
            shape, native_layout = (1, channels, 1, 4, 6), "NCTHW"
        elif case in ("sd3", "boogu"):
            shape, native_layout = (1, channels, 4, 6), "NCHW"
        elif case in ("wan", "ltx-video"):
            shape, native_layout = (1, channels, 2, 4, 6), "NCTHW"
        elif case == "ltx-audio":
            shape, native_layout = (1, channels, 8, 16), "NCTF"
        else:
            raise ValueError(f"Unknown probe case {case!r}")
        latents = torch.zeros(shape, device=device)
        kwargs = {"return_dict": False}
        if case == "ltx-video" and model.config.timestep_conditioning:
            kwargs["timestep"] = torch.zeros(1, device=device)
        decoded = model.decode(latents, **kwargs)[0]
        sample_rate = None
        if audio:
            vocoder = _load_diffusers(root, "vocoder", device)
            decoded = vocoder(decoded)
            sample_rate = int(vocoder.config.output_sampling_rate)
            decoded_layout = "NCT"
        else:
            decoded_layout = native_layout

    expected_rank = len(decoded_layout)
    assert decoded.ndim == expected_rank, (case, decoded_layout, tuple(decoded.shape))
    assert decoded.shape[0] == 1 and torch.isfinite(decoded).all(), case
    if audio:
        assert decoded.shape[1] == 2, (case, tuple(decoded.shape))
        artifact = MediaArtifact(MediaSpec("audio", "decoded", "CT", sample_rate=sample_rate), decoded[0])
    elif case == "h3-video":
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _prepare_minimax_h3_video_output

        assert decoded.shape[1] == 3, tuple(decoded.shape)
        pixels = _prepare_minimax_h3_video_output(decoded)
        artifact = MediaArtifact(MediaSpec("video", "decoded", "THWC", fps=24), pixels[0])
    elif case == "qwen-image":
        assert decoded.shape[1] == 3 and decoded.shape[2] == 1, tuple(decoded.shape)
        artifact = MediaArtifact(
            MediaSpec("image", "decoded", "CHW"), quantize_pixels(decoded[0, :, 0], "minus_one_one", context=case)
        )
    else:
        assert decoded.shape[1] == 3, (case, tuple(decoded.shape))
        kind = "video" if decoded_layout == "NCTHW" else "image"
        artifact = MediaArtifact(
            MediaSpec(kind, "decoded", "CTHW" if kind == "video" else "CHW", fps=24 if kind == "video" else None),
            quantize_pixels(decoded[0], "minus_one_one", context=case),
        )
    artifact = artifact.normalized(context=f"pipeline={case}, request_id=vae-probe", name="decoded")
    return {
        "checkpoint": str(root.resolve()),
        "implementation": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": sum(p.numel() for p in model.parameters()),
        "native_layout": native_layout,
        "native_shape": list(latents.shape),
        "native_dtype": str(latents.dtype),
        "decoded_layout": decoded_layout,
        "decoded_shape": list(decoded.shape),
        "decoded_dtype": str(decoded.dtype),
        "canonical_layout": artifact.spec.layout,
        "canonical_shape": list(artifact.data.shape),
        "sample_rate": sample_rate,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main():
    """Probe only explicitly supplied local checkpoints and fail if any case fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.models_json).read_text())
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.35)
    device = torch.device("cuda", 0)
    results = {}
    for case, checkpoint in cases.items():
        torch.cuda.reset_peak_memory_stats()
        try:
            results[case] = {"status": "passed", **probe(case, checkpoint, device)}
        except Exception as error:
            traceback.print_exc()
            results[case] = {"status": "failed", "checkpoint": checkpoint, "error": str(error)}
        gc.collect()
        torch.cuda.empty_cache()
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        print(case, results[case], flush=True)
    raise SystemExit(any(result["status"] != "passed" for result in results.values()))


if __name__ == "__main__":
    main()
