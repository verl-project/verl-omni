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
"""Generate an image from a DMD2 student export with the training-matched fp32 Euler path."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import (
    QwenImageConditionProvider,
    QwenImageDMD2,
    build_qwen_dmd_sigmas,
)
from verl_omni.trainer.diffusion.distillation.utils import ode_euler_step
from verl_omni.utils.fs import diffusion_model_provenance, resolve_model_local_dir
from verl_omni.workers.engine.lora_adapter_mixin import load_diffusers_lora_adapter


@torch.inference_mode()
def generate(artifact, prompt, seed, base_model=None, device=None):
    """Load a selected-adapter artifact and return one decoded RGB image."""
    from diffusers import QwenImagePipeline

    artifact = Path(artifact)
    metadata = json.loads((artifact / "inference_manifest.json").read_text())
    if metadata["algorithm"] != "dmd2" or metadata["sampler"] != "ode_euler" or metadata["guidance_scale"] != 1.0:
        raise ValueError("This generator requires a conditional-only Euler DMD2 export.")
    with (artifact / "adapter_model.safetensors").open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != metadata["weights_sha256"]:
            raise ValueError("Student artifact weight checksum does not match the manifest.")
    base_model = resolve_model_local_dir(base_model or metadata["base_model"])
    if "base_transformer_config_sha256" in metadata:
        provenance = diffusion_model_provenance(base_model)
        if provenance["base_transformer_config_sha256"] != metadata["base_transformer_config_sha256"]:
            raise ValueError("Base transformer configuration does not match the student export.")
        if (
            metadata["base_model_revision"] is not None
            and provenance["base_model_revision"] != metadata["base_model_revision"]
        ):
            raise ValueError("Base checkpoint revision does not match the student export.")
    device = torch.device(device or get_device_name())
    pipeline = QwenImagePipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    pipeline.transformer.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    load_diffusers_lora_adapter(pipeline.transformer, artifact, "student")
    pipeline.transformer.set_adapter("student")
    pipeline.transformer.eval()
    pipeline.transformer.set_attention_backend("native")
    config = SimpleNamespace(
        local_path=str(base_model),
        path=str(base_model),
        pipeline=SimpleNamespace(height=metadata["height"], width=metadata["width"], guidance_scale=None),
    )
    batch = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor_stack(batch, "raw_prompt", [prompt])
    provider = QwenImageConditionProvider(str(base_model), metadata["max_sequence_length"], " ")
    provider.pipeline = pipeline
    condition, _ = provider.encode(batch, device=device, dtype=torch.bfloat16, require_negative=False)
    shape, geometry = QwenImageDMD2.latent_geometry(pipeline.transformer, config, batch)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = QwenImageDMD2.pack_latents(torch.randn(shape, generator=generator, device=device, dtype=torch.float32))
    sigmas = build_qwen_dmd_sigmas(metadata["num_inference_steps"], metadata["rollout_timestep_shift"], device)
    for current, following in zip(sigmas[:-1], sigmas[1:], strict=True):
        inputs = QwenImageDMD2.prepare_dmd_inputs(pipeline.transformer, config, latents, current, condition, geometry)
        velocity = QwenImageDMD2.forward(pipeline.transformer, config, inputs)
        latents = ode_euler_step(latents, velocity, current, following)
    unpacked = QwenImagePipeline._unpack_latents(
        latents, metadata["height"], metadata["width"], geometry["vae_scale_factor"]
    )
    mean = torch.tensor(pipeline.vae.config.latents_mean, device=device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(pipeline.vae.config.latents_std, device=device).reshape(1, -1, 1, 1, 1)
    decoded = pipeline.vae.decode((unpacked * std + mean).to(pipeline.vae.dtype), return_dict=False)[0][:, :, 0]
    return pipeline.image_processor.postprocess(decoded, output_type="pil")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", help="Matching base checkpoint if moved from its recorded path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    image = generate(args.artifact, args.prompt, args.seed, args.base_model, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
