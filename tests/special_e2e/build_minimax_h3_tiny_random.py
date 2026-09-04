#!/usr/bin/env python
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
"""Build a self-contained tiny random MiniMax-H3 FL2VA checkpoint.

The checkpoint is suitable for T2VA and image-conditioned FL2VA GPU smoke
runs. It never reads, copies, or symlinks a real MiniMax-H3 checkpoint.

It starts from a pinned ``tiny-random/minimax-h3`` snapshot for the tiny DiT
weights and processor metadata. vLLM-Omni's native MiniMax-H3 pipeline differs
from that Diffusers checkpoint in three relevant ways:

* it consumes 24 video-latent channels and 32 audio-latent channels, while the
  HF tiny DiT has 8 channels for both streams;
* it requires a native Qwen3-VL encoder with 64 attention heads, 8 KV heads,
  and a 5120-wide hidden state;
* it loads native remote-code video/audio VAE components, while the HF tiny
  checkpoint stores standard Diffusers VAEs.

The builder expands the affected DiT projections with deterministic random
weights, creates a compact one-layer Qwen3-VL with a 5120-wide output and a
512-token vocabulary, and writes tiny local remote-code VAE stubs that
implement vLLM-Omni's native VAE contract. The stubs preserve H3 latent
geometry for T2VA, FL2VA image encoding, Ref2VA reference encoding, and
video/audio decoding; they are intentionally not suitable for image or audio
quality evaluation.

Layout produced under ``<output-dir>``::

    <output-dir>/
      FL2VA/                         # vLLM-Omni rollout model path
        model_index.json
        transformer/                 # fused-config DiT + Diffusers weights
        text_encoder/
        tokenizer/
        processor/
        video_vae/                   # local tiny remote-code stub
        audio_vae/                   # local tiny remote-code stub
      Ref2VA/                        # vLLM-Omni Ref2VA rollout model path
        model_index.json
        transformer/
        text_encoder/
        tokenizer/
        processor/
        video_vae/                   # local tiny remote-code stub
        audio_vae/                   # local tiny remote-code stub
      transformer/                   # Diffusers actor DiT config + weights

Usage::

    python tests/special_e2e/build_minimax_h3_tiny_random.py \
        --output-dir ~/models/tiny-random/minimax-h3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLTextConfig, Qwen3VLVisionConfig

_HF_TINY_REPO = "tiny-random/minimax-h3"
_HF_TINY_REVISION = "9018dbdcdb02a427905537035e8431c4a738d7c0"
# gpu_smoke globally selects hf-mirror.com, but that mirror cannot serve this
# repository's snapshot metadata reliably. This public source is fetched from
# the canonical Hub endpoint instead.
_HF_TINY_ENDPOINT = "https://huggingface.co"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/minimax-h3")
# Kept solely so existing H3 smoke callers can upgrade this builder without
# changing their invocation. The value is intentionally ignored: this builder
# never reads a real FL2VA checkpoint.
DEFAULT_SOURCE_FL2VA: str | None = None
_SEED = 42
_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32
_PATCH_VOLUME = 4  # MiniMax-H3 uses a (1, 2, 2) video patch.
_CHECKPOINT_FORMAT_VERSION = 4
_VLLM_TEXT_HIDDEN_SIZE = 5120
_VLLM_TEXT_NUM_ATTENTION_HEADS = 64
_VLLM_TEXT_NUM_KEY_VALUE_HEADS = 8
_VLLM_TEXT_HEAD_DIM = 8
_VLLM_TEXT_INTERMEDIATE_SIZE = 64
_VLLM_TEXT_VOCAB_SIZE = 512
_VLLM_TEXT_NUM_LAYERS = 1

# Diffusers-format field -> vLLM-Omni fused-arch field. Keys with the same name
# in both schemas are copied unchanged below.
_DIFFUSERS_TO_FUSED = {
    "num_refiner_layers": "token_refiner_num_layers",
    "ffn_dim": "ffn_hidden_size",
    "in_channels": "latents_dim",
    "audio_in_channels": "audio_latents_dim",
    "freq_dim": "timestep_input_dim",
    "time_embed_hidden_dim": "time_embed_hidden_size",
    "rope_freq_dim": "rope_inv_freq_len",
}
_SHARED_TRANSFORMER_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "attention_head_dim",
    "num_layers",
    "patch_size",
    "text_dim",
    "time_embed_dim",
    "rope_theta",
    "norm_eps",
    "qk_norm_eps",
    "final_norm_eps",
)

# H3's Diffusers state dict names and the target shapes after changing the
# video/audio latent channels. All other HF tiny tensors keep their exact shape.
_RESIZED_TRANSFORMER_TENSORS = {
    "audio_proj_in.weight": (64, _AUDIO_LATENT_CHANNELS),
    "audio_proj_out.bias": (_AUDIO_LATENT_CHANNELS,),
    "audio_proj_out.weight": (_AUDIO_LATENT_CHANNELS, 64),
    "context_embedder.weight": (64, _VLLM_TEXT_HIDDEN_SIZE),
    "proj_in.weight": (64, _VIDEO_LATENT_CHANNELS * _PATCH_VOLUME),
    "proj_out.bias": (_VIDEO_LATENT_CHANNELS * _PATCH_VOLUME,),
    "proj_out.weight": (_VIDEO_LATENT_CHANNELS * _PATCH_VOLUME, 64),
}

_VIDEO_VAE_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Tiny deterministic MiniMax-H3 video-VAE compatibility component."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class _VideoProcessor:
    @staticmethod
    def revert_tensor(value: torch.Tensor) -> torch.Tensor:
        return value


class _TinyVideoCore(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.parallel_tiling = False
        # MiniMaxH3VideoVAE computes the decoder tile grid in pixel space
        # before it calls decode_base. The tiny component deliberately has one
        # tile, but it must expose the same native VAE contract.
        self.vae_ratio = 16
        self.processor = _VideoProcessor()

    def split_tiles(self, input_len: int, is_decoder: bool = False):
        del is_decoder
        return [0], [input_len], []

    def encode_images(self, image, *, use_fp16_latent: bool = True):
        del use_fp16_latent
        width, height = image.size
        latent = torch.zeros(
            self.latent_channels,
            1,
            max(1, height // 16),
            max(1, width // 16),
            device=self.scale.device,
            dtype=self.scale.dtype,
        )
        return [latent + self.scale * 0]

    def encode_videos(self, frames, *, use_fp16_latent: bool = True):
        del use_fp16_latent
        image = frames[0] if isinstance(frames, (list, tuple)) else frames
        return self.encode_images(image)[0].unsqueeze(0)

    def decode_base(self, latent: torch.Tensor) -> torch.Tensor:
        # Native H3 maps latent T=2 to five output frames and each extra five
        # latent steps to another 17 frames. This preserves the outer pipeline's
        # frame geometry while deliberately producing meaningless random-model
        # pixels.
        latent_t = int(latent.shape[2])
        frames = 1 if latent_t == 1 else 5 + max(0, latent_t - 2) * 17 // 5
        decoded = latent[:, :3]
        return F.interpolate(
            decoded,
            size=(frames, max(1, latent.shape[3] * 16), max(1, latent.shape[4] * 16)),
            mode="trilinear",
            align_corners=False,
        )


class TinyH3VideoVAE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = _TinyVideoCore(int(config["latent_channels"]))

    @classmethod
    def from_pretrained(cls, component_path: str | Path):
        component_path = Path(component_path)
        config = json.loads((component_path / "config.json").read_text())
        model = cls(config)
        state = torch.load(component_path / "tiny_video_vae.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
'''

_AUDIO_VAE_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Tiny deterministic MiniMax-H3 audio-VAE compatibility component."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class _TinyAudioCore(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.attn_proj = False

    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del sample_rate
        return waveform

    def encoder(self, audio: torch.Tensor) -> torch.Tensor:
        # 40 Hz latent timeline for H3's fixed 32 kHz audio stream.
        length = max(1, int(audio.shape[-1]) // 800)
        return torch.zeros(
            audio.shape[0],
            length,
            self.latent_channels,
            device=audio.device,
            dtype=audio.dtype,
        ) + self.scale * 0

    @staticmethod
    def mean_proj(latent: torch.Tensor) -> torch.Tensor:
        return latent


class TinyH3AudioVAE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = _TinyAudioCore(int(config["latent_channels"]))

    @classmethod
    def from_pretrained(cls, component_path: str | Path):
        component_path = Path(component_path)
        config = json.loads((component_path / "config.json").read_text())
        model = cls(config)
        state = torch.load(component_path / "tiny_audio_vae.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # vLLM-Omni expects [batch, 1, samples] before it restores channels.
        return torch.zeros(
            latent.shape[0],
            1,
            max(1, int(latent.shape[-1]) * 800),
            device=latent.device,
            dtype=latent.dtype,
        ) + self.model.scale * 0
'''


def _hardlink_or_copy(source: Path, target: Path) -> None:
    """Materialize a snapshot file without retaining a dependency on its cache."""
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    """Copy a HF snapshot component, hard-linking immutable blobs when possible."""
    if target.exists():
        shutil.rmtree(target)
    for entry in source.rglob("*"):
        relative = entry.relative_to(source)
        destination = target / relative
        if entry.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            _hardlink_or_copy(entry, destination)


_TOKENIZER_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<unk>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)


def _write_vllm_tokenizer(component_dir: Path, chat_template: Path) -> dict[str, int]:
    """Write a compact Qwen2-compatible tokenizer with H3's special tokens."""
    component_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=_VLLM_TEXT_VOCAB_SIZE,
        min_frequency=1,
        special_tokens=list(_TOKENIZER_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = [
        "A short cinematic video of a person walking in a city.",
        "A colorful bird flies over the ocean with ambient music.",
        "tiny random MiniMax H3 t2va smoke test",
        *[f"token_{index:04d} cinematic scene object_{index % 97:02d}" for index in range(1024)],
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    if tokenizer.get_vocab_size() != _VLLM_TEXT_VOCAB_SIZE:
        raise RuntimeError(f"expected {_VLLM_TEXT_VOCAB_SIZE} tokenizer entries, got {tokenizer.get_vocab_size()}")
    tokenizer.save(str(component_dir / "tokenizer.json"))

    special_token_ids = {token: tokenizer.token_to_id(token) for token in _TOKENIZER_SPECIAL_TOKENS}
    if any(token_id is None for token_id in special_token_ids.values()):
        raise RuntimeError("the compact Qwen tokenizer is missing an H3 special token")
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 4096,
        "unk_token": "<unk>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": list(_TOKENIZER_SPECIAL_TOKENS[2:]),
        "clean_up_tokenization_spaces": False,
    }
    (component_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    special_tokens_map = {
        "unk_token": "<unk>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": list(_TOKENIZER_SPECIAL_TOKENS[2:]),
    }
    (component_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2) + "\n")
    _hardlink_or_copy(chat_template, component_dir / "chat_template.jinja")
    return {token: int(token_id) for token, token_id in special_token_ids.items()}


def _write_vllm_text_encoder(component_dir: Path, special_token_ids: dict[str, int]) -> None:
    """Write the smallest Qwen3-VL accepted by vLLM-Omni's H3 encoder."""
    text_config = Qwen3VLTextConfig(
        vocab_size=_VLLM_TEXT_VOCAB_SIZE,
        hidden_size=_VLLM_TEXT_HIDDEN_SIZE,
        intermediate_size=_VLLM_TEXT_INTERMEDIATE_SIZE,
        num_hidden_layers=_VLLM_TEXT_NUM_LAYERS,
        num_attention_heads=_VLLM_TEXT_NUM_ATTENTION_HEADS,
        num_key_value_heads=_VLLM_TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=_VLLM_TEXT_HEAD_DIM,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        pad_token_id=special_token_ids["<|endoftext|>"],
    )
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=_VLLM_TEXT_HIDDEN_SIZE,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=special_token_ids["<|image_pad|>"],
        video_token_id=special_token_ids["<|video_pad|>"],
        vision_start_token_id=special_token_ids["<|vision_start|>"],
        vision_end_token_id=special_token_ids["<|vision_end|>"],
        tie_word_embeddings=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_SEED)
        model = Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
    model.save_pretrained(component_dir, safe_serialization=True)


def _copy_tokenizer_to_processor(tokenizer_dir: Path, processor_dir: Path) -> None:
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        destination = processor_dir / filename
        destination.unlink(missing_ok=True)
        shutil.copy2(tokenizer_dir / filename, destination)


def _download_hf_tiny(cache_dir: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            _HF_TINY_REPO,
            repo_type="model",
            revision=_HF_TINY_REVISION,
            endpoint=_HF_TINY_ENDPOINT,
            cache_dir=cache_dir,
            allow_patterns=[
                "text_encoder/*",
                "tokenizer/*",
                "processor/*",
                "transformer/*",
                "transformer_ref/*",
            ],
        )
    )


def _expanded_transformer_config(source: Path) -> dict:
    config = json.loads(source.read_text())
    config["in_channels"] = _VIDEO_LATENT_CHANNELS
    config["audio_in_channels"] = _AUDIO_LATENT_CHANNELS
    config["text_dim"] = _VLLM_TEXT_HIDDEN_SIZE
    return config


def _make_fused_transformer_config(diffusers_config: dict) -> dict:
    """Translate a Diffusers H3 config to vLLM-Omni's fused-arch schema."""
    hidden_size = int(diffusers_config["hidden_size"])
    fused: dict = {}
    for name in _SHARED_TRANSFORMER_FIELDS:
        if name in diffusers_config:
            fused[name] = diffusers_config[name]
    for source_name, target_name in _DIFFUSERS_TO_FUSED.items():
        if source_name in diffusers_config:
            fused[target_name] = diffusers_config[source_name]
    fused["adaln_out_features"] = 18 * hidden_size
    fused["final_adaln_out_features"] = 2 * hidden_size
    fused["_class_name"] = "MiniMaxH3DiTModel"
    fused["_diffusers_version"] = diffusers_config.get("_diffusers_version", "0.40.0.dev0")
    return fused


def _resize_tensor(tensor: torch.Tensor, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Copy the HF tiny tensor into a deterministic random tensor of ``shape``."""
    if tuple(tensor.shape) == shape:
        return tensor.contiguous()
    result = torch.empty(shape, dtype=tensor.dtype)
    result.normal_(mean=0.0, std=0.1, generator=generator)
    slices = tuple(slice(0, min(source, target)) for source, target in zip(tensor.shape, shape, strict=True))
    result[slices] = tensor[slices]
    return result.contiguous()


def _write_expanded_transformer(source_dir: Path, target_dir: Path, *, config: dict, seed: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source_weights = source_dir / "diffusion_pytorch_model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensors[name] = _resize_tensor(
                tensor,
                _RESIZED_TRANSFORMER_TENSORS.get(name, tuple(tensor.shape)),
                generator,
            )
    save_file(tensors, target_dir / "diffusion_pytorch_model.safetensors")
    (target_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _write_video_vae(component_dir: Path) -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "_class_name": "TinyH3VideoVAE",
        "auto_map": {"AutoModel": "tiny_h3_video_vae.TinyH3VideoVAE"},
        "latent_channels": _VIDEO_LATENT_CHANNELS,
        "latents_mean": [0.0] * _VIDEO_LATENT_CHANNELS,
        "latents_std": [1.0] * _VIDEO_LATENT_CHANNELS,
    }
    (component_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (component_dir / "tiny_h3_video_vae.py").write_text(_VIDEO_VAE_SOURCE)
    namespace: dict = {}
    exec(_VIDEO_VAE_SOURCE, namespace)
    torch.save(namespace["TinyH3VideoVAE"](config).state_dict(), component_dir / "tiny_video_vae.pt")


def _write_audio_vae(component_dir: Path) -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "_class_name": "TinyH3AudioVAE",
        "auto_map": {"AutoModel": "tiny_h3_audio_vae.TinyH3AudioVAE"},
        "latent_channels": _AUDIO_LATENT_CHANNELS,
        "latents_mean": [0.0] * _AUDIO_LATENT_CHANNELS,
        "latents_std": [1.0] * _AUDIO_LATENT_CHANNELS,
        "sample_rate": 32000,
    }
    (component_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (component_dir / "tiny_h3_audio_vae.py").write_text(_AUDIO_VAE_SOURCE)
    namespace: dict = {}
    exec(_AUDIO_VAE_SOURCE, namespace)
    torch.save(namespace["TinyH3AudioVAE"](config).state_dict(), component_dir / "tiny_audio_vae.pt")


def _write_model_index(path: Path, *, partition: str, tasks: list[str]) -> None:
    # vLLM-Omni reads this release metadata and uses fixed subfolder names for
    # all components, so a full Diffusers component index is intentionally not
    # needed here.
    index = {
        "_class_name": "MiniMaxH3Pipeline",
        "_minimax_h3": {
            "partition": partition,
            "tasks": tasks,
            "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        },
    }
    path.write_text(json.dumps(index, indent=2) + "\n")


def _write_partition(
    output: Path,
    name: str,
    *,
    partition: str,
    tasks: list[str],
    hf_root: Path,
) -> Path:
    """Write one self-contained MiniMax-H3 partition (FL2VA or Ref2VA).

    The two partitions share the same component geometry; they differ only in
    the partition directory name and the ``model_index.json`` release metadata
    that tells vLLM-Omni which task set and base schedule to use.
    """
    component_dir = output / name
    component_dir.mkdir()
    _write_model_index(
        component_dir / "model_index.json",
        partition=partition,
        tasks=tasks,
    )
    # vLLM-Omni needs the fused-arch schema while the actor needs the same
    # expanded weights under Diffusers' schema.
    diffusers_config = _expanded_transformer_config(hf_root / "transformer" / "config.json")
    _write_expanded_transformer(
        hf_root / "transformer",
        component_dir / "transformer",
        config=_make_fused_transformer_config(diffusers_config),
        seed=_SEED,
    )
    special_token_ids = _write_vllm_tokenizer(
        component_dir / "tokenizer",
        hf_root / "tokenizer" / "chat_template.jinja",
    )
    _write_vllm_text_encoder(component_dir / "text_encoder", special_token_ids)
    _copy_tree(hf_root / "processor", component_dir / "processor")
    _copy_tokenizer_to_processor(component_dir / "tokenizer", component_dir / "processor")
    # Qwen3VLProcessor requires this key even though the HF tiny processor
    # snapshot only supplies the generic preprocessor metadata.
    (component_dir / "processor" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}) + "\n")
    _write_video_vae(component_dir / "video_vae")
    _write_audio_vae(component_dir / "audio_vae")
    return component_dir


def _checkpoint_is_complete(output_dir: Path) -> bool:
    required = (
        output_dir / "tiny_checkpoint.json",
        output_dir / "FL2VA" / "model_index.json",
        output_dir / "FL2VA" / "transformer" / "config.json",
        output_dir / "FL2VA" / "transformer" / "diffusion_pytorch_model.safetensors",
        output_dir / "FL2VA" / "text_encoder" / "config.json",
        output_dir / "FL2VA" / "text_encoder" / "model.safetensors",
        output_dir / "FL2VA" / "tokenizer" / "tokenizer.json",
        output_dir / "FL2VA" / "processor" / "tokenizer.json",
        output_dir / "FL2VA" / "video_vae" / "config.json",
        output_dir / "FL2VA" / "audio_vae" / "config.json",
        output_dir / "transformer" / "config.json",
        output_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        output_dir / "Ref2VA" / "model_index.json",
        output_dir / "Ref2VA" / "transformer" / "config.json",
        output_dir / "Ref2VA" / "transformer" / "diffusion_pytorch_model.safetensors",
        output_dir / "Ref2VA" / "text_encoder" / "config.json",
        output_dir / "Ref2VA" / "text_encoder" / "model.safetensors",
        output_dir / "Ref2VA" / "tokenizer" / "tokenizer.json",
        output_dir / "Ref2VA" / "processor" / "tokenizer.json",
        output_dir / "Ref2VA" / "video_vae" / "config.json",
        output_dir / "Ref2VA" / "audio_vae" / "config.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads((output_dir / "tiny_checkpoint.json").read_text())
    except (OSError, ValueError):
        return False
    return metadata.get("format_version") == _CHECKPOINT_FORMAT_VERSION


def ensure_tiny_minimax_h3_checkpoint(
    output_dir: str,
    *,
    source_fl2va: str | None = None,
    hf_cache_dir: str | None = None,
    skip_if_exists: bool = True,
) -> str:
    """Build the self-contained checkpoint, or return an existing complete one.

    ``source_fl2va`` is accepted for backward compatibility with the original
    H3 smoke runner and is deliberately ignored.
    """
    del source_fl2va
    output = Path(os.path.expanduser(output_dir))
    if skip_if_exists and _checkpoint_is_complete(output):
        print(f"tiny MiniMax-H3 checkpoint already present at {output}", flush=True)
        return str(output)

    hf_root = _download_hf_tiny(hf_cache_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    _write_partition(
        output,
        "FL2VA",
        partition="fl2va",
        tasks=["t2va", "fl2va"],
        hf_root=hf_root,
    )
    _write_partition(
        output,
        "Ref2VA",
        partition="ref2va",
        tasks=["ref2va"],
        hf_root=hf_root,
    )
    # Both vLLM-Omni and the actor need the exact same expanded weight geometry,
    # but they consume different config schemas.
    actor_config = _expanded_transformer_config(hf_root / "transformer_ref" / "config.json")
    _write_expanded_transformer(
        hf_root / "transformer_ref",
        output / "transformer",
        config=actor_config,
        seed=_SEED + 1,
    )
    (output / "tiny_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": _CHECKPOINT_FORMAT_VERSION,
                "text_hidden_size": _VLLM_TEXT_HIDDEN_SIZE,
                "text_num_attention_heads": _VLLM_TEXT_NUM_ATTENTION_HEADS,
                "text_num_key_value_heads": _VLLM_TEXT_NUM_KEY_VALUE_HEADS,
                "text_num_hidden_layers": _VLLM_TEXT_NUM_LAYERS,
                "text_vocab_size": _VLLM_TEXT_VOCAB_SIZE,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"self-contained tiny MiniMax-H3 checkpoint assembled at {output}", flush=True)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the checkpoint is complete.")
    args = parser.parse_args()
    ensure_tiny_minimax_h3_checkpoint(
        args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()
