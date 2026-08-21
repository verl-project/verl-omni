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
"""Build a tiny local BAGEL-7B-MoT-style checkpoint with random weights (offline).

Layout mirrors ``ByteDance-Seed/BAGEL-7B-MoT`` so vllm-omni ``BagelPipeline`` and
verl-omni ``BagelForTraining.from_pretrained`` can both load it:

  config.json / llm_config.json / vit_config.json
  ema.safetensors   (MoT language model + gen projections + ViT tower)
  ae.safetensors    (AutoEncoder matching vllm-omni ``default_ae_params``)
  tokenizer files + Siglip preprocessor_config.json

``BagelPipeline`` builds ``vit_model`` unconditionally, even when ``visual_und=False``,
so ``ema.safetensors`` must carry ``vit_model.*`` weights too.

The LLM/ViT stacks are shrunk; the VAE keeps the product geometry because
vllm-omni always constructs ``AutoEncoder(default_ae_params())``.

Usage:
    python tests/special_e2e/build_bagel_tiny_random.py \\
        --output-dir ~/models/tiny-random/BAGEL-MoT
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import Qwen2TokenizerFast, SiglipVisionConfig, SiglipVisionModel
from vllm_omni.diffusion.models.bagel.autoencoder import AutoEncoder, AutoEncoderParams

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelForTraining, BagelTrainingConfig

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/BAGEL-MoT")

_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)

_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
]


def _build_tiny_chatml_tokenizer(*, vocab_size: int = 2048) -> Qwen2TokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=_SPECIAL_TOKENS)
    corpus = [
        "a red circle on a white background",
        "a blue square on a black background",
        "a green triangle next to an orange rectangle",
        "<|im_start|>user\nhello<|im_end|>\n",
        " ".join(str(i) for i in range(256)),
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return Qwen2TokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        model_max_length=2048,
        chat_template=_CHATML_TEMPLATE,
    )


def _tiny_llm_config(vocab_size: int, *, bos_token_id: int, eos_token_id: int) -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "max_position_embeddings": 2048,
        "max_window_layers": 2,
        "model_type": "qwen2",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "qk_norm": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": 4096,
        "tie_word_embeddings": True,  # BagelForTraining has no separate lm_head to checkpoint
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": vocab_size,
    }


def _tiny_vit_config() -> dict:
    return {
        "hidden_size": 64,
        "image_size": 224,
        "intermediate_size": 128,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "patch_size": 14,
        "num_channels": 3,
    }


def _training_state_to_ema(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map ``BagelForTraining`` keys to published ``ema.safetensors`` names."""
    mapped: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
            mapped[name] = tensor.contiguous()
        else:
            mapped[f"language_model.model.{name}"] = tensor.contiguous()
    return mapped


def _build_ae_state_dict() -> dict[str, torch.Tensor]:
    """Random AE weights matching vllm-omni ``default_ae_params()`` geometry."""
    # Keep in sync with vllm_omni.diffusion.models.bagel.pipeline_bagel.default_ae_params.
    params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    ae = AutoEncoder(params)
    return {name: tensor.detach().contiguous() for name, tensor in ae.state_dict().items()}


def _build_vit_state_dict(vit_config: dict) -> dict[str, torch.Tensor]:
    """Random SiglipVisionModel weights, keyed like ``SiglipNaViTWrapper`` expects.

    Mirrors its ``hasattr(vision_model, "vision_model")`` unwrap so keys land under
    one ``vision_model.`` level regardless of transformers version.
    """
    config = SiglipVisionConfig(**vit_config, vision_use_head=False)
    vit = SiglipVisionModel(config)
    inner = vit.vision_model if hasattr(vit, "vision_model") else vit
    return {
        f"vit_model.vision_model.{name}": tensor.detach().contiguous() for name, tensor in inner.state_dict().items()
    }


def _build_bagel_und_state_dict(
    *, hidden_size: int, vit_hidden_size: int, vit_max_num_patch_per_side: int
) -> dict[str, torch.Tensor]:
    """Random connector/vit_pos_embed weights; ``Bagel`` always builds these regardless of ``visual_und``."""
    fc1, fc2 = torch.nn.Linear(vit_hidden_size, hidden_size), torch.nn.Linear(hidden_size, hidden_size)
    return {
        "connector.fc1.weight": fc1.weight.detach().contiguous(),
        "connector.fc1.bias": fc1.bias.detach().contiguous(),
        "connector.fc2.weight": fc2.weight.detach().contiguous(),
        "connector.fc2.bias": fc2.bias.detach().contiguous(),
        "vit_pos_embed.pos_embed": torch.randn(vit_max_num_patch_per_side**2, hidden_size),
    }


def ensure_tiny_bagel_checkpoint(
    output_dir: str,
    *,
    seed: int = 42,
    vocab_size: int = 2048,
    skip_if_exists: bool = True,
) -> str:
    """Build and save a tiny BAGEL checkpoint if it does not already exist."""
    output_dir = os.path.expanduser(output_dir)
    marker = os.path.join(output_dir, "ema.safetensors")
    if skip_if_exists and os.path.isfile(marker) and os.path.isfile(os.path.join(output_dir, "config.json")):
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    tokenizer = _build_tiny_chatml_tokenizer(vocab_size=vocab_size)
    tokenizer.save_pretrained(output_dir)

    bos_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    eos_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    soi_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    eoi_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    pad_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    vocab_size = max(int(len(tokenizer)), soi_id + 1, eoi_id + 1)

    llm_config = _tiny_llm_config(vocab_size, bos_token_id=bos_id, eos_token_id=eos_id)
    vit_config = _tiny_vit_config()
    vit_max_num_patch_per_side = 16
    root_config = {
        "architectures": ["BagelForConditionalGeneration"],
        "model_type": "bagel",
        "visual_gen": True,
        "visual_und": False,  # spares only the AR stage; DiT always builds vit_model/connector/vit_pos_embed
        "llm_config": llm_config,
        "vit_config": {**vit_config, "num_channels": 3},
        "vae_config": {"z_channels": 16, "downsample": 8},
        "latent_patch_size": 2,
        "max_latent_size": 32,
        "vit_max_num_patch_per_side": vit_max_num_patch_per_side,
        "connector_act": "gelu_pytorch_tanh",
        "interpolate_pos": False,
        "timestep_shift": 1.0,
        "start_of_image_id": soi_id,
        "end_of_image_id": eoi_id,
        "torch_dtype": "bfloat16",
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(root_config, f, indent=2)
    with open(os.path.join(output_dir, "llm_config.json"), "w", encoding="utf-8") as f:
        json.dump(llm_config, f, indent=2)
    with open(os.path.join(output_dir, "vit_config.json"), "w", encoding="utf-8") as f:
        json.dump(vit_config, f, indent=2)
    with open(os.path.join(output_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "bos_token_id": bos_id,
                "pad_token_id": pad_id,
                "do_sample": True,
                "eos_token_id": [eos_id, pad_id],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            },
            f,
            indent=2,
        )
    with open(os.path.join(output_dir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "do_convert_rgb": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_processor_type": "SiglipImageProcessor",
                "image_std": [0.5, 0.5, 0.5],
                "processor_class": "BagelProcessor",
                "rescale_factor": 0.00392156862745098,
                "resample": 3,
                "size": {"height": 224, "width": 224},
            },
            f,
            indent=2,
        )

    train_config = BagelTrainingConfig(
        hidden_size=llm_config["hidden_size"],
        intermediate_size=llm_config["intermediate_size"],
        num_hidden_layers=llm_config["num_hidden_layers"],
        num_attention_heads=llm_config["num_attention_heads"],
        num_key_value_heads=llm_config["num_key_value_heads"],
        vocab_size=vocab_size,
        rms_norm_eps=llm_config["rms_norm_eps"],
        rope_theta=llm_config["rope_theta"],
        max_position_embeddings=llm_config["max_position_embeddings"],
        latent_patch_size=2,
        max_latent_size=32,
        latent_channel=16,
        vae_downsample=8,
        start_of_image_id=soi_id,
        end_of_image_id=eoi_id,
    )
    model = BagelForTraining(train_config)
    training_state_dict = model.state_dict()
    ema_state_dict = _training_state_to_ema(training_state_dict)
    ema_state_dict.update(_build_vit_state_dict(vit_config))
    ema_state_dict.update(
        _build_bagel_und_state_dict(
            hidden_size=llm_config["hidden_size"],
            vit_hidden_size=vit_config["hidden_size"],
            vit_max_num_patch_per_side=vit_max_num_patch_per_side,
        )
    )
    # Qwen2MoTForCausalLM always allocates an untied lm_head; reuse embed_tokens for shape.
    ema_state_dict["language_model.lm_head.weight"] = (
        training_state_dict["embed_tokens.weight"].detach().clone().contiguous()
    )
    save_file(ema_state_dict, os.path.join(output_dir, "ema.safetensors"))
    save_file(_build_ae_state_dict(), os.path.join(output_dir, "ae.safetensors"))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny BAGEL MoT checkpoint offline (random weights).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true", help="Rebuild even when ema.safetensors already exists")
    args = parser.parse_args()

    output_dir = ensure_tiny_bagel_checkpoint(
        args.output_dir,
        seed=args.seed,
        vocab_size=args.vocab_size,
        skip_if_exists=not args.force,
    )
    print(f"Tiny BAGEL checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
