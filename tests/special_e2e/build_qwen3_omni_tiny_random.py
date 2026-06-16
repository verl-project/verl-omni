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
"""Build a tiny local Qwen3-Omni Thinker checkpoint with random weights (fully offline).

No tiny-random Qwen3-Omni checkpoint is published on the Hub, so the GSPO smoke
test builds one locally. Following ``build_sd3_tiny_random.py``, everything is
constructed in memory with **no Hub access**: the config is the default
``Qwen3OmniMoeConfig`` shrunk to a couple of layers, and the tokenizer is a tiny
ChatML BPE trained in process (the chat template is what verl's dataset loader
needs; tokenization quality is irrelevant for a plumbing smoke test).

Only the Thinker is exercised downstream (talker / code2wav are dropped at
FSDP-wrap time via ``_verl_strip_modules``), so those sub-configs are left at
their defaults.

Usage:
    python tests/special_e2e/build_qwen3_omni_tiny_random.py \
        --output-dir ~/models/tiny-random/Qwen3-Omni
"""

from __future__ import annotations

import argparse
import os

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import AutoModelForCausalLM, Qwen2TokenizerFast

# Importing the patch module registers Qwen3OmniMoeConfig with
# AutoModelForCausalLM (the Thinker is decoder-only despite its config class).
# It is NOT applied by ``import verl_omni`` — the patch is loaded on demand via
# ``actor_rollout_ref.model.external_lib`` — so apply it explicitly here.
from verl_omni.models.transformers.qwen3_omni_thinker import apply_qwen3_omni_thinker_patches

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/Qwen3-Omni")

# Minimal ChatML template (verl's dataset loader calls apply_chat_template).
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)

# Multimodal special tokens Qwen3OmniMoeProcessor expects on the tokenizer
# (read as tokenizer.image_token / .video_token / ... via extra_special_tokens).
_MM_EXTRA_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "audio_token": "<|audio_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "vision_eos_token": "<|vision_end|>",
    "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>",
}


def _build_tiny_chatml_tokenizer(*, vocab_size: int = 2048) -> Qwen2TokenizerFast:
    """Train a tiny byte-level BPE tokenizer in memory with ChatML special tokens.

    Returns a ``Qwen2TokenizerFast`` (not a bare ``PreTrainedTokenizerFast``) so it
    is accepted by ``Qwen3OmniMoeProcessor``, which also reads multimodal special
    tokens (image/video/audio) off the tokenizer via ``extra_special_tokens``.
    """
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + list(_MM_EXTRA_SPECIAL_TOKENS.values())
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Solve for x: 2 + 2 = 4. The final answer is 42.",
        "<|im_start|>user\nWhat is 1 + 1?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n",
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
        extra_special_tokens=_MM_EXTRA_SPECIAL_TOKENS,
    )


def _build_tiny_config(vocab_size: int):
    """Default Qwen3OmniMoeConfig shrunk to a 2-layer Thinker (no Hub access)."""
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig

    config = Qwen3OmniMoeConfig()

    # Thinker-only smoke test: skip the talker / code2wav stack entirely
    # (also dropped at FSDP-wrap time via _verl_strip_modules).
    config.enable_audio_output = False

    text = config.thinker_config.text_config
    text.num_hidden_layers = 2
    text.hidden_size = 128
    text.intermediate_size = 256
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 32
    text.num_experts = 4
    text.num_experts_per_tok = 2
    text.moe_intermediate_size = 128
    text.vocab_size = vocab_size
    # The default config leaves rope_scaling unset, but the M-RoPE rotary
    # embedding requires it. mrope_section must sum to head_dim // 2 (= 16 here).
    text.rope_scaling = {"rope_type": "default", "mrope_section": [8, 4, 4]}

    vision = config.thinker_config.vision_config
    # Shrink dims but keep every field vLLM-Omni's vision tower reads (the default
    # config omits some, e.g. image_size). deepstack_visual_indexes must stay < depth.
    vision.depth = 4
    vision.hidden_size = 128
    vision.intermediate_size = 256
    vision.num_heads = 4
    vision.out_hidden_size = 128  # must match text hidden_size for the projector
    vision.image_size = 768
    vision.in_chans = 3
    vision.in_channels = 3
    vision.patch_size = 16
    vision.spatial_patch_size = 16
    vision.spatial_merge_size = 2
    vision.temporal_patch_size = 2
    vision.num_position_embeddings = 2304
    vision.apply_vit_abs_pos_embed = True
    vision.tokens_per_second = 2
    vision.hidden_act = "gelu_pytorch_tanh"
    vision.deepstack_visual_indexes = [1, 2, 3]

    audio = config.thinker_config.audio_config
    audio.num_hidden_layers = 2
    audio.encoder_layers = 2
    audio.d_model = 128
    audio.encoder_attention_heads = 4
    audio.encoder_ffn_dim = 256
    audio.num_mel_bins = 128
    audio.output_dim = 128  # must match text hidden_size for the projector
    audio.downsample_hidden_size = 64
    audio.max_source_positions = 1500
    audio.n_window = 50
    audio.n_window_infer = 800
    audio.conv_chunksize = 500
    audio.scale_embedding = False
    audio.activation_function = "gelu"

    return config


def _build_tiny_processor(tokenizer):
    """Assemble a Qwen3OmniMoeProcessor from default sub-processors + the tiny tokenizer.

    vLLM-Omni's rollout model loader requires the multimodal processor files
    (preprocessor_config.json etc.); the image/video/audio sub-processors are
    vocab-independent, so their defaults are fine for a text-only smoke test.
    """
    from transformers import Qwen2VLImageProcessor, WhisperFeatureExtractor
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeProcessor

    try:
        from transformers import Qwen2VLVideoProcessor
    except ImportError:
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    return Qwen3OmniMoeProcessor(
        # Match the real checkpoint's image-processor geometry (patch_size=16,
        # merge_size=2, temporal_patch_size=2) so it stays consistent with the
        # vision config; the default Qwen2VLImageProcessor uses patch_size=14,
        # which mismatches and breaks the patch-embed reshape.
        image_processor=Qwen2VLImageProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=3136,
            max_pixels=12845056,
        ),
        video_processor=Qwen2VLVideoProcessor(),
        feature_extractor=WhisperFeatureExtractor(),
        tokenizer=tokenizer,
        chat_template=_CHATML_TEMPLATE,
    )


def build(output_dir: str, *, seed: int = 42, dtype: torch.dtype = torch.bfloat16) -> str:
    """Construct and save a tiny random-weight Qwen3-Omni Thinker checkpoint."""
    apply_qwen3_omni_thinker_patches()
    torch.manual_seed(seed)

    tokenizer = _build_tiny_chatml_tokenizer()
    config = _build_tiny_config(vocab_size=len(tokenizer))

    # Keep generation/special-token ids consistent between model and tokenizer.
    text = config.thinker_config.text_config
    text.bos_token_id = tokenizer.bos_token_id
    text.eos_token_id = tokenizer.eos_token_id
    text.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_config(config).to(dtype)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    # Save the full multimodal processor (tokenizer + image/video/audio configs);
    # the vLLM-Omni rollout loader requires preprocessor_config.json to exist.
    processor = _build_tiny_processor(tokenizer)
    processor.save_pretrained(output_dir)
    _merge_image_processor_config(output_dir, processor)
    return output_dir


def _merge_image_processor_config(output_dir: str, processor) -> None:
    """Re-merge image-processor keys into preprocessor_config.json (save_pretrained lets
    the audio feature extractor overwrite ``image_processor_type``, which the loader needs)."""
    import json

    pc_path = os.path.join(output_dir, "preprocessor_config.json")
    with open(pc_path) as f:
        merged = json.load(f)
    image_dict = processor.image_processor.to_dict()
    image_dict.pop("processor_class", None)
    merged.update(image_dict)  # adds image_processor_type + image params
    with open(pc_path, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)


def ensure_tiny_qwen3_omni_checkpoint(
    output_dir: str,
    *,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    skip_if_exists: bool = True,
) -> str:
    """Build the tiny checkpoint only if it is not already present."""
    output_dir = os.path.expanduser(output_dir)
    if skip_if_exists and os.path.isfile(os.path.join(output_dir, "config.json")):
        return output_dir
    return build(output_dir, seed=seed, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny Qwen3-Omni Thinker checkpoint offline (random weights).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when output-dir already contains config.json",
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    output_dir = ensure_tiny_qwen3_omni_checkpoint(
        args.output_dir,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny Qwen3-Omni checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
