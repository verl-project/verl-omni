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
"""Build a local tiny-random MiniCPM-o style checkpoint for smoke tests.

The checkpoint is intentionally minimal but uses the same loading surface as
MiniCPM-o: ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)`` and
``AutoModel.from_pretrained(..., trust_remote_code=True)``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import textwrap

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/MiniCPM-o-4_5")

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{{ message['content'] }}"
    "{{ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

_CONFIG_CODE = r"""
from transformers import PretrainedConfig


class MiniCPMOConfig(PretrainedConfig):
    model_type = "minicpmo"

    def __init__(
        self,
        vocab_size=2048,
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        init_vision=True,
        init_audio=True,
        init_tts=True,
        audio_config=None,
        slice_config=None,
        tts_config=None,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.version = "4.5"
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.audio_chunk_length = 1.0
        self.audio_pool_step = 5
        self.batch_vision_input = True
        self.drop_vision_last_layer = False
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_act = "silu"
        self.image_size = 448
        self.init_audio = init_audio
        self.init_tts = init_tts
        self.init_vision = init_vision
        self.initializer_range = 0.02
        self.listen_speak_type = "asr"
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.patch_size = 14
        self.query_num = 64
        self.rms_norm_eps = 1e-6
        self.rope_scaling = None
        self.rope_theta = 1000000
        self.slice_config = slice_config or {
            "max_slice_nums": 1,
            "model_type": "minicpmv",
            "patch_size": 14,
            "scale_resolution": 448,
        }
        self.slice_mode = True
        self.sliding_window = None
        self.stream_input = True
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.tie_word_embeddings = False
        self.use_cache = True
        self.use_image_id = True
        self.use_sliding_window = False
        self.vision_batch_size = 16
        self.audio_config = audio_config or {
            "_attn_implementation_autoset": True,
            "architectures": ["MiniCPMWhisperEncoder"],
            "d_model": hidden_size,
            "decoder_attention_heads": num_attention_heads,
            "decoder_ffn_dim": intermediate_size,
            "decoder_layers": num_hidden_layers,
            "encoder_attention_heads": num_attention_heads,
            "encoder_ffn_dim": intermediate_size,
            "encoder_layers": num_hidden_layers,
            "model_type": "whisper",
            "num_hidden_layers": num_hidden_layers,
            "num_mel_bins": 80,
            "torch_dtype": "float32",
            "vocab_size": vocab_size,
        }
        self.tts_config = tts_config or {
            "_attn_implementation_autoset": True,
            "attention_type": "full_attention",
            "attn_implementation": "eager",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "llm_dim": hidden_size,
            "llm_hidden_size": hidden_size,
            "llm_intermediate_size": intermediate_size,
            "model_type": "minicpmtts",
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "use_text": True,
        }
        self.vision_config = vision_config or {
            "_attn_implementation_autoset": True,
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": hidden_size,
            "image_size": 448,
            "intermediate_size": intermediate_size,
            "layer_norm_eps": 1e-6,
            "model_type": "siglip_vision_model",
            "num_attention_heads": num_attention_heads,
            "num_channels": 3,
            "num_hidden_layers": num_hidden_layers,
            "patch_size": 14,
        }
        self.architectures = ["MiniCPMO"]
        self.auto_map = {
            "AutoConfig": "configuration_minicpm_o_tiny.MiniCPMOConfig",
            "AutoModel": "modeling_minicpm_o_tiny.MiniCPMO",
            "AutoModelForCausalLM": "modeling_minicpm_o_tiny.MiniCPMO",
        }
"""

_MODEL_CODE = r"""
import types

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_minicpm_o_tiny import MiniCPMOConfig
except ImportError:
    from configuration_minicpm_o_tiny import MiniCPMOConfig


def prepare_inputs_for_generation(self, input_ids, **kwargs):
    return {"input_ids": input_ids, **kwargs}


class MiniCPMODecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn = torch.softmax(q @ k.transpose(-2, -1) / max(q.shape[-1], 1) ** 0.5, dim=-1)
        hidden_states = residual + self.o_proj(attn @ v)
        mlp = torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.norm(hidden_states + self.down_proj(mlp))


class MiniCPMOTinyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniCPMODecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        del attention_mask, position_ids, labels, kwargs
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(self.norm(hidden_states))
        return CausalLMOutputWithPast(logits=logits)


class MiniCPMO(PreTrainedModel):
    config_class = MiniCPMOConfig
    base_model_prefix = "llm"
    _no_split_modules = ["MiniCPMODecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.llm = MiniCPMOTinyLLM(config)
        self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)
        self.audio_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.code2wav = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def forward(self, *args, **kwargs):
        return self.llm(*args, **kwargs)
"""


def _write_remote_code(output_dir: str) -> None:
    with open(os.path.join(output_dir, "configuration_minicpm_o_tiny.py"), "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(_CONFIG_CODE).lstrip())
    with open(os.path.join(output_dir, "modeling_minicpm_o_tiny.py"), "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(_MODEL_CODE).lstrip())


def _load_remote_class(output_dir: str):
    sys.path.insert(0, output_dir)
    try:
        config_spec = importlib.util.spec_from_file_location(
            "configuration_minicpm_o_tiny",
            os.path.join(output_dir, "configuration_minicpm_o_tiny.py"),
        )
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_spec.name] = config_module
        config_spec.loader.exec_module(config_module)

        model_spec = importlib.util.spec_from_file_location(
            "modeling_minicpm_o_tiny",
            os.path.join(output_dir, "modeling_minicpm_o_tiny.py"),
        )
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules[model_spec.name] = model_module
        model_spec.loader.exec_module(model_module)
        return config_module.MiniCPMOConfig, model_module.MiniCPMO
    finally:
        sys.path.remove(output_dir)


def _build_tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "<image>", "<video>", "<audio>"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(
        [
            "Question Answer preferred rejected image video audio.",
            "<|im_start|>user\n<image>What is shown?<|im_end|>\n<|im_start|>assistant\nA dummy answer.<|im_end|>\n",
            "<|im_start|>user\n<audio>What sound is present?<|im_end|>\n",
        ],
        trainer=trainer,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        model_max_length=2048,
        chat_template=_CHAT_TEMPLATE,
        additional_special_tokens=["<image>", "<video>", "<audio>"],
    )


def build_checkpoint(output_dir: str, *, vocab_size: int = 2048, force: bool = False) -> None:
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if not force and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"MiniCPM-o tiny-random checkpoint already exists: {output_dir}", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)
    _write_remote_code(output_dir)
    config_cls, model_cls = _load_remote_class(output_dir)
    tokenizer = _build_tokenizer(vocab_size)
    config = config_cls(vocab_size=len(tokenizer))
    torch.manual_seed(0)
    model = model_cls(config)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _write_remote_code(output_dir)
    print(f"Wrote MiniCPM-o tiny-random checkpoint to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local tiny-random MiniCPM-o checkpoint.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_checkpoint(args.output_dir, vocab_size=args.vocab_size, force=args.force)


if __name__ == "__main__":
    main()
