#!/usr/bin/env python3
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
"""Build a tiny random 16-codebook Qwen3-TTS checkpoint for GPU smoke tests."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def _copy_files(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    for name in names:
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, destination / name)


def _save_model(model, output_dir: Path) -> None:
    state_dict = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}
    save_file(state_dict, output_dir / "model.safetensors", metadata={"format": "pt"})


def _write_config(source: Path, destination: Path, updates) -> None:
    config = json.loads(source.read_text(encoding="utf-8"))
    updates(config)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _scaled_mrope_section(section: list[int], head_dim: int) -> list[int]:
    target = head_dim // 2
    total = sum(section)
    scaled = [value * target // total for value in section]
    scaled[-1] += target - sum(scaled)
    return scaled


def _set_model_codebooks(config: dict) -> None:
    talker_config = config["talker_config"]
    talker_config["num_code_groups"] = 16
    talker_config["code_predictor_config"]["num_code_groups"] = 16
    rope_scaling = talker_config["rope_scaling"]
    rope_scaling["mrope_section"] = _scaled_mrope_section(rope_scaling["mrope_section"], talker_config["head_dim"])


def _set_tokenizer_quantizers(config: dict) -> None:
    config["dtype"] = "bfloat16"
    config["encoder_valid_num_quantizers"] = 16
    config["encoder_config"]["num_quantizers"] = 16
    config["decoder_config"]["num_quantizers"] = 16


def build(source_model_path: Path, output_dir: Path, seed: int = 42) -> Path:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = Qwen3TTSConfig.from_pretrained(source_model_path)
    model_config.talker_config.num_code_groups = 16
    model_config.talker_config.code_predictor_config.num_code_groups = 16
    rope_scaling = model_config.talker_config.rope_scaling
    rope_scaling["mrope_section"] = _scaled_mrope_section(
        rope_scaling["mrope_section"], model_config.talker_config.head_dim
    )
    model = Qwen3TTSForConditionalGeneration(model_config).to(torch.bfloat16)
    _save_model(model, output_dir)
    _write_config(
        source_model_path / "config.json",
        output_dir / "config.json",
        _set_model_codebooks,
    )
    _copy_files(
        source_model_path,
        output_dir,
        ("generation_config.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"),
    )

    tokenizer_source = source_model_path / "speech_tokenizer"
    tokenizer_output = output_dir / "speech_tokenizer"
    tokenizer_output.mkdir(exist_ok=True)
    tokenizer_config = Qwen3TTSTokenizerV2Config.from_pretrained(tokenizer_source)
    tokenizer_config.encoder_valid_num_quantizers = 16
    tokenizer_config.encoder_config.num_quantizers = 16
    tokenizer_config.decoder_config.num_quantizers = 16
    tokenizer = Qwen3TTSTokenizerV2Model(tokenizer_config).to(torch.bfloat16)
    _save_model(tokenizer, tokenizer_output)
    _write_config(
        tokenizer_source / "config.json",
        tokenizer_output / "config.json",
        _set_tokenizer_quantizers,
    )
    _copy_files(tokenizer_source, tokenizer_output, ("configuration.json", "preprocessor_config.json"))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build(args.source_model_path, args.output_dir, args.seed))


if __name__ == "__main__":
    main()
