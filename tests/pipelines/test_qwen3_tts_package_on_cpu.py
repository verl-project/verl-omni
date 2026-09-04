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
"""Check the pinned upstream qwen-tts TF5 source on the repository stack."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_smoke_helper(filename: str):
    path = Path(__file__).parents[1] / f"special_e2e/{filename}.py"
    spec = importlib.util.spec_from_file_location(f"{filename}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_speaker_dimension_matches_model_config(tmp_path):
    builder = _load_smoke_helper("create_dummy_qwen3_tts_grpo_data")
    model_config_path = tmp_path / "config.json"
    model_config_path.write_text(
        json.dumps(
            {
                "talker_config": {"hidden_size": 128},
                "speaker_encoder_config": {"enc_dim": 128},
            }
        ),
        encoding="utf-8",
    )

    assert builder._speaker_dimension(model_config_path) == 128

    model_config_path.write_text(
        json.dumps(
            {
                "talker_config": {"hidden_size": 128},
                "speaker_encoder_config": {"enc_dim": 1024},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="speaker encoder output must match"):
        builder._speaker_dimension(model_config_path)


def test_smoke_tiny_model_mrope_section_matches_head_dimension():
    builder = _load_smoke_helper("build_qwen3_tts_tiny_random")

    section = builder._scaled_mrope_section([24, 20, 20], head_dim=64)

    assert section == [12, 10, 10]
    assert sum(section) == 64 // 2


def test_qwen_tts_registers_and_runs_without_a_transformers_compatibility_layer(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    if importlib.util.find_spec("qwen_tts") is None:
        pytest.skip("qwen-tts is an optional dependency")

    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers import AutoConfig, AutoModelForTextToWaveform

    assert int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
    AutoModelForTextToWaveform.register(
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        exist_ok=True,
    )
    assert AutoConfig.for_model("qwen3_tts").__class__ is Qwen3TTSConfig
    assert AutoModelForTextToWaveform._model_mapping[Qwen3TTSConfig] is Qwen3TTSForConditionalGeneration

    predictor = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "num_code_groups": 16,
        "layer_types": ["full_attention"],
        "pad_token_id": None,
    }
    talker = {
        "code_predictor_config": predictor,
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "num_code_groups": 16,
        "text_hidden_size": 8,
        "text_vocab_size": 80,
        "codec_eos_token_id": 50,
        "codec_nothink_id": 51,
        "codec_think_bos_id": 52,
        "codec_think_eos_id": 53,
        "codec_pad_id": 54,
        "codec_bos_id": 55,
        "spk_id": {},
        "codec_language_id": {},
        "rope_scaling": {
            "rope_type": "default",
            "type": "default",
            "mrope_section": [1, 1, 0],
            "interleaved": True,
        },
    }
    config = Qwen3TTSConfig(
        talker_config=talker,
        speaker_encoder_config={},
        tts_model_type="custom",
        tokenizer_type="12hz",
        tts_pad_token_id=60,
        tts_bos_token_id=61,
        tts_eos_token_id=62,
    )
    model = Qwen3TTSForConditionalGeneration(config)
    output = model.talker(
        inputs_embeds=torch.randn(2, 5, 8),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        use_cache=False,
        output_hidden_states=False,
    )

    assert output.logits.shape == (2, 5, 64)

    import vllm_omni.platforms as platforms
    from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

    monkeypatch.setattr(platforms, "_current_omni_platform", UnspecifiedOmniPlatform())
    from verl_omni.pipelines.qwen3_tts.talker_forward import (
        TalkerTokens,
        build_talker_batch,
        codec0_input_embeddings,
    )

    codes = torch.randint(1, 31, (3, 16), dtype=torch.long)
    batch = build_talker_batch(
        [torch.tensor([1, 2, 3])],
        [codes],
        TalkerTokens.from_config(model.config),
        sub_codebook_vocab=32,
    )
    embeddings = codec0_input_embeddings(model.talker, batch, torch.zeros(1, 8))
    assert embeddings.shape == (*batch.input_ids.shape[:2], 8)
    assert torch.isfinite(embeddings).all()

    from verl_omni.pipelines.qwen3_tts.talker_training_adapter import Qwen3TTSTalkerAdapter

    speaker_path = tmp_path / "speaker.json"
    speaker_path.write_text(json.dumps([0.0] * 8), encoding="utf-8")
    configured = Qwen3TTSTalkerAdapter.configure_model(
        model,
        SimpleNamespace(
            use_remove_padding=False,
            override_config={"tts_spk_embed_path": str(speaker_path), "tts_language": "Auto"},
        ),
    )
    trainable_names = {name for name, parameter in configured.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all(name.startswith(("talker.model.", "talker.codec_head.")) for name in trainable_names)
    assert any(not parameter.requires_grad for parameter in configured.parameters())
    assert configured.get_input_embeddings() is configured.talker.model.codec_embedding
