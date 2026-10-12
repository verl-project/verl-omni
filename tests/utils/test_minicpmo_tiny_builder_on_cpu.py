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
"""CPU tests for the tiny MiniCPM-o checkpoint builder's config shrink."""

import importlib.util
import sys
from pathlib import Path

import pytest

_BUILDER_PATH = Path(__file__).resolve().parents[2] / "tests/special_e2e/build_minicpmo_tiny_random.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("minicpmo_tiny_builder_under_test", _BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_config(**overrides) -> dict:
    """The source-checkpoint fields the shrink touches, plus merge siblings."""
    config = {
        "version": "4.5",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_window_layers": 36,
        "vocab_size": 151748,
        "query_num": 64,
        "stream_input": True,
        "init_tts": True,
        "audio_config": {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
        },
        "vision_config": {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "intermediate_size": 4304,
            "image_size": 980,
            "patch_size": 14,
        },
        "auto_map": {"AutoModel": "modeling_minicpmo.MiniCPMO"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def test_shrink_applies_overlays_and_keeps_siblings():
    builder = _load_builder()

    config = builder._shrink_config(_source_config())

    assert config["hidden_size"] == 256
    assert config["num_hidden_layers"] == 2
    assert config["init_tts"] is False
    # Sub-dicts merge; untouched siblings stay at source values.
    assert config["audio_config"]["num_mel_bins"] == 80
    assert config["audio_config"]["max_source_positions"] == 1500
    assert config["vision_config"]["hidden_size"] == 1152
    assert config["vision_config"]["image_size"] == 980
    # Pipeline routing and prompt-parity keys must survive the shrink.
    assert config["version"] == "4.5"
    assert config["stream_input"] is True
    assert config["query_num"] == 64
    assert config["auto_map"] == {"AutoModel": "modeling_minicpmo.MiniCPMO"}
    # The source dict is not mutated.
    assert _source_config()["hidden_size"] == 4096


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"hidden_size": 100}, "multiple of 128"),
        ({"num_attention_heads": 5}, "head_dim"),
        ({"audio_config": {"encoder_ffn_dim": 1025}}, "multiple of 4"),
        ({"audio_config": {"d_model": 250}}, "divisible by encoder_attention_heads"),
        ({"vision_config": {"image_size": 981}}, "divisible by patch_size"),
    ],
)
def test_validate_rejects_invariant_violations(override, match):
    """The cross-field invariants gate the finished config, not just the source."""
    builder = _load_builder()

    config = builder._shrink_config(_source_config())
    for key, value in override.items():
        if isinstance(value, dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value

    with pytest.raises(ValueError, match=match):
        builder._validate_tiny_config(config)


def test_shrink_output_passes_validation():
    """The unmodified shrink output satisfies every cross-field invariant."""
    builder = _load_builder()

    builder._validate_tiny_config(builder._shrink_config(_source_config()))


def test_shrink_fails_closed_on_missing_source_keys():
    builder = _load_builder()

    config = _source_config()
    del config["audio_config"]["encoder_ffn_dim"]
    with pytest.raises(KeyError, match="encoder_ffn_dim"):
        builder._shrink_config(config)
