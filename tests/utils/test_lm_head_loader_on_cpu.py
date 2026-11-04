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
"""CPU tests for the config-driven teacher lm_head loader."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from verl_omni.utils.lm_head import (
    _discover_weight_map,
    _filter_head_keys,
    load_lm_head_weight,
)


def _llama_config(vocab_size=6, hidden_size=2, num_attention_heads=2, tie_word_embeddings=False):
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "tie_word_embeddings": tie_word_embeddings,
    }


def _write_model(
    tmp_path,
    weights: dict[str, torch.Tensor],
    shard_by_key: bool = False,
    config: dict | None = None,
    single_file_name: str = "model.safetensors",
):
    if config is not None:
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)
    if not shard_by_key:
        save_file({k: v.contiguous() for k, v in weights.items()}, os.path.join(tmp_path, single_file_name))
        return
    weight_map = {}
    for i, (k, v) in enumerate(weights.items()):
        shard = f"model-{i:05d}-of-{len(weights):05d}.safetensors"
        save_file({k: v.contiguous()}, os.path.join(tmp_path, shard))
        weight_map[k] = shard
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)


class TestDiscoverWeightMap:
    def test_single_file_lists_all_keys(self, tmp_path):
        _write_model(tmp_path, {"thinker.lm_head.weight": torch.randn(5, 3), "other.weight": torch.randn(4, 4)})
        assert set(_discover_weight_map(str(tmp_path))) == {"thinker.lm_head.weight", "other.weight"}

    def test_single_file_nonstandard_name(self, tmp_path):
        _write_model(tmp_path, {"lm_head.weight": torch.randn(5, 3)}, single_file_name="merged.safetensors")
        assert set(_discover_weight_map(str(tmp_path))) == {"lm_head.weight"}

    def test_sharded_index_resolves_shard(self, tmp_path):
        _write_model(
            tmp_path,
            {"thinker.lm_head.weight": torch.randn(5, 3), "other.weight": torch.randn(4, 4)},
            shard_by_key=True,
        )
        weight_map = _discover_weight_map(str(tmp_path))
        assert weight_map["thinker.lm_head.weight"].endswith("-00000-of-00002.safetensors")
        assert weight_map["other.weight"].endswith("-00001-of-00002.safetensors")


class TestFilterHeadKeys:
    def test_untied_filters_lm_head_keys(self):
        keys = _filter_head_keys(
            {"lm_head.weight": "a", "thinker.lm_head.weight": "b", "model.embed_tokens.weight": "c"},
            config={},
        )
        assert set(keys) == {"lm_head.weight", "thinker.lm_head.weight"}

    def test_tied_filters_embed_keys(self):
        keys = _filter_head_keys(
            {"model.embed_tokens.weight": "a", "lm_head.weight": "b"},
            config={"tie_word_embeddings": True},
        )
        assert keys == ["model.embed_tokens.weight"]


class TestLoadLmHeadWeight:
    def test_tied_picks_thinker_embed_by_shape(self, tmp_path):
        # Reviewer scenario: talker embed also matches the suffix; only the
        # thinker embed matches the text config's (vocab_size, hidden_size).
        thinker = torch.randn(6, 2)
        talker = torch.randn(4, 2)
        _write_model(
            tmp_path,
            {
                "thinker.model.embed_tokens.weight": thinker,
                "talker.model.embed_tokens.weight": talker,
            },
            config=_llama_config(vocab_size=6, hidden_size=2, tie_word_embeddings=True),
        )
        out = load_lm_head_weight(str(tmp_path))
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"
        assert torch.equal(out, thinker.float())

    def test_untied_prefers_lm_head(self, tmp_path):
        w = torch.randn(6, 2)
        _write_model(
            tmp_path,
            {"lm_head.weight": w, "model.embed_tokens.weight": torch.randn(6, 2)},
            config=_llama_config(),
        )
        assert torch.equal(load_lm_head_weight(str(tmp_path)), w.float())

    def test_sharded_checkpoint(self, tmp_path):
        w = torch.randn(6, 2)
        _write_model(
            tmp_path,
            {"thinker.lm_head.weight": w, "other.weight": torch.randn(4, 4)},
            shard_by_key=True,
            config=_llama_config(),
        )
        assert torch.equal(load_lm_head_weight(str(tmp_path)), w.float())

    def test_explicit_keys_skip_inference(self, tmp_path):
        w = torch.randn(5, 3)
        _write_model(tmp_path, {"custom.head.weight": w, "lm_head.weight": torch.randn(9, 3)})
        out = load_lm_head_weight(str(tmp_path), keys=("custom.head.weight",))
        assert torch.equal(out, w.float())

    def test_shape_mismatch_raises(self, tmp_path):
        _write_model(
            tmp_path,
            {"thinker.lm_head.weight": torch.randn(7, 2)},
            config=_llama_config(vocab_size=6, hidden_size=2),
        )
        with pytest.raises(ValueError, match="text config"):
            load_lm_head_weight(str(tmp_path))
