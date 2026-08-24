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
"""CPU tests for MiniMax H3 LoRA mapping."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    _LORA_STACKED_PARAMS_MAPPING,
    _LORA_VLLM_TARGET_MODULES,
    MiniMaxH3RolloutWeightSyncMixin,
)

FFN_HALF = 16
RANK = 4
HIDDEN = 8


def _make_mixin() -> MiniMaxH3RolloutWeightSyncMixin:
    mixin = MiniMaxH3RolloutWeightSyncMixin.__new__(MiniMaxH3RolloutWeightSyncMixin)
    mixin.transformer = SimpleNamespace(arch=SimpleNamespace(ffn_hidden_size=FFN_HALF))
    return mixin


def _trainer_lora_tensors(block: str = "transformer_blocks.3") -> dict[str, torch.Tensor]:
    """Build a PEFT-style LoRA state dict."""
    prefix = f"base_model.model.{block}"
    tensors = {}
    for mod, out_f in [
        ("attn.to_q", HIDDEN),
        ("attn.to_k", HIDDEN),
        ("attn.to_v", HIDDEN),
        ("attn.to_out.0", HIDDEN),
        ("ff.net.0.proj", 2 * FFN_HALF),
        ("ff.net.2", HIDDEN),
    ]:
        tensors[f"{prefix}.{mod}.lora_A.weight"] = torch.randn(RANK, HIDDEN)
        tensors[f"{prefix}.{mod}.lora_B.weight"] = torch.randn(out_f, RANK)
    return tensors


def _peft_config() -> dict:
    return {
        "r": RANK,
        "lora_alpha": 8,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    }


class TestLoraNameMapping:
    def test_all_targets_translated_to_vllm_paths(self):
        mapped, _ = _make_mixin().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        expected = {
            "transformer.blocks.3.attn.to_q",
            "transformer.blocks.3.attn.to_k",
            "transformer.blocks.3.attn.to_v",
            "transformer.blocks.3.attn.out_proj",
            "transformer.blocks.3.mlp.fc1_0",
            "transformer.blocks.3.mlp.fc1_1",
            "transformer.blocks.3.mlp.fc2",
        }
        modules = {name.rsplit(".lora_", 1)[0] for name in mapped}
        assert modules == expected

    def test_refiner_blocks_renamed(self):
        mapped, _ = _make_mixin().map_lora_update_to_engine(
            _trainer_lora_tensors(block="token_refiner.refiner_blocks.1"), _peft_config()
        )
        assert "transformer.token_refiner.blocks.1.attn.to_q.lora_A.weight" in mapped

    def test_prefixes_dropped(self):
        name = "_fsdp_wrapped_module.base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"
        tensors = {name: torch.randn(RANK, HIDDEN)}
        mapped, _ = _make_mixin().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["transformer.blocks.0.attn.to_q.lora_A.weight"]

    def test_fc1_lora_b_swapped_then_split_per_slice(self):
        tensors = _trainer_lora_tensors()
        key = "base_model.model.transformer_blocks.3.ff.net.0.proj.lora_B.weight"
        original = tensors[key]
        mapped, _ = _make_mixin().map_lora_update_to_engine(tensors, _peft_config())
        # vllm slice order is swapped vs diffusers: slice 0 = diffusers' second half.
        assert torch.equal(mapped["transformer.blocks.3.mlp.fc1_0.lora_B.weight"], original[FFN_HALF:])
        assert torch.equal(mapped["transformer.blocks.3.mlp.fc1_1.lora_B.weight"], original[:FFN_HALF])

    def test_fc1_lora_a_shared_unswapped_and_fc2_untouched(self):
        tensors = _trainer_lora_tensors()
        a_key = "base_model.model.transformer_blocks.3.ff.net.0.proj.lora_A.weight"
        b_key = "base_model.model.transformer_blocks.3.ff.net.2.lora_B.weight"
        mapped, _ = _make_mixin().map_lora_update_to_engine(tensors, _peft_config())
        assert torch.equal(mapped["transformer.blocks.3.mlp.fc1_0.lora_A.weight"], tensors[a_key])
        assert torch.equal(mapped["transformer.blocks.3.mlp.fc1_1.lora_A.weight"], tensors[a_key])
        assert torch.equal(mapped["transformer.blocks.3.mlp.fc2.lora_B.weight"], tensors[b_key])

    def test_target_modules_rewritten_to_vllm_names(self):
        _, config = _make_mixin().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        assert config["target_modules"] == _LORA_VLLM_TARGET_MODULES
        assert config["r"] == RANK  # other fields untouched

    @pytest.mark.parametrize(
        "target_modules",
        [
            "all-linear",
            ["to_q", "adaln_proj.linear"],
            ["to_q", "proj_in"],
        ],
    )
    def test_rejects_targets_not_transportable_by_layered_summon(self, target_modules):
        config = {**_peft_config(), "target_modules": target_modules}
        with pytest.raises(ValueError, match="all-linear|unsupported targets"):
            _make_mixin().map_lora_update_to_engine({}, config)

    def test_non_lora_names_pass_through(self):
        tensors = {"some_unrelated.weight": torch.randn(2, 2)}
        mapped, _ = _make_mixin().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["some_unrelated.weight"]


class TestVllmManagerInterplay:
    """Guard against the original failure: zero module matches on the vllm DiT."""

    @pytest.fixture()
    def match(self):
        pytest.importorskip("vllm_omni")
        from vllm_omni.diffusion.lora.utils import _match_target_modules

        return _match_target_modules

    # Leaf module names as they appear on the fused vllm MiniMaxH3DiTModel.
    VLLM_MODULES = [
        "transformer.blocks.0.attn.qkv_proj",
        "transformer.blocks.0.attn.out_proj",
        "transformer.blocks.0.mlp.fc1",
        "transformer.blocks.0.mlp.fc2",
        "transformer.token_refiner.blocks.0.attn.qkv_proj",
    ]

    def test_original_targets_match_nothing_on_vllm(self, match):
        original_targets = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
        for module in self.VLLM_MODULES:
            assert not match(module, original_targets)

    def test_rewritten_targets_match_vllm_modules(self, match):
        # qkv_proj and fc1 match via the packed-sublayer fallback (tested separately below).
        assert match("transformer.blocks.0.attn.out_proj", _LORA_VLLM_TARGET_MODULES)
        assert match("transformer.blocks.0.mlp.fc2", _LORA_VLLM_TARGET_MODULES)

    def test_packed_sublayer_names_match(self, match):
        # The manager's packed fallback checks prefix + sub-suffix from the mapping.
        sub_suffixes = [sub.strip(".").split(".")[-1] for _, sub, _ in _LORA_STACKED_PARAMS_MAPPING]
        assert sub_suffixes == ["to_q", "to_k", "to_v", "fc1_0", "fc1_1"]
        for prefix, subs in (("attn", ["to_q", "to_k", "to_v"]), ("mlp", ["fc1_0", "fc1_1"])):
            for sub in subs:
                assert match(f"transformer.blocks.0.{prefix}.{sub}", _LORA_VLLM_TARGET_MODULES)

    def test_stacked_mapping_declares_packed_slices(self):
        packed = [packed for packed, _, _ in _LORA_STACKED_PARAMS_MAPPING]
        shard_ids = [shard for _, _, shard in _LORA_STACKED_PARAMS_MAPPING]
        assert packed == [".qkv_proj"] * 3 + [".fc1"] * 2
        assert shard_ids == ["q", "k", "v", "0", "1"]


class TestInstallLoraLayout:
    def test_sets_stacked_params_mapping_once(self):
        mixin = _make_mixin()
        mixin._install_lora_layout()
        assert mixin.transformer.stacked_params_mapping == _LORA_STACKED_PARAMS_MAPPING
        # Idempotent: a pre-existing mapping is not clobbered.
        mixin.transformer.stacked_params_mapping = ["custom"]
        mixin._install_lora_layout()
        assert mixin.transformer.stacked_params_mapping == ["custom"]
