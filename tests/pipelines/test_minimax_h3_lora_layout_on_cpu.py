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

"""Regression for native H3 QKV metadata preventing FC1 LoRA registration."""

from types import SimpleNamespace

import pytest
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MiniMaxH3RolloutWeightSyncMixin
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin


@pytest.mark.parametrize("mixin", [MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3WeightSyncMixin], ids=["nft", "flow"])
@pytest.mark.parametrize("partition", ["fl2va", "combined"])
def test_existing_native_qkv_mapping_is_extended_without_mutation(mixin, partition):
    native = MiniMaxH3DiTModel.stacked_params_mapping
    pipeline = mixin.__new__(mixin)
    pipeline.partition = partition
    pipeline.transformer = SimpleNamespace(stacked_params_mapping=native)
    pipeline.transformers_ref = SimpleNamespace(stacked_params_mapping=native)
    selected = (
        pipeline.transformers_ref
        if mixin is MiniMaxH3WeightSyncMixin and partition == "combined"
        else pipeline.transformer
    )
    untouched = pipeline.transformer if selected is pipeline.transformers_ref else pipeline.transformers_ref
    install = getattr(pipeline, "_install_lora_layout", None) or pipeline.install_h3_lora_layout
    install()
    mapping = selected.stacked_params_mapping
    normalized = {(packed.rsplit(".", 1)[-1], sub.rsplit(".", 1)[-1], shard) for packed, sub, shard in mapping}
    assert {
        ("qkv_proj", "to_q", "q"),
        ("qkv_proj", "to_k", "k"),
        ("qkv_proj", "to_v", "v"),
        ("fc1", "fc1_0", "0"),
        ("fc1", "fc1_1", "1"),
    } <= normalized
    assert list(mapping[: len(native)]) == list(native)
    assert len(normalized) == len(mapping)
    assert untouched.stacked_params_mapping is native
    assert MiniMaxH3DiTModel.stacked_params_mapping is native
    install()
    assert selected.stacked_params_mapping is mapping
