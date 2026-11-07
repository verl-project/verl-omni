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
"""GPU regression tests for layered LoRA collection from nested FSDP units."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from verl.utils.fsdp_utils import fsdp_version

from verl_omni.utils.fsdp_utils import collect_lora_params

_BOOGU_LAYER_PREFIXES = (
    "double_stream_layers.",
    "single_stream_layers.",
    "context_refiner.",
    "noise_refiner.",
    "ref_image_refiner.",
)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP LoRA collection tests.")
    if dist.is_initialized():
        yield
        return

    torch.cuda.set_device(0)
    rendezvous = Path(tmp_path_factory.mktemp("fsdp_lora_dist")) / "init"
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _build_tiny_boogu_model():
    boogu_transformer = pytest.importorskip("boogu.models.transformers.transformer_boogu")
    model = boogu_transformer.BooguImageTransformer2DModel(
        hidden_size=64,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        multiple_of=16,
        axes_dim_rope=(4, 4, 8),
        instruction_feature_configs={
            "instruction_feat_dim": 32,
            "reduce_type": "mean",
            "num_instruction_feature_layers": 1,
        },
        prompt_tuning_configs={"use_prompt_tuning": False},
    )
    model.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"]))
    return model


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_layered_collection_matches_peft_keys_for_nested_fsdp_units(strategy):
    model = _build_tiny_boogu_model().to(torch.cuda.current_device())
    block_types = {
        type(model.double_stream_layers[0]),
        type(model.single_stream_layers[0]),
        type(model.context_refiner[0]),
        type(model.noise_refiner[0]),
        type(model.ref_image_refiner[0]),
    }
    if strategy == "fsdp":
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy(block_types),
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
        )
    else:
        for submodule in model.modules():
            if type(submodule) in block_types:
                fully_shard(submodule)
        fully_shard(model)
        fsdp_model = model

    assert fsdp_version(fsdp_model) == (1 if strategy == "fsdp" else 2)

    params = collect_lora_params(
        fsdp_model,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
        layer_prefixes=_BOOGU_LAYER_PREFIXES,
    )

    assert params
    assert all("lora_" in name for name in params)
    assert all("_fsdp_wrapped_module" not in name for name in params)
    for prefix in _BOOGU_LAYER_PREFIXES:
        assert any(name.startswith(prefix) for name in params), prefix
