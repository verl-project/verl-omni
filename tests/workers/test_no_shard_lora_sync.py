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
"""Single-GPU regression for FSDP1 LoRA weight synchronization."""

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl_omni.utils.fsdp_utils import collect_lora_params


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_no_shard_repeated_lora_sync(tmp_path):
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        model = get_peft_model(Tiny().cuda(), LoraConfig(r=2, target_modules=["proj"]))
        wrapped = FSDP(model, device_id=0, use_orig_params=True)
        print("strategy:", wrapped.sharding_strategy, flush=True)
        params = collect_lora_params(wrapped, layered_summon=True, base_sync_done=True)
        assert params and all(v.device.type == "cpu" and torch.isfinite(v).all() for v in params.values())
        before = {k: v.clone() for k, v in params.items()}
        optimizer = torch.optim.SGD((p for p in wrapped.parameters() if p.requires_grad), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            loss = wrapped(torch.randn(2, 8, device="cuda")).square().mean()
            loss.backward()
            optimizer.step()
            params = collect_lora_params(wrapped, layered_summon=True, base_sync_done=True)
            expected = collect_lora_params(wrapped, layered_summon=False, base_sync_done=True)
            assert params.keys() == expected.keys()
            assert all(torch.equal(params[k], expected[k]) for k in params)
            assert all(torch.isfinite(v).all() for v in params.values())
        assert any(not torch.equal(before[k], params[k]) for k in params)
        print("PASS: two optimizer steps, finite CPU adapter snapshots, layered/non-layered exact equality", flush=True)
    finally:
        dist.destroy_process_group()
