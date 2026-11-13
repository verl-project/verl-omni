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
"""CPU tests for FSDP2 LoRA training with the frozen base replicated (``replicate_frozen_base``)."""

import copy
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        return self.norm(self.to_q(x))


class _TinyDiT(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _lora_dit():
    torch.manual_seed(0)
    model = get_peft_model(_TinyDiT(), LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.1)
    return model


def _replicate_rank(rank: int, root: str, perturb_rank1: bool) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from verl_omni.utils.fsdp_utils import apply_fsdp2
    from verl_omni.workers.engine.fsdp.diffusers_impl import _check_replicated_params

    dist.init_process_group("gloo", init_method=f"file://{Path(root) / 'rendezvous'}", rank=rank, world_size=2)
    try:
        model = _lora_dit()
        if perturb_rank1:
            if rank == 1:
                with torch.no_grad():
                    model.base_model.model.transformer_blocks[1].to_q.base_layer.weight[0, 0] += 1e-3
            with pytest.raises(RuntimeError, match="differ across ranks"):
                _check_replicated_params(model)
            return

        reference = copy.deepcopy(model)
        _check_replicated_params(model)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
        apply_fsdp2(model, {"mesh": mesh}, {}, replicate_frozen=True)

        for name, param in model.named_parameters():
            assert isinstance(param, DTensor) == ("lora_" in name), name

        x = torch.randn(3, 4, generator=torch.Generator().manual_seed(1))
        out = model(x)
        expected = reference(x)
        assert torch.allclose(out, expected, atol=1e-6)
        out.square().sum().backward()
        expected.square().sum().backward()
        for (name, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            if "lora_" in name:
                assert torch.allclose(param.grad.full_tensor(), ref_param.grad, atol=1e-6), name
            else:
                assert param.grad is None, name
    finally:
        dist.destroy_process_group()


class TestReplicateFrozen:
    @pytest.mark.parametrize("perturb_rank1", [False, True], ids=["identical-trains", "rank-mismatch-raises"])
    def test_two_ranks(self, tmp_path, perturb_rank1):
        mp.spawn(_replicate_rank, args=(str(tmp_path), perturb_rank1), nprocs=2, join=True)

    def test_trainable_base_parameter_is_rejected(self):
        from verl_omni.utils.fsdp_utils import apply_fsdp2

        model = _lora_dit()
        model.base_model.model.transformer_blocks[0].norm.weight.requires_grad_(True)
        with pytest.raises(ValueError, match="every non-LoRA parameter to be frozen"):
            apply_fsdp2(model, {"mesh": None}, {}, replicate_frozen=True)

    @pytest.mark.parametrize(
        "param_dtype,expected_base",
        [(torch.bfloat16, torch.bfloat16), (None, torch.float32)],
        ids=["cast-to-compute-dtype", "fp32-islands-kept"],
    )
    def test_base_is_stored_in_the_compute_dtype(self, param_dtype, expected_base):
        from verl_omni.workers.engine.fsdp.diffusers_impl import _cast_replicated_params

        model = _lora_dit()
        _cast_replicated_params(model, param_dtype)
        for name, param in model.named_parameters():
            assert param.dtype == (torch.float32 if "lora_" in name else expected_base), name

    @pytest.mark.parametrize(
        "adapter_on,lora_rank,strategy,offload_policy,expected",
        [
            (False, 8, "fsdp2", False, False),
            (True, 0, "fsdp2", False, False),
            (True, 8, "fsdp2", False, True),
            (True, 8, "fsdp2", True, False),
            (True, 8, "fsdp", False, ValueError),
        ],
        ids=["adapter-off", "no-lora", "enabled", "cpu-offload-falls-back", "fsdp1-rejected"],
    )
    def test_enable_gate(self, monkeypatch, adapter_on, lora_rank, strategy, offload_policy, expected):
        from verl_omni.workers.engine.fsdp import diffusers_impl

        adapter = types.SimpleNamespace(replicate_frozen_base=lambda model_config: adapter_on)
        monkeypatch.setattr(diffusers_impl.DiffusionModelBase, "get_class", lambda model_config: adapter)
        monkeypatch.setattr(
            diffusers_impl, "resolve_lora_config", lambda model_config: types.SimpleNamespace(merge=False)
        )
        engine = types.SimpleNamespace(
            model_config=types.SimpleNamespace(lora_rank=lora_rank),
            engine_config=types.SimpleNamespace(
                strategy=strategy, param_offload=False, offload_policy=offload_policy, forward_only=False
            ),
        )
        if expected is ValueError:
            with pytest.raises(ValueError, match="strategy=fsdp2"):
                diffusers_impl._replicate_frozen_enabled(engine)
        else:
            assert diffusers_impl._replicate_frozen_enabled(engine) is expected
