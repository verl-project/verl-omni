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
"""CPU tests for diffusion FSDP LoRA param collection."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from verl_omni.utils.fsdp_utils import collect_lora_params


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.to_q(x)


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, x):
        return self.transformer_blocks[0](x)


def _peft_dit():
    model = get_peft_model(
        _TinyDiT(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"),
    )
    model.eval()
    return model


def test_fsdp2_non_layered_collects_lora_without_child_units(monkeypatch):
    """SD3.5 v1 uses FSDP2 + layered_summon=False; child-unit PEFT lookup is empty."""
    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(fsdp_utils, "_iter_fsdp2_submodules", lambda _: iter(()))

    params = collect_lora_params(
        module,
        layered_summon=False,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert any("lora_" in name for name in params)
    assert all(".default." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_layered_diffusers_falls_back_when_prefix_walker_is_empty(monkeypatch):
    """Qwen-Image e2e: layered_summon=True; prefix walker skips non-block FSDP units."""
    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()

    def _version(m):
        return 2 if m is module else 0

    monkeypatch.setattr(fsdp_utils, "fsdp_version", _version)

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert any("lora_" in name for name in params)
    assert all(".default." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_layered_collects_fsdp_leaf_lora_when_peft_dump_is_empty(monkeypatch):
    """Qwen-Image-Edit: named targets + FSDP1 leaf wrap; PEFT prefix dump is {}."""
    from collections import OrderedDict
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    class _Adapter(nn.Module):
        def __init__(self, fill: float):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2, 4), fill))

    class _LoraA(nn.Module):
        def __init__(self):
            super().__init__()
            self.default = _Adapter(1.0)
            self.old = _Adapter(2.0)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = _LoraA()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([_Block()])

    module = _Tiny()

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda m: 1 if m is module else 0)
    monkeypatch.setattr(fsdp_utils, "_peft_lora_params_to_cpu", lambda *args, **kwargs: OrderedDict())
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert list(params) == ["transformer_blocks.0.attn.lora_A.weight"]
    torch.testing.assert_close(params["transformer_blocks.0.attn.lora_A.weight"], torch.ones(2, 4))


def test_layered_diffusers_falls_back_for_frozen_non_default_adapter(monkeypatch):
    """DiffusionNFT syncs a frozen "old" adapter; PEFT's own state-dict lookup misses it."""
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    module.add_adapter("old", LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"))

    def _version(m):
        return 2 if m is module else 0

    monkeypatch.setattr(fsdp_utils, "fsdp_version", _version)
    # Simulate PEFT's adapter-name lookup missing the frozen adapter's tensors even
    # though they are present in the submodule's own state dict / named_parameters.
    monkeypatch.setattr(fsdp_utils, "get_peft_model_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
        adapter_name="old",
    )
    assert any("lora_" in name and ".old." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_collect_strips_nested_fsdp_wrapper_tokens_for_vllm(monkeypatch):
    """Qwen-Image e2e: nested FSDP wraps leak into PEFT dump keys; vLLM rejects them."""
    from collections import OrderedDict

    import verl_omni.utils.fsdp_utils as fsdp_utils

    dirty = "transformer_blocks.0._fsdp_wrapped_module.attn.to_q.lora_A.default._fsdp_wrapped_module.weight"
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 1)
    monkeypatch.setattr(
        fsdp_utils,
        "_layered_summon_lora_params_diffusers",
        lambda *args, **kwargs: OrderedDict({dirty: torch.ones(2, 4)}),
    )

    params = collect_lora_params(
        nn.Linear(4, 4, bias=False),
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert list(params) == ["transformer_blocks.0.attn.to_q.lora_A.weight"]
    assert all("_fsdp_wrapped_module" not in name for name in params)
    assert all(".default." not in name for name in params)


class _FakeFsdp1Wrap(nn.Module):
    """Nested FSDP1 leaf stand-in: module tree keeps ``_fsdp_wrapped_module``.

    ``state_dict()`` keys drop the wrapper token (real FSDP1 leaf state dicts do);
    ``named_parameters()`` keeps it — the asymmetry the layered collector must bridge.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self._fsdp_wrapped_module = block

    def state_dict(self, *args, **kwargs):
        return {
            name.replace("_fsdp_wrapped_module.", ""): tensor
            for name, tensor in self._fsdp_wrapped_module.state_dict(*args, **kwargs).items()
        }


def _inplace_lora_dit(num_blocks: int = 2) -> nn.Module:
    """Two-block DiT with in-place LoRA injection, per-block filled adapters."""
    from peft import LoraConfig as _LoraConfig
    from peft import inject_adapter_in_model

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4, bias=False)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    class _DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(_FakeFsdp1Wrap(_Block()) for _ in range(num_blocks))

    model = _DiT()
    inject_adapter_in_model(_LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"), model)
    for i, block in enumerate(model.transformer_blocks):
        block._fsdp_wrapped_module.attn.to_q.lora_A["default"].weight.data.fill_(float(i + 1))
        block._fsdp_wrapped_module.attn.to_q.lora_B["default"].weight.data.fill_(float(i + 11))
    return model


def test_layered_fsdp1_collect_keeps_block_prefix_per_peft_prefix_rules(monkeypatch):
    """peft >= 0.22 derives tuner prefixes from wrapper-free module paths; the
    per-block state dict must be keyed the same way or every block falls to the
    block-local fallback and later blocks silently overwrite earlier ones
    (the gpu_smoke ``LoRA adapter binding is incomplete`` regression)."""
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _inplace_lora_dit()
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 1)
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )

    assert sorted(params) == [
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer_blocks.1.attn.to_q.lora_A.weight",
        "transformer_blocks.1.attn.to_q.lora_B.weight",
    ]
    torch.testing.assert_close(params["transformer_blocks.0.attn.to_q.lora_A.weight"], torch.full((2, 4), 1.0))
    torch.testing.assert_close(params["transformer_blocks.1.attn.to_q.lora_A.weight"], torch.full((2, 4), 2.0))
    torch.testing.assert_close(params["transformer_blocks.0.attn.to_q.lora_B.weight"], torch.full((4, 2), 11.0))
    torch.testing.assert_close(params["transformer_blocks.1.attn.to_q.lora_B.weight"], torch.full((4, 2), 12.0))


def test_layered_fsdp1_fallback_names_carry_block_prefix(monkeypatch):
    """When PEFT's own state-dict lookup misses, the per-block fallback must still
    emit root-relative names; block-local names collide across blocks."""
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _inplace_lora_dit()
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 1)
    monkeypatch.setattr(fsdp_utils, "get_peft_model_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )

    assert sorted(params) == [
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer_blocks.1.attn.to_q.lora_A.weight",
        "transformer_blocks.1.attn.to_q.lora_B.weight",
    ]
    torch.testing.assert_close(params["transformer_blocks.1.attn.to_q.lora_A.weight"], torch.full((2, 4), 2.0))
