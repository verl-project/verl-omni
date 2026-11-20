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

import pytest
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


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("adapter_name", ["default", "old"])
@pytest.mark.parametrize("empty_peft_dump", [False, True])
def test_layered_lora_export_preserves_all_block_paths(monkeypatch, version, adapter_name, empty_peft_dump):
    """Block-local fallback keys must retain their parent path and selected adapter."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from peft import get_peft_model_state_dict, inject_adapter_in_model

    import verl_omni.utils.fsdp_utils as fsdp_utils
    from verl_omni.workers.engine.lora_export import export_lora_for_sync

    model = _TinyDiT()
    model.transformer_blocks.append(_Block())
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none")
    inject_adapter_in_model(config, model)
    inject_adapter_in_model(config, model, adapter_name="old")
    with torch.no_grad():
        for index, (_, parameter) in enumerate(model.named_parameters()):
            parameter.fill_(index + 1)
    expected = get_peft_model_state_dict(model, adapter_name=adapter_name)
    assert len(expected) == 4

    class Wrapped(nn.Module):
        def __init__(self, module):
            super().__init__()
            self._fsdp_wrapped_module = module

        def state_dict(self, *args, **kwargs):
            return self._fsdp_wrapped_module.state_dict(*args, **kwargs)

    if version == 1:
        for index, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[index] = Wrapped(block)
        module = Wrapped(model)
        units = [module, *model.transformer_blocks]
    else:
        module = model
        units = [model, *model.transformer_blocks]
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda item: version if any(item is unit for unit in units) else 0)
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )
    if empty_peft_dump:
        monkeypatch.setattr(fsdp_utils, "get_peft_model_state_dict", lambda *args, **kwargs: {})

    stream, peft_config = export_lora_for_sync(
        module,
        model_config=SimpleNamespace(lora={"merge": False}, fsdp_layer_prefixes=["transformer_blocks."]),
        base_sync_done=True,
        adapter_name=adapter_name,
        layered_summon=True,
        is_diffusers=True,
        key_prefix="transformer.",
    )
    actual = dict(stream)
    expected = {f"transformer.{name}": tensor for name, tensor in expected.items()}
    print("LAYERED_LORA_ASSERT_ENTERED", version, adapter_name, empty_peft_dump, flush=True)
    assert set(actual) == set(expected), "block-local fallback must not lose or overwrite layer paths"
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)

    if version == 1 and adapter_name == "old" and empty_peft_dump:
        from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension

        class Receiver:
            def __init__(self, **kwargs):
                pass

            def receive_weights(self, on_bucket_received):
                weights = list(actual.items())
                on_bucket_received(weights[:2])
                on_bucket_received(weights[2:])

        requests = []
        monkeypatch.setattr(
            "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer.BucketedWeightReceiver", Receiver
        )
        worker = SimpleNamespace(
            device="cpu",
            _get_zmq_handle=lambda: "unused",
            _get_standard_weight_model_and_config=lambda: None,
            remove_lora=lambda _: None,
            add_lora=requests.append,
        )
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, peft_config=peft_config, base_sync_done=True)
        assert len(requests) == 1
        assert set(requests[0].lora_tensors) == set(expected)
        for name, tensor in expected.items():
            torch.testing.assert_close(requests[0].lora_tensors[name], tensor, rtol=0, atol=0)
