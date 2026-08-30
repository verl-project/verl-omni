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
"""CPU tests for generic LoRA adapter lifecycle helpers."""

from types import SimpleNamespace

import torch
from diffusers.loaders import PeftAdapterMixin

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


class _BareLinearBlocks(torch.nn.Module):
    """A ``blocks.*`` module that predates PEFT (no ``add_adapter`` of its own)."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        with torch.no_grad():
            self.blocks[0].weight.zero_()

    def forward(self, x):
        return self.blocks[0](x)


def _peft_capable_blocks() -> torch.nn.Module:
    """Return a PEFT-capable stand-in that keeps ``blocks.*`` names."""
    cls = type("BlocksWithPeft", (_BareLinearBlocks, PeftAdapterMixin), {})
    return cls()


class _Harness(LoRAAdapterMixin):
    def __init__(self, *, policy_state_adapters=("default", "old")):
        self.model_config = SimpleNamespace(
            lora_rank=1,
            lora_alpha=2,
            lora_init_weights="gaussian",
            target_modules="all-linear",
            target_parameters=None,
            exclude_modules=None,
            lora_adapter_path=None,
            policy_state_adapters=policy_state_adapters,
            lora_dtype=None,
            use_shm=False,
        )
        self._is_offload_param = False
        self.module = None


def _fill_default_lora(module: torch.nn.Module, value: float = 1.0) -> None:
    module.set_adapter("default")
    with torch.no_grad():
        for name, param in module.named_parameters():
            if ".lora_A.default." in name or ".lora_B.default." in name:
                param.fill_(value)


def test_build_lora_module_adapts_peft_capable_module_in_place_keeping_block_names():
    harness = _Harness(policy_state_adapters=("default", "old"))

    bare = _peft_capable_blocks()
    module = harness._build_lora_module(bare)

    # Adapted in place (not re-wrapped in a PeftModel): the returned object is
    # the same module, so FSDP wrapping and the module tree are untouched.
    assert module is bare
    assert set(module.peft_config) == {"default", "old"}
    assert module.active_adapters() == ["default"]

    # No ``base_model.model.`` prefix — the ``blocks.*`` names are preserved so
    # LoRA weight export matches the other diffusers transformers.
    trainable_names = [name for name, param in module.named_parameters() if param.requires_grad]
    assert trainable_names == [
        "blocks.0.lora_A.default.weight",
        "blocks.0.lora_B.default.weight",
    ]

    module.set_adapter("old")
    old_trainable_names = [name for name, param in module.named_parameters() if param.requires_grad]
    assert old_trainable_names == [
        "blocks.0.lora_A.old.weight",
        "blocks.0.lora_B.old.weight",
    ]


def test_disable_adapter_supports_generic_peft_disable_adapter_context():
    harness = _Harness(policy_state_adapters=("default",))
    module = harness._build_lora_module(_peft_capable_blocks())
    harness.module = module
    _fill_default_lora(module)

    x = torch.ones(1, 2)
    enabled = module(x)
    with harness.disable_adapter():
        disabled = module(x)
    restored = module(x)

    assert torch.allclose(enabled, torch.full((1, 2), 4.0))
    assert torch.allclose(disabled, torch.zeros(1, 2))
    assert torch.allclose(restored, enabled)
