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
"""Adapter-state updates must respect FSDP2-managed CPU placement."""

from contextlib import nullcontext

import pytest
import torch

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


class AdapterModule(torch.nn.Module):
    """Model two named adapters with small CPU tensors."""

    def __init__(self):
        """Initialize the source adapter and an empty old-policy adapter."""
        super().__init__()
        self.default = torch.nn.Parameter(torch.full((2, 4), 3.0))
        self.old = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        self.active = "default"

    def set_adapter(self, name):
        """Select the active adapter and update its trainable parameters."""
        self.active = name
        self.default.requires_grad_(name == "default")
        self.old.requires_grad_(name == "old")


class Engine(LoRAAdapterMixin):
    """Expose adapter lifecycle helpers without creating a distributed engine."""

    def __init__(self, policy):
        """Set the offload policy, leaving the attribute absent for legacy engines."""
        self.module = AdapterModule()
        self._is_offload_param = False
        if policy is not None:
            self._uses_fsdp2_cpu_offload_policy = policy


@pytest.mark.parametrize("policy", [None, False, True])
def test_copy_and_ema_respect_cpu_offload_policy(monkeypatch, policy):
    """Check copy and EMA values, adapter restoration, and manual-load policy guards."""
    import verl.utils.fsdp_utils as upstream

    import verl_omni.utils.fsdp_utils as local

    loads = []
    monkeypatch.setattr(upstream, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(upstream, "load_fsdp_model_to_gpu", lambda module: loads.append(module))
    monkeypatch.setattr(local, "fsdp_summon_full_params", lambda *args, **kwargs: nullcontext())
    engine = Engine(policy)
    engine.copy_adapter("default", "old")
    torch.testing.assert_close(engine.module.old, torch.full((2, 4), 3.0))
    with torch.no_grad():
        engine.module.default.fill_(5)
    engine.ema_update_adapter("default", "old", decay=0.5)
    torch.testing.assert_close(engine.module.old, torch.full((2, 4), 4.0))
    assert engine.module.active == "default"
    assert all(p.device.type == "cpu" for p in engine.module.parameters())
    assert all(t.device.type == "cpu" for t in engine.module.state_dict().values())
    assert len(loads) == (0 if policy else 2)
