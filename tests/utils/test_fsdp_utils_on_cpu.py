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
"""CPU tests for FSDP2 root ``ignored_params`` injection."""

import torch.nn as nn

from verl_omni.utils.fsdp_utils import apply_fsdp2_excluding_module_names


def test_apply_fsdp2_excluding_injects_ignored_params_on_root_only(monkeypatch):
    import verl.utils.fsdp_utils as verl_fsdp

    root = nn.Sequential()
    root.apm = nn.Linear(4, 4)
    root.llm = nn.Linear(4, 4)
    child = root.llm
    calls = []

    def fake_fully_shard(module, *args, **kwargs):
        del args
        calls.append((module, kwargs.get("ignored_params")))
        return module

    def fake_apply_fsdp2(model, fsdp_kwargs, config):
        del config
        verl_fsdp.fully_shard(child, **fsdp_kwargs)
        verl_fsdp.fully_shard(model, **fsdp_kwargs)

    monkeypatch.setattr(verl_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(verl_fsdp, "apply_fsdp2", fake_apply_fsdp2)

    apply_fsdp2_excluding_module_names(root, {}, {}, ["apm"])

    assert calls[0] == (child, None)
    assert calls[1][0] is root
    assert calls[1][1] == set(root.apm.parameters())
    assert all(param.requires_grad for param in root.apm.parameters())
    assert verl_fsdp.fully_shard is fake_fully_shard
