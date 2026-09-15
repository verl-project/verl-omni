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
"""CPU checks for vLLM-Omni LoRA integration."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension


@pytest.mark.parametrize("enabled,seed", [(None, "42"), ("0", "42"), ("1", None), ("1", "7")])
def test_diffusion_worker_restores_explicit_determinism(monkeypatch, enabled, seed):
    from verl.workers.engine import utils as engine_utils
    from vllm.distributed import parallel_state

    from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils

    for key, value in (("VERL_FULL_DETERMINISM", enabled), ("VERL_SEED", seed)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(engine_utils, "enable_full_determinism", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(parallel_state, "set_custom_all_reduce", lambda enabled: calls.append(("custom_ar", enabled)))
    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", True, raising=False)
    monkeypatch.setattr(rollout_utils, "set_death_signal", lambda: None)
    monkeypatch.setattr(rollout_utils.VLLMOmniHijack, "hijack", lambda: None)
    vLLMOmniColocateWorkerExtension.__new__(vLLMOmniColocateWorkerExtension)
    restored = enabled == "1" and seed is not None
    assert calls == ([("seed", int(seed)), ("custom_ar", False)] if restored else [])
    # Deterministic algorithms fill torch.empty buffers with NaN, which would reach
    # the rollout latents, so the rollout path must turn that back off.
    assert torch.utils.deterministic.fill_uninitialized_memory is not restored


def test_diffusion_lora_stacks_follow_the_worker_device():
    layer = SimpleNamespace(
        lora_a_stacked=(torch.ones(1),),
        lora_b_stacked=(torch.ones(1),),
    )
    worker = SimpleNamespace(
        device=torch.device("meta"),
        lora_manager=SimpleNamespace(_lora_modules={"transformer.block": layer}),
    )

    vLLMOmniColocateWorkerExtension._move_diffusion_lora_stacks_to_device(worker)

    assert layer.lora_a_stacked[0].device.type == "meta"
    assert layer.lora_b_stacked[0].device.type == "meta"
