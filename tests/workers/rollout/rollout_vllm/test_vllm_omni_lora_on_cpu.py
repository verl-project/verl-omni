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

import torch

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension


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


class _FakeMoeEngine:
    """Stands in for a SUPPORTED_MOE_MODELS entry; the gate only isinstance-checks."""


class _ACLGraphWrapper:
    """Mirrors the Ascend wrapper: holds the real model under ``runnable``."""

    def __init__(self, runnable):
        self.runnable = runnable


def _patch_calls(monkeypatch):
    """Point the rollout utils at a recording patch fn and a 1-class whitelist."""
    import verl_omni.workers.rollout.vllm_rollout.utils as rollout_utils

    calls = []

    def recording_patch(model):
        calls.append(model)

    monkeypatch.setattr(rollout_utils, "patch_vllm_moe_model_weight_loader", recording_patch)
    monkeypatch.setattr(rollout_utils, "SUPPORTED_MOE_MODELS", [_FakeMoeEngine])
    return rollout_utils, calls


def test_is_moe_engine_matches_only_whitelisted_outer_classes(monkeypatch):
    rollout_utils, _ = _patch_calls(monkeypatch)
    assert rollout_utils._is_moe_engine(_FakeMoeEngine())
    # Dense engines without a resolvable inner model (MiniCPM-o nests its LLM as
    # .llm) are skipped; a nested match does not count — the inner-model probe is
    # exactly the verl behavior this gate avoids.
    assert not rollout_utils._is_moe_engine(object())
    assert not rollout_utils._is_moe_engine(SimpleNamespace(model=_FakeMoeEngine()))
    # Ascend wraps the model, so unwrap before the class check.
    assert rollout_utils._is_moe_engine(_ACLGraphWrapper(_FakeMoeEngine()))
    assert not rollout_utils._is_moe_engine(_ACLGraphWrapper(object()))


def test_supported_moe_models_registers_qwen3_omni_but_not_minicpm():
    # The production whitelist is keyed on the real engine classes: the
    # Qwen3-Omni thinker must be covered, the dense MiniCPM-o thinker must not.
    import pytest

    import verl_omni.workers.rollout.vllm_rollout.utils as rollout_utils

    qwen3_omni = pytest.importorskip(
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni",
        reason="vllm-omni build without the qwen3_omni module",
    )
    minicpm = pytest.importorskip(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm",
        reason="vllm-omni build without the minicpmo_4_5 module",
    )
    assert qwen3_omni.Qwen3OmniMoeForConditionalGeneration in rollout_utils.SUPPORTED_MOE_MODELS
    assert minicpm.MiniCPMO45OmniLLMForConditionalGeneration not in rollout_utils.SUPPORTED_MOE_MODELS


def test_monkey_patch_model_gates_on_the_whitelist(monkeypatch):
    import pytest

    rollout_utils, calls = _patch_calls(monkeypatch)
    moe_engine = _FakeMoeEngine()
    ar_worker = SimpleNamespace(_get_standard_weight_model_and_config=lambda: (moe_engine, object()))
    vLLMOmniColocateWorkerExtension.monkey_patch_model(ar_worker)
    assert calls == [moe_engine]

    # Dense engines and diffusion-style workers (no standard model): no-op.
    dense_worker = SimpleNamespace(_get_standard_weight_model_and_config=lambda: (object(), object()))
    diffusion_worker = SimpleNamespace(_get_standard_weight_model_and_config=lambda: None)
    vLLMOmniColocateWorkerExtension.monkey_patch_model(dense_worker)
    vLLMOmniColocateWorkerExtension.monkey_patch_model(diffusion_worker)
    assert calls == [moe_engine]

    # A failure on a whitelisted engine is a real bug: let it surface.
    def raising_patch(model):
        raise ValueError("some patch failure")

    monkeypatch.setattr(rollout_utils, "patch_vllm_moe_model_weight_loader", raising_patch)
    with pytest.raises(ValueError, match="patch failure"):
        vLLMOmniColocateWorkerExtension.monkey_patch_model(ar_worker)
