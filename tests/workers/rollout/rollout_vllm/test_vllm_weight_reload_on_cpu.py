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

import importlib
from types import SimpleNamespace
from unittest.mock import sentinel

import pytest
import torch
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

from verl_omni.workers.rollout.vllm_rollout import utils as utils_module


def _make_worker(model, model_config):
    return SimpleNamespace(
        device="cpu",
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc:///tmp/test.sock",
        _get_standard_weight_model_and_config=lambda: (model, model_config),
    )


def _patch_reload_hooks(monkeypatch, events):
    reload_module = importlib.import_module("vllm.model_executor.model_loader.reload")
    monkeypatch.setattr(
        reload_module,
        "initialize_layerwise_reload",
        lambda model: events.append("initialize"),
    )
    monkeypatch.setattr(
        reload_module,
        "finalize_layerwise_reload",
        lambda model, config: events.append("finalize"),
    )
    monkeypatch.setattr(
        utils_module.torch.accelerator,
        "synchronize",
        lambda: events.append("synchronize"),
    )


def _patch_standard_reload_dependencies(monkeypatch, events):
    patch_module = importlib.import_module("verl.utils.vllm.patch")
    loader_utils = importlib.import_module("vllm.model_executor.model_loader.utils")
    monkeypatch.setattr(
        patch_module,
        "patch_vllm_moe_model_weight_loader",
        lambda model: events.append("patch_moe_loader"),
    )
    monkeypatch.setattr(
        loader_utils,
        "process_weights_after_loading",
        lambda model, config, device: events.append("process_after_loading"),
    )
    monkeypatch.setattr(
        utils_module.torch.accelerator,
        "synchronize",
        lambda: events.append("synchronize"),
    )


def test_standard_vllm_bucket_reload_order_on_cpu(monkeypatch):
    events = []
    buckets = [sentinel.bucket_0, sentinel.bucket_1]
    model_config = sentinel.model_config
    model = SimpleNamespace(load_weights=lambda weights: events.append(("load", weights)))

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            for bucket in buckets:
                on_bucket_received(bucket)

    monkeypatch.setattr(
        bucketed_weight_transfer,
        "BucketedWeightReceiver",
        FakeReceiver,
    )
    _patch_reload_hooks(monkeypatch, events)
    _patch_standard_reload_dependencies(monkeypatch, events)
    monkeypatch.setattr(utils_module, "_is_npu_platform", lambda: True)
    monkeypatch.setattr(utils_module, "_has_layerwise_reload_metadata", lambda model: True)

    utils_module.vLLMOmniColocateWorkerExtension.update_weights_from_ipc(_make_worker(model, model_config))

    assert events == [
        "patch_moe_loader",
        "initialize",
        ("load", sentinel.bucket_0),
        ("load", sentinel.bucket_1),
        "finalize",
        "synchronize",
    ]


@pytest.mark.parametrize(
    ("failure_site", "expected_events"),
    [
        ("receive", ["patch_moe_loader", "initialize", "receive", "finalize"]),
        ("load", ["patch_moe_loader", "initialize", "receive", "load", "finalize"]),
    ],
)
def test_standard_vllm_reload_finalizes_before_reraising(
    monkeypatch,
    failure_site,
    expected_events,
):
    events = []
    original_error = RuntimeError(f"{failure_site} failed")

    def load_weights(weights):
        events.append("load")
        if failure_site == "load":
            raise original_error

    model = SimpleNamespace(load_weights=load_weights)

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            events.append("receive")
            if failure_site == "receive":
                raise original_error
            on_bucket_received(sentinel.bucket)

    monkeypatch.setattr(
        bucketed_weight_transfer,
        "BucketedWeightReceiver",
        FakeReceiver,
    )
    _patch_reload_hooks(monkeypatch, events)
    _patch_standard_reload_dependencies(monkeypatch, events)
    monkeypatch.setattr(utils_module, "_is_npu_platform", lambda: True)
    monkeypatch.setattr(utils_module, "_has_layerwise_reload_metadata", lambda model: True)

    with pytest.raises(RuntimeError) as exc_info:
        utils_module.vLLMOmniColocateWorkerExtension.update_weights_from_ipc(_make_worker(model, sentinel.model_config))

    assert exc_info.value is original_error
    assert events == expected_events
    assert "synchronize" not in events


@pytest.mark.parametrize("is_npu", [False, True])
def test_standard_vllm_reload_falls_back_to_full_post_load_processing(monkeypatch, is_npu):
    events = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def load_weights(self, weights):
            events.append(("load", weights))

    model = TinyModel()
    assert not utils_module._has_layerwise_reload_metadata(model)

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received(sentinel.bucket)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    _patch_standard_reload_dependencies(monkeypatch, events)
    monkeypatch.setattr(utils_module, "_is_npu_platform", lambda: is_npu)

    utils_module.vLLMOmniColocateWorkerExtension.update_weights_from_ipc(_make_worker(model, sentinel.model_config))

    assert events == [
        "patch_moe_loader",
        ("load", sentinel.bucket),
        "process_after_loading",
        "synchronize",
    ]


def test_layerwise_reload_metadata_detection_uses_vllm_registry():
    reload_module = importlib.import_module("vllm.model_executor.model_loader.reload")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))

    assert not utils_module._has_layerwise_reload_metadata(model)

    reload_module.record_metadata_for_reloading(model)

    assert utils_module._has_layerwise_reload_metadata(model)
