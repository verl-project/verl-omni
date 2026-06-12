# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from verl.utils.vllm import TensorLoRARequest


def _install_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, child_name, sys.modules[name])
    return sys.modules[name]


def _install_lightweight_imports():
    repo_root = Path(__file__).resolve().parents[3]

    for package_name, package_path in {
        "verl_omni": repo_root / "verl_omni",
        "verl_omni.utils": repo_root / "verl_omni" / "utils",
        "verl_omni.workers": repo_root / "verl_omni" / "workers",
        "verl_omni.workers.rollout": repo_root / "verl_omni" / "workers" / "rollout",
        "verl_omni.workers.rollout.vllm_rollout": repo_root
        / "verl_omni"
        / "workers"
        / "rollout"
        / "vllm_rollout",
    }.items():
        package = _install_module(package_name)
        package.__path__ = [str(package_path)]

    class FakeOmniDiffusionConfig:
        def __post_init__(self):
            pass

    class FakeDiffusionLoRAManager:
        pass

    class FakeDiffusersPipelineLoader:
        pass

    class FakeDiffusersAdapterPipeline:
        pass

    class FakeCustomPipelineWorkerExtension:
        pass

    @contextmanager
    def set_default_torch_dtype(_dtype):
        yield

    _install_module("vllm.utils.import_utils", resolve_obj_by_qualname=lambda _name: None)
    _install_module("vllm.utils.mem_utils", GiB_bytes=1024**3)
    _install_module("vllm.utils.torch_utils", set_default_torch_dtype=set_default_torch_dtype)

    _install_module("vllm_omni")
    _install_module("vllm_omni.diffusion")
    _install_module("vllm_omni.diffusion.data", OmniDiffusionConfig=FakeOmniDiffusionConfig)
    _install_module(
        "vllm_omni.diffusion.lora.manager",
        DiffusionLoRAManager=FakeDiffusionLoRAManager,
        logger=logging.getLogger("test_vllm_omni_lora"),
    )
    _install_module(
        "vllm_omni.diffusion.model_loader.diffusers_loader",
        DiffusersPipelineLoader=FakeDiffusersPipelineLoader,
    )
    _install_module(
        "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter",
        DiffusersAdapterPipeline=FakeDiffusersAdapterPipeline,
    )
    _install_module("vllm_omni.diffusion.registry", initialize_model=lambda *_args, **_kwargs: None)
    _install_module(
        "vllm_omni.diffusion.worker.diffusion_worker",
        CustomPipelineWorkerExtension=FakeCustomPipelineWorkerExtension,
    )


_install_lightweight_imports()

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest
from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils

pytestmark = pytest.mark.cpu


def _make_worker(*, lora_enabled: bool = True):
    worker = object.__new__(rollout_utils.vLLMOmniColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace(lora_config=object() if lora_enabled else None)
    worker._get_zmq_handle = lambda: "ipc:///tmp/test.sock"
    return worker


def test_omni_tensor_lora_request_uses_verl_tensor_request():
    assert issubclass(OmniTensorLoRARequest, TensorLoRARequest)


def test_update_weights_from_ipc_accumulates_lora_buckets(monkeypatch):
    received = []

    class FakeReceiver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("a", torch.tensor([1]))])
            on_bucket_received([("b", torch.tensor([2]))])

    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "BucketedWeightReceiver", FakeReceiver)

    worker = _make_worker()
    worker.remove_lora = lambda _adapter_id: None
    worker._update_weights = lambda weights, peft_config, base_sync_done: received.append(
        (list(weights), peft_config, base_sync_done)
    )

    worker.update_weights_from_ipc(peft_config={"r": 16}, base_sync_done=True)

    assert len(received) == 1
    assert [name for name, _ in received[0][0]] == ["a", "b"]
    assert received[0][1] == {"r": 16}
    assert received[0][2] is True


def test_update_weights_from_ipc_drains_adapter_update_when_lora_disabled(monkeypatch):
    drained = []

    class FakeReceiver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("adapter.weight", torch.tensor([1]))])
            drained.append(True)

    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "BucketedWeightReceiver", FakeReceiver)

    worker = _make_worker(lora_enabled=False)
    worker.remove_lora = lambda _adapter_id: pytest.fail("disabled LoRA should not remove adapters")
    worker._update_weights = lambda *_args, **_kwargs: pytest.fail("disabled LoRA should drain without loading")

    worker.update_weights_from_ipc(peft_config={"r": 16}, base_sync_done=True)

    assert drained == [True]


def test_update_weights_releases_lora_tensor_reference():
    requests = []
    worker = _make_worker()
    worker.add_lora = requests.append

    weights = [("a", torch.tensor([1])), ("b", torch.tensor([2]))]
    worker._update_weights(weights, peft_config={"r": 16}, base_sync_done=True)

    assert len(requests) == 1
    assert requests[0].peft_config == {"r": 16}
    assert requests[0].lora_tensors is None


def test_get_zmq_handle_can_use_fixed_split_placement_handles(monkeypatch):
    monkeypatch.setenv("VERL_VLLM_WEIGHT_SYNC_ZMQ_HANDLES", "ipc:///tmp/r0.sock,ipc:///tmp/r1.sock")
    worker = _make_worker()
    del worker._get_zmq_handle
    worker.rank = 1

    assert worker._get_zmq_handle() == "ipc:///tmp/r1.sock"
