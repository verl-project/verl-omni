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
"""CPU checks for engine-side LoRA merge (``model.lora.merge='engine'``).

Covers the rollout-side consumption of the engine's merged weight-update
session: payload routing in the colocated worker extension, fail-closed
guards, and the engine-args selection on both rollout strategies. The
CUDA-IPC handle production itself needs a GPU and is exercised by the GPU
smoke entries instead.
"""

from argparse import Namespace
from types import MethodType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl_omni.workers.rollout.vllm_rollout.utils as rollout_utils
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import OmniStrategyBase

_PAYLOAD = {
    "names": ["base_model.model.foo.lora_A.weight"],
    "dtype_names": ["float32"],
    "shapes": [[8, 16]],
    "ipc_handles": [{"gpu-uuid": (0, 1, 2)}],
    "peft_config": {"r": 8, "lora_alpha": 16, "target_modules": ["foo"]},
    "lora_int_id": 1,
}


class _FakeWorker:
    """Records the weight-transfer session calls the extension drives."""

    def __init__(self, backend="omni_lora_ipc", fail_on_update=False):
        """Configure the fake transfer backend and whether receive fails."""
        self.vllm_config = SimpleNamespace(weight_transfer_config=SimpleNamespace(backend=backend))
        self.weight_transfer_engine = object()
        self._pending_lora_peft_config = None
        self.calls = []
        self.fail_on_update = fail_on_update

    def start_weight_update(self):
        """Record the session-open call."""
        self.calls.append(("start",))

    def update_weights(self, payload):
        """Record the receive call, optionally failing it."""
        self.calls.append(("update", payload))
        if self.fail_on_update:
            raise RuntimeError("receive failed")

    def finish_weight_update(self):
        """Record the session-close call."""
        self.calls.append(("finish",))

    def bind_extension(self):
        """Attach the real extension method so unbound-call patterns work."""
        # update_weights_from_ipc drives the session through the extension's
        # own method; bind it so the unbound-method call pattern works.
        self._apply_merged_lora_payload = MethodType(vLLMOmniColocateWorkerExtension._apply_merged_lora_payload, self)
        return self


def _no_hijack(monkeypatch):
    """Fail the test if the runtime-adapter ingest patch installs."""

    def _fail():
        """Stand-in that turns any hijack call into a test failure."""
        raise AssertionError("the runtime-adapter ingest patch must not install in engine-merge mode")

    monkeypatch.setattr(rollout_utils.VLLMOmniHijack, "hijack", _fail)


def test_payload_drives_the_merged_session_in_order(monkeypatch):
    """A merged payload runs start/update/finish with the payload verbatim."""
    _no_hijack(monkeypatch)
    worker = _FakeWorker().bind_extension()

    vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, lora_ipc_payload=dict(_PAYLOAD))

    assert [name for name, *_ in worker.calls] == ["start", "update", "finish"]
    assert worker.calls[1][1] == _PAYLOAD


def test_payload_fails_closed_without_a_transfer_engine(monkeypatch):
    """A missing transfer engine refuses before any session call runs."""
    _no_hijack(monkeypatch)
    worker = _FakeWorker().bind_extension()
    worker.weight_transfer_engine = None

    with pytest.raises(RuntimeError, match="no weight-transfer engine"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, lora_ipc_payload=dict(_PAYLOAD))
    assert worker.calls == []


def test_payload_fails_closed_on_non_merged_backends():
    """Wrong or absent weight-transfer backend refuses before any session call."""
    worker = _FakeWorker(backend=None).bind_extension()
    with pytest.raises(ValueError, match="weight-transfer backend"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, lora_ipc_payload=dict(_PAYLOAD))
    assert worker.calls == []

    worker = _FakeWorker(backend="ipc").bind_extension()
    with pytest.raises(ValueError, match="lora.merge='engine'"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, lora_ipc_payload=dict(_PAYLOAD))
    assert worker.calls == []


def test_payload_rejects_the_adapter_tensor_path(monkeypatch):
    """Adapter-tensor spellings of the sync are refused in engine-merge mode."""
    _no_hijack(monkeypatch)
    worker = _FakeWorker()
    with pytest.raises(ValueError, match="one payload per sync"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
            worker, peft_config={"r": 8}, lora_ipc_payload=dict(_PAYLOAD)
        )
    assert worker.calls == []

    worker._pending_lora_peft_config = {"r": 8}
    with pytest.raises(ValueError, match="one payload per sync"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, lora_ipc_payload=dict(_PAYLOAD))
    assert worker.calls == []


def test_failed_session_unwedges_worker_state_and_reraises(monkeypatch):
    """A failed receive retries finish (unwedge) and re-raises the original error."""
    _no_hijack(monkeypatch)
    worker = _FakeWorker(fail_on_update=True).bind_extension()

    with pytest.raises(RuntimeError, match="receive failed"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, lora_ipc_payload=dict(_PAYLOAD))

    # finish runs twice: once in the try (skipped by the failure) and once as
    # the cleanup that unwedges the worker's active-session state.
    assert [name for name, *_ in worker.calls] == ["start", "update", "finish"]


def _strategy_server(merge):
    """A minimal async server whose model_config carries the given merge."""
    return SimpleNamespace(
        config=SimpleNamespace(engine_kwargs={}),
        model_config=OmegaConf.create({"lora_rank": 32, "lora": {"merge": merge}}),
    )


def test_engine_merge_selects_itself_on_engine_args(monkeypatch):
    """Engine merge sets the IPC backend itself; diffusion adds lora_application."""
    monkeypatch.setattr(
        "verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base.resolve_lora_config",
        lambda model_config: SimpleNamespace(engine_merge=True),
        raising=False,
    )
    for diffusion, expect_lora_application in ((True, "merged"), (False, None)):
        engine_args = {"enable_lora": False}
        OmniStrategyBase._apply_engine_merge_engine_args(
            SimpleNamespace(server=_strategy_server("engine")), engine_args, diffusion=diffusion
        )
        assert engine_args["weight_transfer_config"] == {"backend": "omni_lora_ipc"}
        assert engine_args.get("lora_application") is expect_lora_application


def test_runtime_mode_leaves_engine_args_untouched():
    """Runtime-adapter and actor-merge modes set no weight-transfer args."""
    for merge in (True, False):
        engine_args = {"enable_lora": False}
        OmniStrategyBase._apply_engine_merge_engine_args(
            SimpleNamespace(server=_strategy_server(merge)), engine_args, diffusion=True
        )
        assert "weight_transfer_config" not in engine_args
        assert "lora_application" not in engine_args


def test_ar_strategy_injects_engine_merge_engine_args(monkeypatch):
    """The AR strategy derives the IPC backend and skips the ingest patch."""
    calls = []
    monkeypatch.setattr(
        "verl_omni.utils.vllm_omni.patch.patch_vllm_omni_lora_config",
        lambda: calls.append("patch"),
    )
    engine_args = {}

    ARStrategy(_strategy_server("engine")).prepare_engine_args(engine_args, Namespace())

    assert engine_args["weight_transfer_config"] == {"backend": "omni_lora_ipc"}
    assert "lora_application" not in engine_args
    assert calls == []


def test_engine_merge_refuses_the_spelling_without_lora_configured():
    """The engine spelling with no LoRA configured fails closed."""
    from verl_omni.utils.config import resolve_lora_config

    model = OmegaConf.create({"lora_rank": 0, "lora": {"merge": "engine"}})
    with pytest.raises(ValueError, match="no LoRA is configured"):
        resolve_lora_config(model)


def test_merged_payload_builder_shapes_the_upstream_contract(monkeypatch):
    """CPU-visible half of the producer contract: sorted names, per-tensor
    UUID-keyed handle dicts from the patched reducer, staging refs returned."""
    import torch

    import verl_omni.workers.engine_workers as ew

    recorded = []

    def fake_reduce(tensor):
        """Record the handle tuple the patched reducer would produce."""
        args = ("handle", tensor.shape, tensor.stride(), tensor.dtype)
        recorded.append(args)
        return None, args

    monkeypatch.setattr("torch.multiprocessing.reductions.reduce_tensor", fake_reduce)
    monkeypatch.setattr(
        ew,
        "get_torch_device",
        lambda: SimpleNamespace(
            current_device=lambda: 0, get_device_properties=lambda index: SimpleNamespace(uuid="gpu-uuid")
        ),
        raising=True,
    )
    # CPU host: the device move is the one CUDA-only step; stub the move and
    # keep sorting, handle assembly, and payload construction real.
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None: self, raising=True)

    weights = {
        "base_model.model.b.lora_B.weight": torch.zeros(4, 8),
        "base_model.model.a.lora_A.weight": torch.zeros(8, 2),
    }
    payload, staged = ew._build_merged_lora_ipc_payload(weights, peft_config={"r": 8})

    assert payload["names"] == sorted(weights)
    assert payload["dtype_names"] == ["float32", "float32"]
    assert payload["shapes"] == [[8, 2], [4, 8]]
    assert payload["ipc_handles"] == [{"gpu-uuid": recorded[0]}, {"gpu-uuid": recorded[1]}]
    assert payload["peft_config"] == {"r": 8}
    assert payload["lora_int_id"] == 1
    # The staging refs share storage with the gathered tensors (detach views).
    assert staged[0].data_ptr() == weights["base_model.model.a.lora_A.weight"].data_ptr()
    assert staged[1].data_ptr() == weights["base_model.model.b.lora_B.weight"].data_ptr()
