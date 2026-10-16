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
"""Exercise model-scoped transfer ordering with deterministic deferred host copies."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

GUARD = runpy.run_path(Path(__file__).resolve().parents[2] / "verl_omni/utils/fsdp_offload.py")
install_guard = GUARD["install_fsdp2_cpu_transfer_guard"]


class Tensor:
    def __init__(self, values, device="npu", index=15, pinned=False):
        self.values = values
        self.device = SimpleNamespace(type=device, index=index)
        self.shape = (len(values),)
        self.pinned = pinned

    def numel(self):
        return len(self.values)

    def is_pinned(self):
        return self.pinned


class Shard:
    def __init__(self, dim):
        self.dim = dim


class Replicate:
    pass


class DTensor(Tensor):
    def __init__(self, values, device="npu", index=15, *, size=None, placements=None, mesh=(2,), pinned=False):
        super().__init__(values, device, index, pinned)
        self.local = Tensor(values, device, index, pinned)
        self.shape = (2 * len(values) + 1 if size is None else size,)
        self.placements = (Shard(0),) if placements is None else placements
        self.device_mesh = SimpleNamespace(size=lambda dim: mesh[dim])

    def to_local(self):
        return self.local


class Device:
    def __init__(self):
        self.pending = []
        self.waited_devices = []
        self.queried_devices = []

    def current_stream(self, source):
        self.queried_devices.append(source)
        return SimpleNamespace(synchronize=lambda: self.synchronize(source))

    def synchronize(self, device=None):
        self.waited_devices.append(device)
        for copy in self.pending:
            copy()
        self.pending.clear()

    def to_cpu(self, tensor):
        result = type(tensor)([-999.0] * len(tensor.values), "cpu")

        def copy():
            result.values[:] = tensor.values

        self.pending.append(copy)
        return result


class FSDPModule:
    """Model the recursive _apply -> per-child CPU repadding order."""

    def __init__(self, children=(), parameter=None, gradient=None, buffer=None, cpu_copy=False):
        self.children = children
        self.parameter = parameter
        self.gradient = gradient
        self.buffer = buffer
        self.cpu_copy = cpu_copy

    def _apply(self, fn, recurse=True):
        if recurse:
            for child in self.children:
                child._apply(fn)
        for name in ("parameter", "gradient", "buffer"):
            if (tensor := getattr(self, name)) is not None:
                setattr(self, name, fn(tensor))
        if self.parameter is not None and self.parameter.device.type == "cpu" and self.cpu_copy:
            # Like an uneven FSDP shard, this allocation is detached from the
            # DMA destination; a later device wait cannot repair its contents.
            self.parameter = DTensor(self.parameter.values.copy(), "cpu")
        return self


@pytest.fixture
def device(monkeypatch):
    accelerator = Device()
    fsdp = ModuleType("torch.distributed.fsdp")
    fsdp.FSDPModule = FSDPModule
    tensors = ModuleType("torch.distributed.tensor")
    tensors.DTensor, tensors.Shard, tensors.Replicate = DTensor, Shard, Replicate
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp)
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", tensors)
    monkeypatch.setattr(torch, "get_device_module", lambda device: accelerator)
    return accelerator


@pytest.mark.parametrize("backend", ["npu", "cuda"])
def test_guard_waits_before_nested_child_repads_not_after_root_returns(device, backend):
    expected = [float(i) for i in range(119)]
    bad_leaf = FSDPModule(parameter=DTensor(expected.copy(), backend), cpu_copy=True)
    bad_root = FSDPModule(children=[FSDPModule(children=[bad_leaf])])
    bad_root._apply(device.to_cpu)
    device.synchronize()
    assert bad_leaf.parameter.values == [-999.0] * 119

    good_leaf = FSDPModule(parameter=DTensor(expected.copy(), backend), cpu_copy=True)
    good_root = FSDPModule(children=[FSDPModule(children=[good_leaf])])
    assert install_guard(good_root, backend)
    assert good_root._apply(device.to_cpu) is good_root
    assert good_leaf.parameter.values == expected
    assert device.pending == []
    assert device.waited_devices[-1].index == 15


def test_guard_leaves_plain_buffers_async_and_uses_each_actual_source_device(device):
    root = FSDPModule(
        parameter=DTensor([1.0], index=2), gradient=DTensor([2.0], index=5), buffer=Tensor([3.0], index=6)
    )
    install_guard(root, "npu")
    root._apply(device.to_cpu)
    assert root.parameter.values == [1.0]
    assert root.gradient.values == [2.0]
    assert root.buffer.values == [-999.0]
    assert [value.index for value in device.waited_devices] == [2, 5]
    device.synchronize()
    assert root.buffer.values == [3.0]


def test_guard_is_idempotent_and_does_not_patch_other_models_or_the_class(device):
    guarded = FSDPModule(parameter=DTensor([7.0]), cpu_copy=True)
    other = FSDPModule(parameter=DTensor([7.0]), cpu_copy=True)
    original = FSDPModule._apply
    assert install_guard(guarded, "npu")
    bound = guarded._apply
    assert not install_guard(guarded, "npu")
    assert guarded._apply is bound
    guarded._apply(device.to_cpu)
    other._apply(device.to_cpu)
    device.synchronize()
    assert guarded.parameter.values == [7.0]
    assert other.parameter.values == [-999.0]
    assert FSDPModule._apply is original


@pytest.mark.parametrize("source,target", [("cpu", "cpu"), ("cpu", "npu"), ("meta", "npu"), ("npu", "npu")])
def test_guard_does_not_synchronize_other_conversion_directions(device, source, target):
    root = FSDPModule(parameter=DTensor([7.0], source))
    install_guard(root, "npu")
    root._apply(lambda tensor: DTensor(tensor.values.copy(), target))
    assert root.parameter.values == [7.0]
    assert device.waited_devices == []


@pytest.mark.parametrize("backend", ["cpu", "meta"])
def test_host_install_is_a_noop_without_importing_accelerator_dependencies(backend):
    assert not install_guard(object(), backend)


def test_recurse_false_is_preserved(device):
    child = FSDPModule(parameter=DTensor([9.0]))
    root = FSDPModule(children=[child], parameter=DTensor([7.0]))
    install_guard(root, "npu")
    root._apply(fn=device.to_cpu, recurse=False)
    assert root.parameter.values == [7.0]
    assert child.parameter.device.type == "npu"
    assert len(device.waited_devices) == 1


def test_guard_propagates_conversion_and_device_errors(device):
    root = FSDPModule(parameter=DTensor([7.0]))
    install_guard(root, "npu")

    def fail(*args):
        raise RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        root._apply(fail)
    assert device.waited_devices == []
    device.synchronize = fail
    with pytest.raises(RuntimeError, match="transfer failed"):
        root._apply(device.to_cpu)


def test_guard_rejects_non_fsdp_modules(device):
    with pytest.raises(TypeError, match="FSDP2 root"):
        install_guard(SimpleNamespace(), "npu")


@pytest.mark.parametrize("values,size", [([1.0] * 4, 8), ([1.0] * 4, 7), ([], 1)])
def test_full_sized_and_empty_shards_do_not_query_or_wait_on_streams(device, values, size):
    root = FSDPModule(parameter=DTensor(values, size=size))
    install_guard(root, "npu")
    root._apply(device.to_cpu)
    assert device.queried_devices == device.waited_devices == []
    device.synchronize()
    assert root.parameter.values == values


@pytest.mark.parametrize("pinned", [False, True])
def test_even_shards_wait_only_if_policy_will_copy_to_pinned_memory(device, pinned):
    root = FSDPModule(parameter=DTensor([7.0], size=2), cpu_copy=not pinned)
    install_guard(root, "npu", pin_memory=True)

    def convert(tensor):
        result = device.to_cpu(tensor)
        result.local.pinned = pinned
        return result

    root._apply(convert)
    assert len(device.waited_devices) == (0 if pinned else 1)
    device.synchronize()
    assert root.parameter.values == [7.0]


def test_hsdp_uses_shard_mesh_dimension_not_total_world_size(device):
    param = DTensor([7.0] * 4, size=8, placements=(Replicate(), Shard(0)), mesh=(2, 2))
    root = FSDPModule(parameter=param)
    install_guard(root, "npu")
    root._apply(device.to_cpu)
    assert device.queried_devices == []


def test_combined_sharding_is_conservatively_guarded(device):
    param = DTensor([7.0] * 2, size=8, placements=(Shard(0), Shard(0)), mesh=(2, 2))
    root = FSDPModule(parameter=param)
    install_guard(root, "npu")
    root._apply(device.to_cpu)
    assert len(device.waited_devices) == 1


def test_capture_source_stream_before_converter_changes_current_stream(device, monkeypatch):
    other_stream = Device()
    root = FSDPModule(parameter=DTensor([7.0]), cpu_copy=True)
    install_guard(root, "npu")

    def convert(tensor):
        result = device.to_cpu(tensor)
        monkeypatch.setattr(torch, "get_device_module", lambda source: other_stream)
        return result

    root._apply(convert)
    assert root.parameter.values == [7.0]
    assert len(device.waited_devices) == 1
    assert other_stream.waited_devices == []


def test_guard_does_not_drain_an_unrelated_stream(device):
    unrelated = Device()
    outstanding = unrelated.to_cpu(Tensor([5.0]))
    root = FSDPModule(parameter=DTensor([7.0]), cpu_copy=True)
    install_guard(root, "npu")
    root._apply(device.to_cpu)
    assert root.parameter.values == [7.0]
    assert outstanding.values == [-999.0]
    assert unrelated.waited_devices == []
