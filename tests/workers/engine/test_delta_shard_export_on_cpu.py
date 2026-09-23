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
"""CPU checks for the diffusers engine's delta shard export (``delta_sharded`` backend).

Drives the real ``DiffusersFSDPEngine`` export methods with verl's own delta machinery
(``prime_delta_snapshots`` / ``hf_delta_export``, the same helpers verl's
``tests/checkpoint_engine/test_sharded_delta.py`` uses as the protocol reference):
names must match the full export (``convert_weight_keys`` + ``transformer.`` prefix),
shards must reassemble the full parameter, and a prime -> perturb -> delta -> apply
round trip must reproduce the perturbed state bit-exactly.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _ToyDiT(torch.nn.Module):
    """Two linear layers behind a checkpoint conversion mapping, like a diffusers DiT."""

    # transformers convention: checkpoint (HF) name pattern -> model-local name pattern
    _checkpoint_conversion_mapping = {"^transformer_blocks": "blocks"}

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])


class _FakeModule:
    """Bare ``state_dict()`` holder; carries no ``peft_config`` or conversion mapping."""

    def __init__(self, state: dict):
        self._state = state

    def state_dict(self):
        return self._state


def _make_engine(module) -> PPODiffusersFSDPEngine:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = False
    # DiffusionModelConfig.__post_init__ does I/O; set the field under test directly.
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora", {})
    engine.model_config = model_config
    return engine


def _patch_sync_helpers(monkeypatch):
    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", MagicMock())
    monkeypatch.setattr(diffusers_impl, "load_fsdp_model_to_gpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "offload_fsdp_model_to_cpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: torch.device("cpu"))


def _full_export(engine) -> dict:
    full, _ = engine.get_per_tensor_param()
    # Raw full export: DTensors ship bf16, plain tensors keep their native dtype.
    # The shard export must mirror exactly this cast rule, or the pinned diff base
    # diverges from what the seed sync sent the rollout.
    return dict(full)


def test_shard_export_matches_full_export_names_and_values(monkeypatch):
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_sync_helpers(monkeypatch)
    engine = _make_engine(module)

    full = _full_export(engine)
    shards = list(engine.get_per_tensor_param_shard()[0])

    assert [name for name, _, _ in shards] == list(full.keys())
    for name, local, spec in shards:
        # conversion mapping applied, then the rollout-facing prefix
        assert name.startswith("transformer.transformer_blocks.")
        # plain fp32 params keep their native dtype in both exports (cast parity)
        assert local.dtype == torch.float32
        assert spec.full_shape == tuple(full[name].shape)
        # unsharded (no process group): the local shard is the whole flat tensor
        assert torch.equal(local, full[name].reshape(-1))


def test_shard_coverage_reassembles_full_tensor(monkeypatch):
    from verl.workers.engine.spec import derive_dtensor_placement, translate_flat_indices

    torch.manual_seed(0)
    _patch_sync_helpers(monkeypatch)
    engine = _make_engine(_ToyDiT())

    full = _full_export(engine)
    for name, local, spec in engine.get_per_tensor_param_shard()[0]:
        place, contributes, group = derive_dtensor_placement(spec)
        assert contributes and group is None
        pos = translate_flat_indices(torch.arange(local.numel()), place)
        assert torch.unique(pos).numel() == local.numel() == full[name].numel()
        reassembled = torch.empty_like(full[name]).reshape(-1)
        reassembled[pos] = local
        assert torch.equal(reassembled, full[name].reshape(-1))


def test_delta_round_trip_bit_exact(monkeypatch):
    """Seed -> prime -> perturb -> two delta exports -> apply reproduces the reference.

    The apply side mirrors the rollout receiver: per-slot ``index_copy_`` of the
    shipped (position, value) pairs into the already-loaded weights.
    """
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_sync_helpers(monkeypatch)
    engine = _make_engine(module)

    # Seed: the rollout loads the full export; the engine pins its shard snapshots.
    rollout_state = {name: t.clone() for name, t in _full_export(engine).items()}
    engine.prime_delta_snapshots()

    # Perturb two elements, as an optimizer step would.
    with torch.no_grad():
        module.blocks[0].weight.view(-1)[3] += 0.5
        module.blocks[1].weight.view(-1)[7] -= 1.0

    shipped = 0
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    for slots, dtype_str, counts, hf_idx, hf_val, gather_group in deltas:
        # fp32 plain params ship native; the wire dtype follows the export
        assert dtype_str == "float32" and gather_group is None
        off = 0
        for (name, shape), count in zip(slots, counts.tolist(), strict=True):
            if count:
                idx = hf_idx[off : off + count].to(torch.int64)
                rollout_state[name].reshape(-1).index_copy_(0, idx, hf_val[off : off + count])
                shipped += count
            off += count
    assert shipped == 2

    reference = _full_export(engine)
    for name, ref in reference.items():
        assert torch.equal(rollout_state[name], ref), name

    # The snapshot refreshes on every export: a second delta ships nothing.
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    for slots, _, counts, hf_idx, hf_val, _ in deltas:
        assert counts.tolist() == [0] * len(slots)
        assert hf_idx.numel() == 0 and hf_val.numel() == 0


def test_shard_export_dtensor_local_view(monkeypatch):
    """A DTensor param exports its local shard with the DTensor-derived spec."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from verl.workers.engine.spec import derive_dtensor_placement

    owns_pg = not dist.is_initialized()
    if owns_pg:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29513")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        full = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        dtensor = DTensor.from_local(full.clone(), mesh, [Shard(0)])

        _patch_sync_helpers(monkeypatch)
        engine = _make_engine(_FakeModule({"w": dtensor}))
        ((name, local, spec),) = engine.get_per_tensor_param_shard()[0]

        assert name == "transformer.w"
        assert spec.mesh is mesh and spec.placements == (Shard(0),)
        assert spec.full_shape == (2, 4)
        assert torch.equal(local, full.reshape(-1).to(torch.bfloat16))  # world 1: local == full

        place, contributes, group = derive_dtensor_placement(spec)
        assert contributes and group is not None  # the Shard(0) dim's subgroup
        assert place.is_flat_contiguous and place.flat_offset == 0
    finally:
        # leaving the default pg alive leaks into whatever test runs next in the
        # same session (repo convention: init in the test -> destroy in finally)
        if owns_pg:
            dist.destroy_process_group()


def test_shard_export_rejects_lora():
    module = _ToyDiT()
    module.peft_config = {"default": SimpleNamespace()}
    engine = _make_engine(module)
    with pytest.raises(NotImplementedError, match="full-weight"):
        engine.get_per_tensor_param_shard()


def test_delta_export_requires_seed(monkeypatch):
    """A delta export without a prior prime must fail loud, not diff against garbage."""
    _patch_sync_helpers(monkeypatch)
    engine = _make_engine(_ToyDiT())
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    with pytest.raises(AssertionError, match="seed snapshot"):
        list(deltas)
