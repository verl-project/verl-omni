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

"""Two-rank numerical coverage for clipping explicitly sharded model leaves."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from verl_omni.workers.engine.fsdp.training_utils import clip_grad_norm_sharded_


def _clip_worker(rank, store):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        shard = torch.nn.Parameter(DTensor.from_local(torch.zeros(1), mesh, [Shard(0)]))
        shard.grad = DTensor.from_local(torch.tensor([3.0 + rank]), mesh, [Shard(0)])
        replica = torch.nn.Parameter(DTensor.from_local(torch.zeros(1), mesh, [Replicate()]))
        replica.grad = DTensor.from_local(torch.tensor([12.0]), mesh, [Replicate()])
        norm = clip_grad_norm_sharded_([shard, replica], 6.5, mesh)
        assert norm.item() == pytest.approx(13.0)
        assert shard.grad.to_local().item() == pytest.approx((3.0 + rank) / 2, abs=1e-6)
        assert replica.grad.to_local().item() == pytest.approx(6.0, abs=1e-6)
        # A plain replicated tensor is counted once, too.
        plain = torch.nn.Parameter(torch.zeros(1))
        plain.grad = torch.tensor([12.0])
        assert clip_grad_norm_sharded_([plain], 100, mesh).item() == pytest.approx(12.0)
        # An unused local branch still participates in the scalar collective.
        shard.grad = None if rank == 0 else DTensor.from_local(torch.tensor([4.0]), mesh, [Shard(0)])
        assert clip_grad_norm_sharded_([shard], 2.0, mesh).item() == pytest.approx(4.0)
        # Nonfinite norms do not corrupt other gradients before the engine skips the step.
        shard.grad = DTensor.from_local(torch.tensor([float("inf") if rank == 0 else 4.0]), mesh, [Shard(0)])
        assert torch.isinf(clip_grad_norm_sharded_([shard], 1.0, mesh))
        if rank == 1:
            assert shard.grad.to_local().item() == 4.0
    finally:
        dist.destroy_process_group()


def test_sharded_norm_matches_dense_reference(tmp_path):
    mp.spawn(_clip_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)
