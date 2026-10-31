# Copyright 2026 verl-omni contributors
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

"""Real two-process Gloo with CPU AdamW; no GPU/TP model qualification."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.workers.test_omni_output_version_on_cpu import CPUBackend, TrackedCPU, assert_equal
from verl_omni.workers.utils.cohort_lease import CohortPlan, ShardIdentity, collective_update


def rank_run(rank, rendezvous, output, stale):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=25))
    try:
        baseline, actor = CPUBackend(), TrackedCPU()
        baseline.initialize()
        actor.initialize()
        tracker = actor.enable_parameter_tracking()
        for step in range(3):
            versions = [None, None]
            dist.all_gather_object(versions, tracker.version())
            identity = ShardIdentity(("sample-0", "sample-1"), "a" * 64, "b" * 64, "c" * 64)
            plan = CohortPlan(f"step-{step}", "train", ((0, 1),), (identity,), tuple(versions))
            if stale and rank == 0:
                actor.load_checkpoint(copy.deepcopy(actor.model.state_dict()))
            batch = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 10
            if stale:
                with pytest.raises(RuntimeError, match="rejected"):
                    with collective_update(tracker, plan, identity, 0, group=dist.group.WORLD):
                        actor.backward(batch)
                        actor.optimizer_step()
                assert actor.calls == 0
                break
            baseline.backward(batch)
            baseline.optimizer_step()
            with collective_update(tracker, plan, identity, 0, group=dist.group.WORLD):
                actor.backward(batch)
                actor.optimizer_step()
            assert_equal(baseline.model.state_dict(), actor.model.state_dict())
            assert_equal(baseline.optimizer.state_dict(), actor.optimizer.state_dict())
        Path(output, f"{rank}.json").write_text(json.dumps({"optimizer_calls": actor.calls}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stale", [False, True])
def test_collective_checks_complete_before_any_rank_updates(tmp_path, stale):
    torch.set_num_threads(1)
    # CPU-only fork inherits the isolated utility import namespace used by this suite.
    mp.start_processes(
        rank_run, args=(f"file://{tmp_path}/rendezvous", str(tmp_path), stale), nprocs=2, join=True, start_method="fork"
    )
    for rank in range(2):
        report = json.loads((tmp_path / f"{rank}.json").read_text())
        assert report["optimizer_calls"] == (0 if stale else 3)
