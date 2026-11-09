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

"""CPU/Gloo placement for the same frozen two-rank reward execution contract."""

import os
import uuid

import ray
import torch.distributed as dist
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from transformers import CLIPProcessor
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import get_master_addr_port

from .native_fsdp import (
    _FORWARD_ADAPTERS,
    _PICKSCORE_EXECUTOR,
    FSDPRewardReplica,
    PickScoreFSDPRank,
    terminate_ranks,
    validate_timeout,
)


def load_cpu_state_dict(model, full_state):
    """Use PyTorch's public loader because pinned verl materializes on an accelerator."""
    if dist.get_rank() == 0:
        model.to("cpu")
    else:
        model.to_empty(device="cpu")
    set_model_state_dict(model, full_state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    for _, buffer in sorted(model.named_buffers()):
        dist.broadcast(buffer, src=0)


def build_cpu_replica(
    model_path,
    processor_path,
    timeout=120,
    collective_timeout=30,
    *,
    executor_model=_PICKSCORE_EXECUTOR,
    cpus_per_rank=1,
):
    """Place two CPU ranks on the controller's node without reserving accelerator bundles."""
    validate_timeout(timeout)
    validate_timeout(collective_timeout, "collective_timeout")
    if executor_model not in _FORWARD_ADAPTERS:
        raise ValueError(f"Unsupported FSDP adapter {executor_model!r}")
    if type(cpus_per_rank) is not int or cpus_per_rank <= 0:
        raise ValueError("cpus_per_rank must be a positive integer")
    processor = CLIPProcessor.from_pretrained(processor_path)
    strategy = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    address, port = ray.get(get_master_addr_port.options(scheduling_strategy=strategy).remote(), timeout=timeout)
    workers = []
    prefix = f"native_fsdp_cpu_{uuid.uuid4().hex[:8]}"
    try:
        for rank in range(2):
            env = {
                "RANK": str(rank),
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "2",
                "MASTER_ADDR": address,
                "MASTER_PORT": str(port),
                "WG_PREFIX": prefix,
                "WG_BACKEND": "ray",
            }
            # CPU ranks have no accelerator IDs for Worker's optional manual device setup.
            env.update({key: "" for key in os.environ if key.startswith("RAY_EXPERIMENTAL_NOSET_")})
            workers.append(
                ray.remote(PickScoreFSDPRank)
                .options(
                    num_cpus=cpus_per_rank,
                    num_gpus=0,
                    scheduling_strategy=strategy,
                    runtime_env={"env_vars": env},
                    name=f"{prefix}_{rank}",
                )
                .remote(model_path, collective_timeout, executor_model, "cpu", cpus_per_rank)
            )
        group = RayWorkerGroup(
            worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(ray.remote(PickScoreFSDPRank)), use_gpu=False
        )
        return FSDPRewardReplica(group, processor, timeout)
    except BaseException as primary:
        try:
            terminate_ranks(workers, min(timeout, 10))
        except BaseException as cleanup:
            raise BaseExceptionGroup("CPU FSDP construction and cleanup failed", [primary, cleanup]) from None
        raise
