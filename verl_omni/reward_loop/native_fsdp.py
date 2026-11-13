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
"""Synchronous two-rank FSDP2 inference for a named PickScore deployment.

One verl RayWorkerGroup is one logical replica, with one process per shard.
The controller owns ordering and group termination. PickScore owns its forward;
the existing score function still owns score interpretation. The unsharded native executor contract remains unchanged.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import os
import threading
import time
import uuid

import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

from verl_omni.utils.reward_score.image_preference_utils import clip_feature_tensor

from .accelerator_reward_workers import _IndexedResourcePool
from .reward_model import NativeManagedRewardModel
from .reward_model_executor import _await_owned

logger = logging.getLogger(__name__)


class NativeFSDPGroupError(RuntimeError):
    """A logical replica is unusable; no rank-local retry is permitted."""


def validate_executor_config(executor):
    """Validate the model-specific FSDP adapter outside shared worker config."""
    if executor.model != "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel":
        raise ValueError("The initial native FSDP adapter supports PickScoreNativeModel only")
    if set(executor.kwargs) - {"processor_path"}:
        raise ValueError("Native FSDP executor.kwargs supports processor_path only (float32 inference)")


def validate_timeout(value, name="timeout"):
    """Reject non-finite or non-positive deadlines before allocating workers."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def terminate_ranks(workers, timeout, status_method="status"):
    """Attempt every termination and confirm every owned actor has exited."""
    errors = []
    for worker in workers:
        try:
            ray.kill(worker, no_restart=True)
        except BaseException as exc:
            errors.append(exc)
    deadline = time.monotonic() + timeout
    for worker in workers:
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Unconfirmed FSDP rank termination")
                try:
                    ray.get(getattr(worker, status_method).remote(), timeout=remaining)
                except ray.exceptions.ActorDiedError:
                    break
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise BaseExceptionGroup("FSDP rank termination failed; job termination is required", errors)
    logger.info("Native FSDP: confirmed termination of %d rank actors", len(workers))


class PickScoreForward(torch.nn.Module):
    """Put CLIP's feature methods behind one FSDP-managed forward boundary."""

    def __init__(self, clip):
        super().__init__()
        self.clip = clip.eval().requires_grad_(False)
        # FSDP2 rejects scalar parameters. Preserve the value in a length-one tensor.
        self.clip.logit_scale = torch.nn.Parameter(clip.logit_scale.detach().reshape(1), requires_grad=False)
        self.config = clip.config

    def forward(self, image_inputs, text_inputs):
        """Return raw paired features, retaining the existing native infer contract."""
        return {
            "image_embeddings": clip_feature_tensor(self.clip.get_image_features(**image_inputs)),
            "text_embeddings": clip_feature_tensor(self.clip.get_text_features(**text_inputs)),
            "logit_scale": self.clip.logit_scale.exp().squeeze(0),
        }


class PickScoreFSDPRank(Worker):
    """One shard process; every rank receives every command in the same order."""

    def __init__(self, model_path: str, collective_timeout: int = 30):
        super().__init__()
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        self.model_path = model_path
        self.model = None
        get_torch_device().set_device(0)
        initialize_global_process_group_ray(timeout_second=collective_timeout)
        self.mesh = init_device_mesh(get_device_name(), (dist.get_world_size(),))
        self.sequence = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def wake_up(self):
        """Load once on rank zero, construct other ranks on meta, then shard."""
        if self.model is None:
            if dist.get_rank() == 0:
                clip = CLIPModel.from_pretrained(self.model_path, dtype=torch.float32)
            else:
                config = CLIPConfig.from_pretrained(self.model_path)
                with torch.device("meta"):
                    clip = CLIPModel(config)
            model = PickScoreForward(clip)
            full_state = model.state_dict() if dist.get_rank() == 0 else {}
            apply_fsdp2(
                model,
                {"mesh": self.mesh, "reshard_after_forward": True},
                {"wrap_policy": {"transformer_layer_cls_to_wrap": ["CLIPEncoderLayer"]}},
            )
            fsdp2_load_full_state_dict(model, full_state, device_mesh=self.mesh)
            self.model = model
        return self.status()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def infer(self, sequence: int, inputs):
        """Run the same complete batch on all ranks and publish only rank zero."""
        if self.model is None:
            raise RuntimeError("FSDP replica is asleep")
        if sequence != self.sequence:
            raise RuntimeError(f"Out-of-order request: expected {self.sequence}, got {sequence}")
        self.sequence += 1
        device = torch.device(get_device_name(), 0)
        inputs = {key: {name: tensor.to(device) for name, tensor in value.items()} for key, value in inputs.items()}
        # Use no_grad: inference_mode disables version tracking needed by some FSDP versions.
        with torch.no_grad():
            output = self.model(**inputs)
        # Explicitly release the root's gathered parameters in forward-only mode too.
        self.model.reshard()
        get_torch_device().synchronize()
        return {key: value.detach().cpu() for key, value in output.items()} if dist.get_rank() == 0 else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sleep(self):
        """Drop weights on every rank; wake reloads the immutable checkpoint."""
        self.model = None
        gc.collect()
        get_torch_device().empty_cache()
        return self.status()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def status(self):
        """Expose local shard sizes and rank identity for acceptance checks."""
        parameters = list(self.model.parameters()) if self.model is not None else []
        return {
            "rank": dist.get_rank(),
            "pid": os.getpid(),
            "gpu_uuid": str(get_torch_device().get_device_properties(0).uuid),
            "world_size": dist.get_world_size(),
            "sequence": self.sequence,
            "awake": self.model is not None,
            "all_parameters_sharded": bool(parameters) and all(isinstance(p, DTensor) for p in parameters),
            "local_numel": sum(p.to_local().numel() for p in parameters),
            "global_numel": sum(p.numel() for p in parameters),
            "allocated_bytes": get_torch_device().memory_allocated(),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def close(self):
        """Release weights and the owned process group after synchronous drain."""
        self.sleep()
        dist.destroy_process_group()


class FSDPRewardReplica:
    """Serialize synchronous operations on one dedicated two-rank worker group.

    A rank failure, timeout, or partial submission invalidates the whole replica.
    All owned actors are terminated; this object never retries a collective or
    turns a failed group into a successful partial result.
    """

    def __init__(self, worker_group, processor, timeout: float = 120):
        if worker_group.world_size != 2:
            raise ValueError("The initial FSDP prototype requires exactly two ranks")
        validate_timeout(timeout)
        self.worker_group = worker_group
        self.processor = processor
        self.timeout = timeout
        self._lock = threading.Lock()
        self._closed = False
        self._awake = False
        self._sequence = 0
        self._termination_error = None

    def _terminate(self):
        self._closed = True
        self._awake = False
        try:
            terminate_ranks(self.worker_group.workers, min(self.timeout, 10))
        except BaseException as exc:
            self._termination_error = exc
            raise

    def _run(self, method, *args):
        try:
            refs = getattr(self.worker_group, method)(*args)
            return ray.get(refs, timeout=self.timeout)
        except BaseException as exc:
            try:
                self._terminate()
            except BaseException as cleanup:
                raise BaseExceptionGroup("FSDP operation and cleanup failed", [exc, cleanup]) from None
            if not isinstance(exc, Exception):
                raise
            raise NativeFSDPGroupError(f"FSDP {method} failed; the entire replica was terminated") from exc

    def wake_up(self):
        """Wake all ranks or invalidate the complete group."""
        with self._lock:
            if self._closed:
                raise NativeFSDPGroupError("FSDP replica is closed")
            result = self._run("wake_up")
            self._awake = True
            logger.info("Native FSDP group ready: %s", result)
            return result

    def infer(self, prompts, images):
        """Preprocess before collective admission and return one raw model result."""
        with self._lock:
            if self._closed or not self._awake:
                raise NativeFSDPGroupError("FSDP replica is not awake")
            if not prompts or len(prompts) != len(images):
                raise ValueError("Expected equally sized, non-empty prompts and images")
            inputs = {
                "image_inputs": dict(self.processor(images=images, return_tensors="pt")),
                "text_inputs": dict(
                    self.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                ),
            }
            results = self._run("infer", self._sequence, inputs)
            self._sequence += 1
            if len(results) != 2 or results[0] is None or results[1] is not None:
                self._terminate()
                raise RuntimeError("Expected one result from logical replica rank zero")
            return results[0]

    def sleep(self):
        """Wait behind accepted scoring, then unload every shard."""
        with self._lock:
            if self._closed:
                raise RuntimeError("FSDP replica is closed")
            result = self._run("sleep")
            self._awake = False
            return result

    def close(self):
        """Drain, close, and terminate only this replica's actors."""
        with self._lock:
            if self._closed:
                if self._termination_error is not None:
                    raise self._termination_error
                return
            try:
                self._run("close")
            finally:
                if not self._closed:
                    self._terminate()


class NativeFSDPExecutor:
    """Adapt one ordered group to the existing native scorer/lifecycle interface."""

    def __init__(self, replica):
        self.replica = replica

    def reward_kwargs(self):
        """Keep score interpretation in the configured reward function."""
        return {"reward_model": self}

    async def infer(self, prompts, images):
        """Retain ownership through caller cancellation and return per-sample outputs."""
        output = await _await_owned(asyncio.to_thread(self.replica.infer, prompts, images))
        return [
            {
                "text_embeddings": output["text_embeddings"][index],
                "image_embeddings": output["image_embeddings"][index],
                "logit_scale": output["logit_scale"],
            }
            for index in range(len(prompts))
        ]

    async def wake_up(self):
        """Wake every rank before returning to the lifecycle owner."""
        return await _await_owned(asyncio.to_thread(self.replica.wake_up))

    async def sleep(self):
        """Drain and unload every rank before returning."""
        return await _await_owned(asyncio.to_thread(self.replica.sleep))

    async def close(self):
        """Drain, close, and confirm rank termination."""
        return await _await_owned(asyncio.to_thread(self.replica.close))


class NativeFSDPManagedRewardModel(NativeManagedRewardModel):
    """Retain termination ownership outside the scoring facade's failure domain."""

    def __init__(self, name, model):
        super().__init__(name, model)
        self.timeout = model.fsdp.timeout
        self.rank_workers = []
        self._termination_lock = threading.Lock()
        self._termination_error = None

    def bind_rank_workers(self, workers):
        """Retain rank handles without creating a second collective controller."""
        self.rank_workers = list(workers)

    def _terminate_owned(self):
        with self._termination_lock:
            if self._closed:
                if self._termination_error is not None:
                    raise self._termination_error
                return
            self._closed = True
            self._resident = False
            errors = []
            # Stop facade submissions before terminating possibly active collectives.
            for workers, probe in ((self._workers or [], "ready"), (self.rank_workers, "status")):
                try:
                    terminate_ranks(workers, min(self.timeout, 10), status_method=probe)
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                self._termination_error = BaseExceptionGroup("FSDP owner cleanup failed", errors)
                raise self._termination_error

    async def _watch_facade(self):
        while True:
            await asyncio.sleep(min(1, self.timeout))
            # Async scoring leaves the facade event loop available for this health RPC.
            await asyncio.wait_for(self._workers[0].ready.remote(), timeout=self.timeout)

    async def run_rpc(self, method, *args):
        """Bound facade unresponsiveness and settle an accepted RPC after termination."""
        result = health = None
        try:
            if self._closed:
                raise NativeFSDPGroupError("FSDP deployment is closed")
            result = asyncio.ensure_future(getattr(self._workers[0], method).remote(*args))
            health = asyncio.create_task(self._watch_facade())
            done, _ = await asyncio.wait((result, health), return_when=asyncio.FIRST_COMPLETED)
            if health in done:
                health.result()
            return result.result()
        except BaseException as primary:
            try:
                await _await_owned(asyncio.to_thread(self._terminate_owned))
            except BaseException as cleanup:
                raise BaseExceptionGroup("FSDP facade operation and cleanup failed", [primary, cleanup]) from None
            raise
        finally:
            pending = [task for task in (result, health) if task is not None]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run_worker_lifecycle(self, method):
        if self._workers is None:
            raise RuntimeError(f"Native reward model {self.name!r} has no bound workers")
        await self.run_rpc(method, self.name)
        if method == "close_reward_model":
            await _await_owned(asyncio.to_thread(self._terminate_owned))

    async def _close(self):
        """Wait for active termination before reporting its retained error."""
        await super()._close()
        await asyncio.to_thread(self._terminate_owned)


def attach_replica(workers, processor_path, timeout):
    """Bind the CPU scoring facade to existing rank actors without allocating ranks."""
    group = RayWorkerGroup(
        worker_handles=workers,
        ray_cls_with_init=RayClassWithInitArgs(ray.remote(PickScoreFSDPRank)),
        use_gpu=False,
    )
    return FSDPRewardReplica(group, CLIPProcessor.from_pretrained(processor_path), timeout)


class _RankPool(_IndexedResourcePool):
    def __call__(self):
        return self.store

    @property
    def store(self):
        return [2]


class _RankWorkerGroup(RayWorkerGroup):
    """Keep verl rank setup while mapping ranks to explicit single-node bundles."""

    def __init__(self, *args, cleanup_timeout, **kwargs):
        self._workers = []
        try:
            super().__init__(*args, **kwargs)
        except BaseException as exc:
            try:
                terminate_ranks(self._workers, cleanup_timeout)
            except BaseException as cleanup:
                raise BaseExceptionGroup("FSDP construction and rollback failed", [exc, cleanup]) from None
            raise

    def _get_master_addr_port(self, pg, bundle_index=0, master_port_range=None):
        return super()._get_master_addr_port(pg, self.resource_pool.bundle_indices[0], master_port_range)

    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached):
        return super()._create_worker(
            rank, pg_idx, pg, resource_pool.bundle_indices[rank], resource_pool, ray_cls_with_init, worker_env, detached
        )


def build_replica(
    resource_pool,
    model_path: str,
    processor_path: str,
    timeout: float = 120,
    collective_timeout: float = 30,
    bundle_indices=None,
    _rank_class=None,
):
    """Build one two-rank replica; the caller retains ownership of the parent pool."""
    validate_timeout(timeout)
    validate_timeout(collective_timeout, "collective_timeout")
    if not is_cuda_available or not resource_pool.use_gpu or len(resource_pool.store) != 1:
        raise ValueError("FSDP rewards require a dedicated single-node GPU pool")
    indices = list(range(resource_pool.world_size)) if bundle_indices is None else list(bundle_indices)
    if (
        len(indices) != 2
        or len(set(indices)) != 2
        or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < resource_pool.world_size
            for index in indices
        )
    ):
        raise ValueError("FSDP requires exactly two distinct valid parent-pool bundle indices")
    processor = CLIPProcessor.from_pretrained(processor_path)
    group = _RankWorkerGroup(
        resource_pool=_RankPool(resource_pool, indices),
        ray_cls_with_init=RayClassWithInitArgs(
            ray.remote(_rank_class or PickScoreFSDPRank), model_path, collective_timeout
        ),
        name_prefix=f"native_fsdp_{uuid.uuid4().hex[:8]}",
        device_name=get_device_name(),
        cleanup_timeout=min(timeout, 10),
    )
    return FSDPRewardReplica(group, processor, timeout)
