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

import gc
import logging
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch
from vllm.utils.mem_utils import GiB_bytes

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def _is_npu_platform() -> bool:
    """Return True when vLLM is running on an Ascend NPU device."""
    try:
        from vllm.platforms import current_platform

        return current_platform.device_type == "npu"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NPU memory allocator
# ---------------------------------------------------------------------------


def _get_npu_memory_allocator():
    """Return the singleton CaMemAllocator instance for NPU memory pools."""
    from vllm_ascend.device_allocator.camem import CaMemAllocator

    return CaMemAllocator.get_instance()


# ---------------------------------------------------------------------------
# Permanent patch: suppress diffusers empty-cache calls on NPU
# ---------------------------------------------------------------------------

_empty_cache_patch_applied = False


def _apply_permanent_empty_cache_patch():
    """Permanently patch diffusers to skip NPU ``empty_device_cache`` calls.

    On Ascend NPU, calling ``empty_device_cache`` while a CaMemAllocator
    memory pool is active invalidates the pool's internal bookkeeping.
    Since the memory pool persists for the lifetime of the worker (not just
    during the ``use_memory_pool`` context manager), the patch must remain
    active during generation as well, not only during weight loading.
    """
    global _empty_cache_patch_applied
    if _empty_cache_patch_applied:
        return
    _empty_cache_patch_applied = True

    try:
        from diffusers.models import modeling_utils
        from diffusers.utils import torch_utils
    except Exception:
        return

    original_torch_empty_cache = torch_utils.empty_device_cache

    def empty_device_cache(device_type: str | None = None):
        if device_type is None or device_type == "npu":
            return
        return original_torch_empty_cache(device_type)

    modeling_utils.empty_device_cache = empty_device_cache
    torch_utils.empty_device_cache = empty_device_cache


@contextmanager
def _skip_diffusers_npu_empty_cache():
    """Temporarily patch diffusers so that NPU empty-cache calls are skipped.

    On Ascend NPU, calling ``empty_device_cache`` while inside a CaMemAllocator
    memory pool invalidates the pool's internal bookkeeping.  This context
    manager monkey-patches the two relevant diffusers helpers for the duration
    of a ``with`` block and restores the originals on exit.

    Note: :func:`_apply_permanent_empty_cache_patch` supersedes this context
    manager for the common case, but it is kept for backward compatibility.
    """
    try:
        from diffusers.models import modeling_utils
        from diffusers.utils import torch_utils
    except Exception:
        yield
        return

    original_modeling_empty_cache = modeling_utils.empty_device_cache
    original_torch_empty_cache = torch_utils.empty_device_cache

    def empty_device_cache(device_type: str | None = None):
        if device_type is None or device_type == "npu":
            return
        return original_torch_empty_cache(device_type)

    modeling_utils.empty_device_cache = empty_device_cache
    torch_utils.empty_device_cache = empty_device_cache
    try:
        yield
    finally:
        modeling_utils.empty_device_cache = original_modeling_empty_cache
        torch_utils.empty_device_cache = original_torch_empty_cache


# ---------------------------------------------------------------------------
# NPU sleep helper: unmap only specific tags
# ---------------------------------------------------------------------------


def _npu_sleep_tags_only(allocator, offload_tags: tuple[str, ...], unmap_tags: set[str]):
    """Sleep only the specified tags, avoiding double-unmapping.

    ``CaMemAllocator.sleep()`` unmaps **all** non-persistent allocations, but
    ``CaMemAllocator.wake_up(tags=...)`` only remaps a subset.  On a subsequent
    ``sleep()`` call, unmapping already-unmapped allocations causes
    ``aclrtUnmapMem`` to fail with error 507899 on Ascend NPU.

    This helper iterates over ``allocator.pointer_to_data`` and unmaps only the
    allocations whose tag is in *unmap_tags*, creating CPU backups for those in
    *offload_tags* — mirroring the CaMemAllocator's own logic but scoped to a
    subset of tags.
    """
    from acl.rt import memcpy as acl_memcpy
    from vllm_ascend.device_allocator.camem import CaMemAllocator, unmap_and_release

    sleep_persistent = CaMemAllocator.sleep_persistent_tag

    for ptr, data in allocator.pointer_to_data.items():
        if data.tag == sleep_persistent:
            continue
        if data.tag not in unmap_tags:
            continue
        handle = data.handle
        if data.tag in offload_tags and data.cpu_backup_tensor is None:
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=False)
            cpu_ptr = cpu_backup_tensor.data_ptr()
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            dest_max = cpu_ptr + size_in_bytes * 2
            acl_memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
            data.cpu_backup_tensor = cpu_backup_tensor
        unmap_and_release(handle)

    gc.collect()
    torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# Mixin: NPU-specific overrides for vLLMOmniColocateWorkerExtension
# ---------------------------------------------------------------------------


class vLLMOmniNPUColocateWorkerExtension(vLLMOmniColocateWorkerExtension):
    """Mixin that overrides memory-pool, sleep, and wake_up on Ascend NPU.
    The mixin guards every method with ``_is_npu_platform()`` and falls back to
    the super-class implementation on non-NPU hardware, so it is safe to use
    unconditionally in a cross-platform codebase.

    # TODO (long): Once vLLM-Omni provides first-class NPU support in
    ``CustomPipelineWorkerExtension``, this mixin can be removed and these
    methods can be deleted from verl_omni entirely.
    """

    # Track whether the first sleep has been performed.  On the first sleep
    # all pool allocations are mapped (from weight loading), so the standard
    # ``CaMemAllocator.sleep()`` is safe.  On subsequent sleeps only the tags
    # that were remapped via ``wake_up`` should be unmapped.
    _npu_first_sleep_done: bool = False
    # Tags that have been remapped (via ``wake_up``) since the last sleep.
    _npu_remapped_tags: set = set()

    def __new__(cls, **kwargs):
        if _is_npu_platform():
            _apply_permanent_empty_cache_patch()
        return super().__new__(cls, **kwargs)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if not _is_npu_platform():
            return super()._maybe_get_memory_pool_context(tag)

        if not self.od_config.enable_sleep_mode:
            return nullcontext()

        allocator = _get_npu_memory_allocator()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."

        # The permanent empty-cache patch is applied in __new__; use the
        # context manager as an extra safety net for code paths that might
        # still call the originals directly.
        @contextmanager
        def npu_memory_pool_context():
            with _skip_diffusers_npu_empty_cache(), allocator.use_memory_pool(tag=tag):
                yield

        return npu_memory_pool_context()

    def sleep(self, level: int = 1) -> bool:
        if not _is_npu_platform():
            return super().sleep(level)

        free_bytes_before_sleep = None
        try:
            free_bytes_before_sleep = torch.npu.mem_get_info()[0]
        except Exception:
            pass

        if level == 2 and self.model_runner is not None:
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = _get_npu_memory_allocator()
        offload_tags = ("weights",) if level == 1 else tuple()

        if not self._npu_first_sleep_done:
            # First sleep: all pool allocations are mapped (from weight
            # loading), so the standard CaMemAllocator.sleep() — which unmaps
            # every non-persistent allocation — is safe.
            allocator.sleep(offload_tags=offload_tags)
            self._npu_first_sleep_done = True
        else:
            # Subsequent sleeps: CaMemAllocator.sleep() would try to unmap
            # ALL non-persistent allocations, but wake_up() only remapped a
            # subset.  Unmapping already-unmapped memory causes
            # ``aclrtUnmapMem`` to fail with error 507899 on Ascend NPU.
            # Only unmap the tags that were actually remapped since the last
            # sleep.
            _npu_sleep_tags_only(
                allocator,
                offload_tags=offload_tags,
                unmap_tags=self._npu_remapped_tags,
            )
        self._npu_remapped_tags = set()

        # CaMemAllocator.sleep() frees physical pages via aclrtFreePhysical,
        # but PyTorch's default caching allocator may still hold reserved
        # blocks from tensors allocated outside the pluggable allocator's
        # memory pool (e.g. during vLLM generation). These blocks are invisible
        # to the pluggable allocator and must be released separately so the
        # training worker can use the full NPU memory budget.
        gc.collect()
        torch.npu.empty_cache()

        if free_bytes_before_sleep is not None:
            try:
                free_bytes_after_sleep, total = torch.npu.mem_get_info()
                freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
                used_bytes = total - free_bytes_after_sleep
                logger.info(
                    "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
                    freed_bytes / GiB_bytes,
                    used_bytes / GiB_bytes,
                )
            except Exception:
                pass
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        if not _is_npu_platform():
            return super().wake_up(tags)

        allocator = _get_npu_memory_allocator()
        allocator.wake_up(tags=tags)

        # Track which tags have been remapped so the next sleep() knows which
        # allocations are currently mapped and safe to unmap.
        if tags is None:
            all_tags = {
                data.tag for data in allocator.pointer_to_data.values() if data.tag != allocator.sleep_persistent_tag
            }
            self._npu_remapped_tags.update(all_tags)
        else:
            self._npu_remapped_tags.update(tags)

        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True
