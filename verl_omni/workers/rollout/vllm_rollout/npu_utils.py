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
# Context manager: suppress diffusers empty-cache calls on NPU
# ---------------------------------------------------------------------------


@contextmanager
def _skip_diffusers_npu_empty_cache():
    """Temporarily patch diffusers so that NPU empty-cache calls are skipped.

    On Ascend NPU, calling ``empty_device_cache`` while inside a CaMemAllocator
    memory pool invalidates the pool's internal bookkeeping.  This context
    manager monkey-patches the two relevant diffusers helpers for the duration
    of a ``with`` block and restores the originals on exit.
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

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if not _is_npu_platform():
            return super()._maybe_get_memory_pool_context(tag)

        if not self.od_config.enable_sleep_mode:
            return nullcontext()

        allocator = _get_npu_memory_allocator()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."

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
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

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

        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True
