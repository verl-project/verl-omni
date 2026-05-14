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
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm.utils.mem_utils import GiB_bytes
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _is_npu_platform() -> bool:
    try:
        from vllm.platforms import current_platform

        return current_platform.device_type == "npu"
    except Exception:
        return False


def _get_npu_memory_allocator():
    from vllm_ascend.device_allocator.camem import CaMemAllocator

    return CaMemAllocator.get_instance()


@contextmanager
def _skip_diffusers_npu_empty_cache():
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


class vLLMOmniColocateWorkerExtension(CustomPipelineWorkerExtension):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

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

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            logger.info("Loading standard weights (async)")
            self.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses Ray job id + replica_rank + local_rank to form the handle so it
        matches the sender side regardless of CUDA_VISIBLE_DEVICES differences,
        avoids collisions when multiple replicas share the same node, and is
        unique per Ray job to avoid cross-job collisions on shared hosts. The
        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
        inherited by this vLLM worker subprocess.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"
