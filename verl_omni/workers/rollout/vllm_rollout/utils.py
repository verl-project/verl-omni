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
from typing import Any

import torch
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack
from verl_omni.workers.rollout.vllm_rollout.npu_utils import NPUColocateWorkerMixin

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_weight_sync_zmq_handle(rank: int, default_handle: str) -> str:
    """Return an optional fixed ZMQ handle for split train/vLLM placement."""
    handles = os.getenv("VERL_VLLM_WEIGHT_SYNC_ZMQ_HANDLES", "").strip()
    if not handles:
        return default_handle

    parts = [part.strip() for part in handles.replace(";", ",").split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if rank < 0 or rank >= len(parts):
        raise ValueError(
            "VERL_VLLM_WEIGHT_SYNC_ZMQ_HANDLES must contain either one handle or one handle per rank; "
            f"got {len(parts)} handles and rank={rank}"
        )
    return parts[rank]


def _vllm_lora_enabled(worker: Any) -> bool:
    model_runner = getattr(worker, "model_runner", None)
    if getattr(model_runner, "lora_config", None) is not None:
        return True

    vllm_config = getattr(model_runner, "vllm_config", None) or getattr(worker, "vllm_config", None)
    if vllm_config is None:
        return True
    return getattr(vllm_config, "lora_config", None) is not None


class vLLMOmniColocateWorkerExtension(NPUColocateWorkerMixin, CustomPipelineWorkerExtension):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. NPU (Ascend) memory-pool, sleep, and wake_up — via NPUColocateWorkerMixin
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        adapter_update = bool(peft_config and base_sync_done)
        lora_enabled = _vllm_lora_enabled(self)

        # In async mode, make sure the old lora is removed before adding the new one.
        if adapter_update and lora_enabled:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        if adapter_update and not lora_enabled:
            logger.info("Draining adapter-only weight update because vLLM-Omni LoRA is disabled")
            receiver.receive_weights(on_bucket_received=lambda _weights: None)
            return

        if adapter_update:
            accumulated_weights: list[tuple[str, torch.Tensor]] = []

            def _accumulate(weights: list[tuple[str, torch.Tensor]]) -> None:
                accumulated_weights.extend(weights)

            receiver.receive_weights(on_bucket_received=_accumulate)
            try:
                self._update_weights(
                    accumulated_weights,
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                )
            finally:
                accumulated_weights.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            receiver.receive_weights(
                on_bucket_received=lambda weights: self._update_weights(
                    weights, peft_config=peft_config, base_sync_done=base_sync_done
                )
            )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            if not _vllm_lora_enabled(self):
                logger.info("Skipping adapter-only weight update because vLLM-Omni LoRA is disabled")
                return
            weights = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            lora_request.lora_tensors = None
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
        default_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"
        rank = int(getattr(self, "rank", getattr(self, "local_rank", 0)))
        return get_weight_sync_zmq_handle(rank, default_handle)
