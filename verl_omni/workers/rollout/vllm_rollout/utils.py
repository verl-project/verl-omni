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
import time

import torch
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
    2. NPU (Ascend) memory-pool, sleep, and wake_up — via NPUColocateWorkerMixin
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def _get_standard_weight_model_and_config(self):
        """Return ``(model, model_config)`` for the standard (non-LoRA) AR weight path.

        Reaches the underlying vLLM model + ``ModelConfig`` via the worker's
        ``model_runner``. Returns ``None`` for workers without this chain (e.g. the
        diffusion pipeline worker), so the caller falls back to ``self.load_weights``.
        """
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return None
        model = model_runner.get_model() if hasattr(model_runner, "get_model") else getattr(model_runner, "model", None)
        model_config = getattr(model_runner, "model_config", None)
        if model is not None and model_config is not None and hasattr(model, "load_weights"):
            return model, model_config
        return None

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model.

        For LoRA updates, all LoRA tensors are accumulated across buckets and loaded
        atomically via a single ``add_lora`` call, avoiding per-bucket partial loading.
        For full-weight updates, weights are streamed bucket-by-bucket via
        ``load_weights`` to keep GPU memory usage bounded.
        """

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )

        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            t0 = time.perf_counter()
            self.remove_lora(VLLM_LORA_INT_ID)
            t1 = time.perf_counter()
            logger.debug("remove_lora took %.3f ms", (t1 - t0) * 1000)

            # Accumulate all LoRA tensors across buckets (LoRA weights are small;
            # a single atomic ``add_lora`` is both correct for multi-bucket edge
            # cases and more efficient than per-bucket loading).
            t_recv_start = time.perf_counter()
            accumulated_weights: dict[str, torch.Tensor] = {}
            receiver.receive_weights(on_bucket_received=lambda weights: accumulated_weights.update(weights))
            t_recv_end = time.perf_counter()
            logger.debug(
                "IPC receive took %.3f ms (%d params, %.2f MB)",
                (t_recv_end - t_recv_start) * 1000,
                len(accumulated_weights),
                sum(t.element_size() * t.numel() for t in accumulated_weights.values()) / (1024 * 1024),
            )

            # AR (standard vLLM) workers go through verl's base VLLMHijack, which
            # dispatches on ``isinstance(req, TensorLoRARequest)``; diffusion workers
            # go through vllm-omni's DiffusionLoRAManager, which expects the
            # OmniLoRARequest-derived ``OmniTensorLoRARequest``. Pick by worker type.
            if self._get_standard_weight_model_and_config() is not None:
                from verl.utils.vllm.utils import TensorLoRARequest

                lora_request = TensorLoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                    peft_config=peft_config,
                    lora_tensors=accumulated_weights,
                )
            else:
                lora_request = OmniTensorLoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                    peft_config=peft_config,
                    lora_tensors=accumulated_weights,
                )
            t2 = time.perf_counter()
            self.add_lora(lora_request)
            t3 = time.perf_counter()
            logger.debug("add_lora took %.3f ms", (t3 - t2) * 1000)
            logger.debug(
                "LoRA update total: %.3f ms (remove=%.3f, recv=%.3f, add=%.3f)",
                (t3 - t0) * 1000,
                (t1 - t0) * 1000,
                (t_recv_end - t_recv_start) * 1000,
                (t3 - t2) * 1000,
            )
        else:
            # Full-weight path: stream bucket-by-bucket to bound GPU memory.
            logger.info("Loading standard weights (async)")
            standard = self._get_standard_weight_model_and_config()
            if standard is not None:
                # AR (standard vLLM) model: load each bucket via the low-level
                # model.load_weights (no per-bucket finalize), then run the single
                # post-load processing pass once all buckets are received.
                model, model_config = standard
                receiver.receive_weights(on_bucket_received=lambda weights: model.load_weights(weights))
                from vllm.model_executor.model_loader.utils import process_weights_after_loading

                process_weights_after_loading(model, model_config, self.device)
            else:
                # Diffusion pipeline worker: use its own loader.
                receiver.receive_weights(on_bucket_received=lambda weights: self.load_weights(weights))

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
