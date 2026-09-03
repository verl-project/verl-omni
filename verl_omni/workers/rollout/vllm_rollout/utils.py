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
from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import make_update_zmq_handle

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _split_visible_devices(value: str) -> list[str]:
    """Split a visible-devices env value into stripped, non-empty entries."""
    return [entry.strip() for entry in value.split(",") if entry.strip()]


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

    _pending_lora_peft_config: dict | None = None

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def set_pending_lora_peft_config(self, peft_config: dict | None = None):
        """Stash the actor's LoRA ``peft_config`` for the next
        ``update_weights_from_ipc`` call (separate-async NCCL path only).

        Called out-of-band via ``collective_rpc`` by
        ``OmniCheckpointEngineManager`` before the NCCL weight broadcast.
        ``update_weights_from_ipc`` consumes the stash when its ``peft_config``
        kwarg is absent (the standalone rollout path), then clears it so a
        later full-weight sync is not misrouted.
        """
        self._pending_lora_peft_config = peft_config

    def _move_diffusion_lora_stacks_to_device(self) -> None:
        """Move unregistered LoRA stacks to the worker device before execution."""
        # TODO(@NancyFyong): Move this into vLLM-Omni's DiffusionLoRAManager.
        manager = getattr(self, "lora_manager", None)
        for module in getattr(manager, "_lora_modules", {}).values():
            for name in ("lora_a_stacked", "lora_b_stacked"):
                tensors = getattr(module, name, None)
                if tensors is not None:
                    setattr(module, name, tuple(tensor.to(self.device, non_blocking=True) for tensor in tensors))

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

    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        zmq_update_id: str | None = None,
    ):
        """Update the weights of the rollout model.

        For LoRA updates, all LoRA tensors are accumulated across buckets and loaded
        atomically via a single ``add_lora`` call, avoiding per-bucket partial loading.
        For full-weight updates, weights are streamed bucket-by-bucket via
        ``load_weights`` to keep GPU memory usage bounded.
        """

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if peft_config is None and self._pending_lora_peft_config is not None:
            peft_config = self._pending_lora_peft_config
            base_sync_done = True
            # Consume the stash so a subsequent full-weight sync isn't misrouted.
            self._pending_lora_peft_config = None

        if self.device is None:
            raise RuntimeError("Worker device is not set.")
        zmq_handle = self._get_zmq_handle()
        if zmq_update_id is not None:
            zmq_handle = make_update_zmq_handle(zmq_handle, zmq_update_id)
        receiver = BucketedWeightReceiver(
            zmq_handle=zmq_handle,
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
            receiver.receive_weights(
                on_bucket_received=lambda weights, *args, **kwargs: accumulated_weights.update(weights)
            )
            t_recv_end = time.perf_counter()
            lora_total_bytes = sum(t.element_size() * t.numel() for t in accumulated_weights.values())
            logger.debug(
                "IPC receive took %.3f ms (%d params, %.2f MB)",
                (t_recv_end - t_recv_start) * 1000,
                len(accumulated_weights),
                lora_total_bytes / (1024 * 1024),
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
            if self._get_standard_weight_model_and_config() is None:
                self._move_diffusion_lora_stacks_to_device()
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
                # Re-attach weight_loader on Ascend FusedMoE params via verl's
                # built-in patch (handles ACLGraph unwrap + SUPPORTED_MOE_MODELS
                # whitelist, which Qwen3-Omni is registered into via
                # patch_register_vllm_moe_model_weight_loader).
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(model)

                # On Ascend, process_weights_after_loading transposes w13/w2 for
                # fused-MoE compute; revert it so load_weights sees checkpoint-shape
                # params. The post-load process_weights_after_loading re-transposes.
                from verl_omni.workers.rollout.vllm_rollout.npu_utils import (
                    _is_npu_platform,
                    restore_moe_param_layout,
                )

                if _is_npu_platform():
                    restore_moe_param_layout(model, model_config.hf_text_config.hidden_size)
                receiver.receive_weights(
                    on_bucket_received=lambda weights, *args, **kwargs: model.load_weights(weights)
                )
                from vllm.model_executor.model_loader.utils import process_weights_after_loading

                process_weights_after_loading(model, model_config, self.device)
            else:
                # Diffusion pipeline worker: load via the pipeline. vllm-omni
                # 0.26 removed DiffusionWorker/DiffusionModelRunner.load_weights;
                # each pipeline exposes load_weights via AutoWeightsLoader.
                pipeline = getattr(getattr(self, "model_runner", None), "pipeline", None)
                if pipeline is not None and hasattr(pipeline, "load_weights"):
                    load_fn = pipeline.load_weights
                elif hasattr(self, "load_weights"):
                    load_fn = self.load_weights
                else:
                    raise RuntimeError("Diffusion pipeline worker has no load_weights-capable pipeline")
                receiver.receive_weights(on_bucket_received=lambda weights, *args, **kwargs: load_fn(weights))

    def _get_zmq_handle(self) -> str:
        """Get the ZMQ handle matching the co-located trainer actor on this rank.

        The handle is formed from the Ray job id, the replica rank, and the
        node-local rank. ``self.local_rank`` is stage-local in multi-stage
        deploys (each stage is pinned to a GPU subset), so it is remapped
        through the replica-level device list in VERL_ZMQ_BASE_VISIBLE_DEVICES
        to the node-local rank the actor derives: the index of the worker's
        device within the replica list. Falls back to the stage-local rank
        when the lists are absent or the device is not in the replica list.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        local_rank = int(self.local_rank)
        stage_devices = os.environ.get(get_visible_devices_keyword(), "")
        replica_devices = os.environ.get("VERL_ZMQ_BASE_VISIBLE_DEVICES", "")
        if stage_devices and replica_devices:
            stage_entries = _split_visible_devices(stage_devices)
            replica_entries = _split_visible_devices(replica_devices)
            if 0 <= local_rank < len(stage_entries) and stage_entries[local_rank] in replica_entries:
                local_rank = replica_entries.index(stage_entries[local_rank])
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock"
