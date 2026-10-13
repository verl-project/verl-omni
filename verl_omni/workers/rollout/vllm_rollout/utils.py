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
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import make_update_zmq_handle

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# AR engine classes needing the MoE weight-loader patch; add new MoE omni models here.
SUPPORTED_MOE_MODELS: list[type] = []
try:
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(Qwen3OmniMoeForConditionalGeneration)
except ImportError:
    pass


def _is_moe_engine(model) -> bool:
    """True when the (possibly ACLGraph-wrapped) engine class is whitelisted."""
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(type(model)):
        model = model.runnable
    return isinstance(model, tuple(SUPPORTED_MOE_MODELS))


def _split_visible_devices(value: str) -> list[str]:
    """Split a visible-devices env value into stripped, non-empty entries."""
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _model_has_fused_moe(model) -> bool:
    """Whether the rollout model contains vLLM ``RoutedExperts`` (fused-MoE) modules."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    return any(isinstance(layer, RoutedExperts) for layer in model.modules())


def _bucket_is_delta_flush(weights: list) -> bool:
    """Whether a full-weight stream's bucket belongs to the delta flush wire.

    The ``omni_delta_sharded`` engine rides the stock named_tensors channel: its
    flush sentinels are flattened into the stream with a ``#<flush>`` suffix (see
    ``OmniDeltaShardedCheckpointEngine.receive_weights``), and the spec sentinel
    always leads a flush, so the first bucket's first entry decides the routing.
    Real checkpoint tensor names never start with ``__delta_``.
    """
    from verl_omni.workers.rollout.vllm_rollout.delta_apply import SPEC_NAME

    return bool(weights) and weights[0][0].partition("#")[0] == SPEC_NAME


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

        vllm_config = kwargs.get("vllm_config")
        if vllm_config is not None and any(
            architecture in {"Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeThinkerForConditionalGeneration"}
            for architecture in vllm_config.model_config.architectures
        ):
            from verl_omni.utils.vllm_omni.patch import patch_qwen3_omni_thinker_forward

            patch_qwen3_omni_thinker_forward()

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

    def monkey_patch_model(self) -> None:
        # startup MoE weight-loader patch; re-attached per sync in update_weights_from_ipc
        standard = self._get_standard_weight_model_and_config()
        if standard is not None and _is_moe_engine(standard[0]):
            patch_vllm_moe_model_weight_loader(standard[0])

    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        zmq_update_id: str | None = None,
        delta_flush: bool | None = None,
    ):
        """Update the weights of the rollout model.

        For LoRA updates, all LoRA tensors are accumulated across buckets and loaded
        atomically via a single ``add_lora`` call, avoiding per-bucket partial loading.
        For full-weight updates, weights are streamed bucket-by-bucket via
        ``load_weights`` to keep GPU memory usage bounded.
        The ``omni_delta_sharded`` engine streams its flushes over the same bucketed
        channel as sentinel-named tensors: ``delta_flush=None`` (the stock ServerAdapter
        path, which passes no flag) sniffs the first bucket, and a ``__delta_spec__``
        sentinel routes the stream to the in-place delta apply
        (:mod:`verl_omni.workers.rollout.vllm_rollout.delta_apply`).
        """

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if peft_config is None and self._pending_lora_peft_config is not None:
            peft_config = self._pending_lora_peft_config
            base_sync_done = True
            # Consume the stash so a subsequent full-weight sync isn't misrouted.
            self._pending_lora_peft_config = None

        if delta_flush and peft_config is not None:
            raise ValueError(
                "delta_sharded weight sync does not apply LoRA adapters; "
                "run full-weight training or use a non-delta checkpoint engine backend."
            )

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
            # The stream may instead carry omni_delta_sharded flush sentinels:
            # delta_flush=None sniffs the first bucket (the stock ServerAdapter
            # passes no flag); delta_flush=True skips the sniff and the dense
            # preparation entirely.
            logger.info("Loading standard weights (async)")
            if delta_flush is True:
                delta_state: dict = {}
                receiver.receive_weights(
                    on_bucket_received=lambda weights, is_last=False, *args, **kwargs: self._apply_delta_bucket(
                        weights, delta_state, is_last=is_last
                    )
                )
                self._finish_delta_stream(delta_state)
                return

            delta_ctx = {"delta": None}

            def _route_bucket(weights, is_last=False, *args, **kwargs):
                # delta_ctx carries the sniffed routing flag and, once delta is
                # detected, the lazy applier state (see _apply_delta_bucket).
                # A zero-flush sync still delivers one empty terminal bucket; that
                # must stay a no-op, not a dense load_weights([]) / post-load pass.
                # Once a delta stream has started, an empty trailing bucket still
                # has to reach the applier so is_last can close the flush.
                if delta_ctx["delta"] is None:
                    if not weights:
                        return
                    delta_ctx["delta"] = _bucket_is_delta_flush(weights)
                if delta_ctx["delta"]:
                    self._apply_delta_bucket(weights, delta_ctx, is_last=is_last)
                elif weights:
                    dense_on_bucket(weights)

            standard = self._get_standard_weight_model_and_config()
            if standard is not None:
                model, model_config = standard
                # re-attach the MoE weight loader
                if _is_moe_engine(model):
                    patch_vllm_moe_model_weight_loader(model)

                # On Ascend, process_weights_after_loading transposes w13/w2 for
                # fused-MoE compute; revert it so load_weights sees checkpoint-shape
                # params. The post-load process_weights_after_loading re-transposes.
                from verl_omni.workers.rollout.vllm_rollout.npu_utils import (
                    _is_npu_platform,
                    restore_moe_param_layout,
                )

                is_npu = _is_npu_platform()
                # Use checkpoint-layout restoration for packed MoE weights.
                # Dense omni loaders may copy auxiliary encoder buffers and
                # derive runtime tensors inside load_weights; turning those
                # tensors into meta placeholders breaks their loading contract.
                has_moe = not is_npu and _model_has_fused_moe(model)
                if is_npu or not has_moe:
                    if is_npu:
                        restore_moe_param_layout(model, model_config.hf_text_config.hidden_size)

                    def dense_on_bucket(weights):
                        model.load_weights(weights)

                    receiver.receive_weights(on_bucket_received=_route_bucket)
                    if delta_ctx["delta"]:
                        self._finish_delta_stream(delta_ctx)
                    elif delta_ctx["delta"] is False:
                        from vllm.model_executor.model_loader.utils import process_weights_after_loading

                        process_weights_after_loading(model, model_config, self.device)
                else:
                    # vLLM records checkpoint layouts when constructing the
                    # model. Restore those layouts before loading, then copy
                    # processed weights back into the original kernel storage.
                    # A delta stream into a fused-MoE model raises fail-closed
                    # on its first bucket (see _apply_delta_bucket). Start the
                    # reload only after the sniff says dense, so that raise does
                    # not leave the replica inside initialize_layerwise_reload.
                    from vllm.model_executor.model_loader.reload import (
                        finalize_layerwise_reload,
                        initialize_layerwise_reload,
                    )

                    moe_reload = {"started": False}

                    # Layerwise loaders can retain tensors across buckets. The
                    # receiver reuses its IPC buffer, so retained weights must
                    # own their storage until the layer is ready to process.
                    def dense_on_bucket(weights):
                        if not moe_reload["started"]:
                            initialize_layerwise_reload(model)
                            moe_reload["started"] = True
                        model.load_weights([(name, tensor.clone()) for name, tensor in weights])

                    receiver.receive_weights(on_bucket_received=_route_bucket)
                    if delta_ctx["delta"]:
                        self._finish_delta_stream(delta_ctx)
                    elif delta_ctx["delta"] is False:
                        finalize_layerwise_reload(model, model_config)
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

                def dense_on_bucket(weights):
                    load_fn(weights)

                receiver.receive_weights(on_bucket_received=_route_bucket)
                if delta_ctx["delta"]:
                    self._finish_delta_stream(delta_ctx)

    def _apply_delta_bucket(self, weights, state: dict, is_last: bool = False) -> None:
        """Gates + in-place apply for one bucket of the delta flush stream.

        The gates (NPU, fused-MoE) run lazily on the first bucket so the dense
        preparation in :meth:`update_weights_from_ipc` stays untouched for
        full-weight syncs; where a gate fires, the run fails closed at the first
        delta sync. Flush reassembly and the in-place apply live in
        :mod:`verl_omni.workers.rollout.vllm_rollout.delta_apply`.
        """
        from verl_omni.workers.rollout.vllm_rollout.delta_apply import DeltaFlushReceiver
        from verl_omni.workers.rollout.vllm_rollout.npu_utils import _is_npu_platform

        if "applier" not in state:
            if _is_npu_platform():
                # The NPU full-weight path transposes fused-MoE params around the load;
                # HF-coordinate deltas would land in the transposed layout.
                raise NotImplementedError("delta_sharded weight sync is not supported on Ascend NPU yet")
            standard = self._get_standard_weight_model_and_config()
            if standard is not None:
                model, model_config = standard
                # The full-weight path runs fused-MoE rollouts through the checkpoint
                # layout reload dance around load_weights; the sparse in-place delta
                # apply does not reproduce it, so HF-coordinate expert updates would
                # land on runtime-layout fused storage. Fail closed until that path
                # exists (e.g. the Qwen3-Omni thinker).
                if _model_has_fused_moe(model):
                    raise NotImplementedError(
                        "delta_sharded weight sync does not support fused-MoE rollout models yet; "
                        "use a non-delta checkpoint engine backend for this model."
                    )
                load_target = model
                state["model"], state["model_config"] = model, model_config
            else:
                pipeline = getattr(getattr(self, "model_runner", None), "pipeline", None)
                if pipeline is not None and hasattr(pipeline, "load_weights"):
                    load_target = pipeline
                elif hasattr(self, "load_weights"):
                    load_target = self
                else:
                    raise RuntimeError("Diffusion pipeline worker has no load_weights-capable pipeline")
            state["applier"] = DeltaFlushReceiver(load_target)
        state["applier"].on_bucket(weights, is_last=is_last)

    def _finish_delta_stream(self, state: dict) -> None:
        """Post-stream processing for the delta path: the dense seed is a full
        ``load_weights``, so AR models get the same single post-load processing
        pass as the bucketed full-weight sync."""
        if state.get("applier") is not None and state["applier"].saw_seed and "model_config" in state:
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            process_weights_after_loading(state["model"], state["model_config"], self.device)

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
