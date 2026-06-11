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
import asyncio
import functools
import logging
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh
from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, is_npu_available, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.utils.losses import ppo_loss

from verl_omni.utils.mfu import (
    DiffusionFlopsCounter,
    allgather_diffusion_flops_meta,
    collect_diffusion_flops_meta,
)
from verl_omni.workers.config import (
    DiffusionActorConfig,
    DiffusionModelConfig,
)
from verl_omni.workers.utils.losses import diffusion_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def _timed_await(name: str, timings: dict, coro):
    """Await ``coro`` while recording its wall-clock duration into ``timings``."""
    start = time.perf_counter()
    try:
        return await coro
    finally:
        timings[name] = time.perf_counter() - start


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        # TODO(jhz): Switch to `set_expandable_segments` when the torch_npu library
        # supports `torch.npu.memory._set_allocator_settings`
        if is_npu_available:
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        elif self.config.model_type in ("diffusion_model", "diffusion_dpo_model", "diffusion_nft_model"):
            self.flops_counter = DiffusionFlopsCounter(
                architecture=self.model_config.architecture,
                transformer_config=self.model_config.transformer_config,
            )
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def copy_adapter(self, source: str = "default", target: str = "old"):
        if not hasattr(self.engine, "copy_adapter"):
            raise NotImplementedError(f"Engine {type(self.engine).__name__} does not support copy_adapter.")
        self.engine.copy_adapter(source=source, target=target)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def ema_update_adapter(self, source: str = "default", target: str = "old", decay: float = 0.0):
        if not hasattr(self.engine, "ema_update_adapter"):
            raise NotImplementedError(f"Engine {type(self.engine).__name__} does not support ema_update_adapter.")
        self.engine.ema_update_adapter(source=source, target=target, decay=decay)

    def _postprocess_output(
        self,
        output,
        *,
        global_token_num,
        delta_time,
        forward_only,
        images_seqlens,
        diffusion_flops_meta: Optional[dict] = None,
    ):
        """
        Args:
            output: a dictionary containing loss, model_outputs and metrics
            diffusion_flops_meta: optional dict consumed by ``DiffusionFlopsCounter``
                with keys ``latent_seqlens``, ``prompt_seqlens``, ``num_timesteps``,
                ``num_forward_passes``. Provided only on the diffusion path; ignored
                otherwise.
        """
        # TODO: whether to log memory
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        metrics: dict = output.pop("metrics")
        dp_group = self.engine.get_data_parallel_group()

        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        mfu_divisor = torch.distributed.get_world_size(dp_group) if dp_group is not None else 1
        if isinstance(self.flops_counter, DiffusionFlopsCounter):
            if diffusion_flops_meta is not None:
                # Counter expects global (DP-allgathered) seqlens, matching the
                # convention used for ``global_token_num`` in the LLM path.
                global_meta = allgather_diffusion_flops_meta(
                    diffusion_flops_meta, self.engine.get_data_parallel_group()
                )
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                    delta_time=delta_time, **global_meta
                )
                if promised_flops > 0:
                    final_metrics["mfu"] = estimated_flops / promised_flops / mfu_divisor
                    if forward_only:
                        final_metrics["mfu"] /= 3.0
        elif global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / mfu_divisor
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)
        diffusion_flops_meta = collect_diffusion_flops_meta(
            self.flops_counter,
            data,
            pipeline_config=getattr(self.model_config, "pipeline", None),
        )

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
                diffusion_flops_meta=diffusion_flops_meta,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)
        diffusion_flops_meta = collect_diffusion_flops_meta(
            self.flops_counter,
            data,
            pipeline_config=getattr(self.model_config, "pipeline", None),
        )

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
                diffusion_flops_meta=diffusion_flops_meta,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        self.enable_routing_replay = (
            self.config.actor.strategy == "megatron" and self.config.actor.megatron.router_replay.mode != "disabled"
        )

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig | DiffusionModelConfig = omega_conf_to_dataclass(self.config.model)
        is_diffusion = model_config.get("model_type", "language_model") in (
            "diffusion_model",
            "diffusion_dpo_model",
            "diffusion_nft_model",
        )

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                if not is_diffusion:
                    self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                    self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                    self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop(
                        "log_prob_max_token_len_per_gpu", None
                    )
            ref_config: ActorConfig | DiffusionActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_infer_micro_bsz = self.config.ref.ppo_micro_batch_size_per_gpu
            if ref_infer_micro_bsz is None and is_diffusion:
                # prepare_micro_batches requires a finite micro-batch; align with actor training micro-batch.
                ref_infer_micro_bsz = self.config.actor.ppo_micro_batch_size_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = ref_infer_micro_bsz
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)
            if is_diffusion:
                assert ref_training_config.engine_config.infer_micro_batch_size_per_gpu is not None, (
                    "Diffusion ref infer_batch requires ref.log_prob_micro_batch_size_per_gpu or "
                    "actor.ppo_micro_batch_size_per_gpu (used by prepare_micro_batches)."
                )
            else:
                ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
                ref_training_config.engine_config.infer_max_token_len_per_gpu = (
                    self.config.ref.ppo_max_token_len_per_gpu
                )

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            if is_diffusion:
                # Diffusion models don't use dynamic batching or token packing.
                # Only micro_batch_size_per_gpu configs are needed.
                actor_training_config.engine_config.micro_batch_size_per_gpu = (
                    self.config.actor.ppo_micro_batch_size_per_gpu
                )
                actor_infer_micro_bsz = self.config.rollout.get("log_prob_micro_batch_size_per_gpu", None)
                if actor_infer_micro_bsz is None:
                    actor_infer_micro_bsz = self.config.actor.ppo_micro_batch_size_per_gpu
                actor_training_config.engine_config.infer_micro_batch_size_per_gpu = actor_infer_micro_bsz
                assert actor_training_config.engine_config.infer_micro_batch_size_per_gpu is not None, (
                    "Diffusion infer_batch requires rollout.log_prob_micro_batch_size_per_gpu or "
                    "actor.ppo_micro_batch_size_per_gpu (used by prepare_micro_batches)."
                )
            else:
                actor_use_dynamic_bsz = self.config.actor.get("use_dynamic_bsz", False)
                rollout_log_prob_use_dynamic_bsz = self.config.rollout.get("log_prob_use_dynamic_bsz", False)
                assert actor_use_dynamic_bsz == rollout_log_prob_use_dynamic_bsz

                # assign engine configs
                actor_training_config.engine_config.use_dynamic_bsz = actor_use_dynamic_bsz
                actor_training_config.engine_config.infer_max_token_len_per_gpu = self.config.rollout.get(
                    "log_prob_max_token_len_per_gpu", None
                )
                actor_training_config.engine_config.infer_micro_batch_size_per_gpu = self.config.rollout.get(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.get(
                    "ppo_max_token_len_per_gpu", None
                )
                actor_training_config.engine_config.micro_batch_size_per_gpu = (
                    self.config.actor.ppo_micro_batch_size_per_gpu
                )
                actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

                if actor_use_dynamic_bsz:
                    assert self.config.rollout.get("log_prob_max_token_len_per_gpu") is not None
                    assert self.config.actor.get("ppo_max_token_len_per_gpu") is not None
                else:
                    assert self.config.rollout.get("log_prob_micro_batch_size_per_gpu") is not None
                    assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            elif model_config.get("model_type", "language_model") in (
                "diffusion_model",
                "diffusion_dpo_model",
                "diffusion_nft_model",
            ):
                self.loss_fn = partial(diffusion_loss, config=actor_config)
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="infer_ref_batch")
    @_with_routing_replay_flag(enabled=False)
    def infer_ref_batch(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="infer_actor_batch")
    @_with_routing_replay_flag(enabled=True)
    def infer_actor_batch(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def copy_adapter(self, source: str = "default", target: str = "old"):
        assert "actor" in self.role, "copy_adapter only supports actor role"
        self.actor.copy_adapter(source=source, target=target)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def ema_update_adapter(self, source: str = "default", target: str = "old", decay: float = 0.0):
        assert "actor" in self.role, "ema_update_adapter only supports actor role"
        self.actor.ema_update_adapter(source=source, target=target, decay=decay)

    def _offload_actor_and_empty_cache(self, timings: Optional[dict] = None):
        """Offload actor params to CPU and free cached GPU memory.

        Safe to run from a worker thread (via ``asyncio.to_thread``): FSDP param
        offload moves the local shard without collectives, and any gathered LoRA
        tensors live in separate allocations that are unaffected by moving the
        base param storage to CPU.
        """
        start = time.perf_counter()
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)
        if timings is not None:
            timings["offload_actor_to_cpu"] = time.perf_counter() - start

    def _gather_lora_weights(self, timings: Optional[dict] = None):
        """Gather LoRA adapter params into a CPU dict, without offloading the actor.

        Intended to run in a worker thread (via ``asyncio.to_thread``) so the
        gather overlaps with resuming rollout weight memory, instead of blocking
        the event loop. ``collect_lora_params`` materializes the LoRA tensors on
        CPU (independent allocations), so the subsequent actor offload can run
        concurrently with the rollout-side sync without affecting these tensors.
        """
        gather_start = time.perf_counter()
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon,
            base_sync_done=True,
            adapter_name=self.config.rollout.rollout_adapter,
        )
        lora_weights = {name: tensor for name, tensor in per_tensor_param}
        if timings is not None:
            timings["get_per_tensor_param"] = time.perf_counter() - gather_start
        return lora_weights, peft_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout (``mode="naive"``),
           update rollout directly from the model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout (any non-naive
           ``mode``), send weights only through the checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for
                logging/tracking.
            mode: Weight update strategy. ``"auto"`` resolves from
                ``config.rollout.checkpoint_engine.backend``; ``"naive"`` uses direct
                colocated sync; any other value delegates to
                ``checkpoint_engine.send_weights``.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param(
                adapter_name=self.config.rollout.rollout_adapter
            )
            await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        # Per-component wall-clock timings (seconds) for monitoring.
        timings: dict[str, float] = {}
        update_weights_start = time.perf_counter()

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout weight memory (released during sleep). This targets the
        #    rollout process and is independent of the actor-side param gather below,
        #    so launch it concurrently and await it before pushing weights.
        resume_weights_task = None
        if self.config.rollout.free_cache_engine:
            resume_weights_task = asyncio.create_task(
                _timed_await("resume_weights", timings, self.rollout.resume(tags=["weights"]))
            )

        # 2. Detect the actor's adapter setup *without* triggering the heavy param
        #    gather (which runs collectives), so the right path can be chosen up
        #    front. ``actor_has_lora`` is a cheap attribute check.
        actor_module = getattr(self.actor.engine, "module", None)
        peft_module = getattr(actor_module, "_fsdp_wrapped_module", actor_module)
        actor_has_lora = peft_module is not None and hasattr(peft_module, "peft_config")
        # Steady-state LoRA (base already synced) can overlap the *entire* gather +
        # actor offload with resume; the first base sync still needs the slow path.
        use_lora_fast_path = actor_has_lora and not self.peft_merge and self.base_sync_done

        # 3. sync weights.
        offloaded = False
        offload_task = None
        if use_lora_fast_path:
            # LoRA-only fast path. Three independent stages overlap:
            #   (a) gather the LoRA adapter into a CPU dict in a worker thread,
            #       overlapping with resuming rollout weight memory;
            #   (b) push the adapter to the rollout (the long pole), awaited here
            #       so ``update_weights_sync`` measures the sync alone;
            #   (c) offload the actor base to CPU in a background worker thread,
            #       launched before the sync so it overlaps it, but only awaited
            #       later (just before kv_cache resume, which needs the freed
            #       memory). The gathered LoRA tensors are independent CPU
            #       allocations, so moving the base param storage to CPU cannot
            #       corrupt the in-flight sync.
            self.rollout.sleep_level = 1
            gather_task = asyncio.create_task(asyncio.to_thread(self._gather_lora_weights, timings))
            if resume_weights_task is not None:
                await resume_weights_task
            log_gpu_memory_usage("After resume weights", logger=logger)
            lora_weights, peft_config = await gather_task
            # Launch the actor offload in the background so it overlaps the sync.
            offload_task = asyncio.create_task(asyncio.to_thread(self._offload_actor_and_empty_cache, timings))

            # Use ZMQ IPC to transfer LoRA weights, bypassing Ray serialization.
            # The _execute_method call only carries a small metadata dict (peft_config,
            # base_sync_done, use_shm) — tensor data goes through the ZMQ socket.
            sync_start = time.perf_counter()
            future = await self.rollout._execute_method(
                "update_weights_from_ipc",
                non_block=True,
                kwargs={"peft_config": peft_config, "base_sync_done": True, "use_shm": self.rollout.use_shm},
            )
            bucket_size_mb = self.config.rollout.checkpoint_engine.update_weights_bucket_megabytes
            sender = BucketedWeightSender(
                zmq_handle=self.rollout.zmq_handle,
                bucket_size_mb=bucket_size_mb,
                use_shm=self.rollout.use_shm,
            )
            await sender.async_send_weights(lora_weights.items())
            if future is not None:
                await future
            timings["update_weights_sync"] = time.perf_counter() - sync_start
            offloaded = True
        else:
            # Normal path: resume rollout weight memory, gather actor params and
            # sync via the standard bucketed-IPC pipeline.
            if resume_weights_task is not None:
                await resume_weights_task
            log_gpu_memory_usage("After resume weights", logger=logger)

            # determine if we need a base weight sync (adapter path only)
            per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon,
                base_sync_done=True,
                adapter_name=self.config.rollout.rollout_adapter,
            )

            do_lora_base_sync = False
            if not self.peft_merge and peft_config is not None:
                self.rollout.sleep_level = 1
                do_lora_base_sync = not self.base_sync_done

            # sync weights: For SGLang, we need base first (when needed), then adapter/merged
            if do_lora_base_sync:
                per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                    layered_summon=self.layered_summon,
                    base_sync_done=False,
                    adapter_name=self.config.rollout.rollout_adapter,
                )
                await self.rollout.update_weights(
                    per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
                )

            await self.rollout.update_weights(
                per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
            )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 4. offload model to cpu. In the LoRA-only fast path this was launched in
        #    the background to overlap the sync; await it here (before kv_cache
        #    resume, which needs the freed GPU memory). Otherwise offload inline.
        if offload_task is not None:
            await offload_task
        elif not offloaded:
            self._offload_actor_and_empty_cache(timings)

        # 5. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await _timed_await("resume_kv_cache", timings, self.rollout.resume(tags=["kv_cache"]))
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

        timings["update_weights_total"] = time.perf_counter() - update_weights_start
        logger.debug(
            "update_weights timing (ms): %s",
            {k: round(v * 1000, 2) for k, v in timings.items()},
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
