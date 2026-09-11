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
"""DMD2 sampling and optimization on the existing Diffusers FSDP lifecycle."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer
from verl.utils.metric import Metric
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.utils import prepare_micro_batches

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.distillation.utils import ode_euler_step, standard_cfg, timestep_shift

from .diffusers_impl import DiffusersFSDPEngine


@EngineRegistry.register(model_type="diffusion_dmd_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class DMDDiffusersFSDPEngine(DiffusersFSDPEngine):
    """One student/fake-score optimizer pair with frozen scoring and adapter EMA.

    The validated storage path is named LoRA adapters on one frozen base. Model
    conversion and conditioning are adapter-owned; no image layouts enter here.
    """

    adapter_names = {
        "student": "default",
        "fake_score": "fake_score",
        "teacher_score": "reference",
        "student_ema": "student_ema",
    }
    stream_offsets = {"initial_noise": 0, "rollout_decision": 1, "score_sigma": 3, "score_noise": 4}

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, *, dmd_config):
        if model_config.lora_rank <= 0:
            raise ValueError("The DMD2 MVP requires LoRA; full-module training is not enabled.")
        if engine_config.strategy == "fsdp" and not engine_config.use_orig_params:
            raise ValueError("DMD2 shared adapters require FSDP1 use_orig_params=true.")
        if engine_config.forward_only:
            raise ValueError("A DMD2 engine must own trainable student and fake-score optimizers.")
        self.dmd_config = dmd_config
        self.active_stage = "student"
        self.optimizers = {}
        self.schedulers = {}
        self.role_parameters = {}
        self.optimizer_steps = {"student": 0, "fake_score": 0}
        self.skipped_steps = {"student": 0, "fake_score": 0}
        self.generators = {}
        self.pending_generator_states = {}
        self.last_step_succeeded = False
        self.forward_finite = True
        model_config = deepcopy(model_config)
        allowed = {"default", "fake_score", "student_ema", "reference"}
        if not set(model_config.policy_state_adapters).issubset(allowed):
            raise ValueError(
                "DMD2 manages default, fake_score and student_ema adapters; other policy states are unsupported."
            )
        object.__setattr__(model_config, "policy_state_adapters", ("default", "fake_score", "student_ema"))
        self.model_adapter = DiffusionModelBase.get_class(model_config)
        for method in (
            "build_conditioning_provider",
            "latent_geometry",
            "pack_latents",
            "prepare_dmd_inputs",
            "prediction_to_x0",
            "sampling_sigmas",
        ):
            if not callable(getattr(self.model_adapter, method, None)):
                raise TypeError(f"The selected DMD2 model adapter must implement {method}.")
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def initialize(self):
        """Reuse model loading and checkpoint management with reproducible LoRA initialization."""
        with torch.random.fork_rng(devices=[get_device_id()], device_type=get_device_name()):
            torch.manual_seed(self.engine_config.seed)
            super().initialize()
        self.copy_adapter("default", "fake_score")
        self.copy_adapter("default", "student_ema")
        self.select_stage("student")
        self.condition_provider = self.model_adapter.build_conditioning_provider(self.model_config, self.dmd_config)

    def _build_optimizer(self, module):
        return build_optimizer(
            (parameter for parameter in module.parameters() if parameter.requires_grad), self.optimizer_config
        )

    def _build_model_optimizer(self):
        super()._build_model_optimizer()
        self.optimizers["student"] = self.optimizer
        self.schedulers["student"] = self.lr_scheduler
        self.optimizer_configs = {
            "student": self.optimizer_config,
            "fake_score": deepcopy(self.dmd_config.fake_score_optim),
        }
        self.optimizer_configs["fake_score"].total_training_steps = (
            self.optimizer_config.total_training_steps * self.dmd_config.fake_update_ratio
        )
        for stage in ("student", "fake_score"):
            with self.use_adapter(self.adapter_names[stage]):
                self.role_parameters[stage] = tuple(
                    parameter for parameter in self.module.parameters() if parameter.requires_grad
                )
        if not all(self.role_parameters.values()):
            raise ValueError("Every DMD2 optimizer must own parameters.")
        if {id(p) for p in self.role_parameters["student"]} & {id(p) for p in self.role_parameters["fake_score"]}:
            raise ValueError("Student and fake-score optimizers must not share parameters.")
        self.optimizers["fake_score"] = build_optimizer(
            self.role_parameters["fake_score"], self.optimizer_configs["fake_score"]
        )
        previous = self.optimizer_config
        try:
            self.optimizer_config = self.optimizer_configs["fake_score"]
            self.schedulers["fake_score"] = self._build_lr_scheduler(self.optimizers["fake_score"])
        finally:
            self.optimizer_config = previous

    def select_stage(self, stage):
        """Select optimizer ownership before entering the ordinary engine train context."""
        if stage not in {"student", "fake_score"}:
            raise ValueError(f"Invalid DMD2 update stage {stage!r}.")
        self.active_stage = stage
        self._set_adapter(self.adapter_names[stage])
        self.optimizer = self.optimizers[stage]
        self.lr_scheduler = self.schedulers[stage]
        self.optimizer_config = self.optimizer_configs[stage]

    def to(self, device, model=True, optimizer=True, grad=True):
        """Reuse base model offload and move both optimizer states when requested."""
        super().to(device=device, model=model, optimizer=False, grad=grad)
        if optimizer:
            for item in self.optimizers.values():
                if device == "cpu":
                    offload_fsdp_optimizer(item)
                else:
                    load_fsdp_optimizer(item, device)

    def generator(self, stream, device):
        """Use independent, checkpointed streams with the prototype's DP seed spacing."""
        if stream not in self.generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(
                self.engine_config.seed + self.get_data_parallel_rank() * 5 + self.stream_offsets[stream]
            )
            if stream in self.pending_generator_states:
                generator.set_state(self.pending_generator_states.pop(stream))
            self.generators[stream] = generator
        return self.generators[stream]

    def noise(self, shape, device, stream):
        """Draw fp32 noise without sharing state with logging or data shuffling."""
        return torch.randn(shape, dtype=torch.float32, device=device, generator=self.generator(stream, device))

    @staticmethod
    def expand_sigma(sigma, latents):
        """Broadcast explicit per-sample sigma over the architecture's latent axes."""
        sigma = sigma.reshape(-1)
        if sigma.numel() not in (1, latents.shape[0]):
            raise ValueError("Sigma batch size does not match the latent batch.")
        return sigma.reshape(-1, *((1,) * (latents.ndim - 1)))

    def predict(self, role, latents, sigma, condition, geometry, *, grad_enabled=False):
        """Run one role with adapter restoration, without nested engine train/eval contexts."""
        grad_context = nullcontext() if grad_enabled else torch.no_grad()
        with self.use_adapter(self.adapter_names[role]), grad_context, torch.profiler.record_function(f"dmd/{role}"):
            self.module.eval()
            inputs = self.model_adapter.prepare_dmd_inputs(
                self.module, self.model_config, latents, sigma, condition, geometry
            )
            prediction = self.model_adapter.forward(self.module, self.model_config, inputs)
            if prediction.shape != latents.shape:
                raise ValueError("DMD2 model prediction must preserve the declared latent shape.")
            return prediction

    def student_sample(self, noise, condition, geometry, *, grad_enabled):
        """Backward-simulate to a rank-synchronized exit, retaining only its graph."""
        steps = self.model_config.pipeline.num_inference_steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer.")
        sigmas = self.model_adapter.sampling_sigmas(self.model_config, self.dmd_config, noise.device)
        exit_step = torch.randint(
            steps, (1,), device=noise.device, generator=self.generator("rollout_decision", noise.device)
        )
        torch.distributed.broadcast(exit_step, src=0)
        exit_index = int(exit_step.item())
        sample = noise
        for index in range(exit_index + 1):
            prediction = self.predict(
                "student", sample, sigmas[index], condition, geometry, grad_enabled=grad_enabled and index == exit_index
            )
            if index == exit_index:
                return self.model_adapter.prediction_to_x0(
                    sample, prediction, self.expand_sigma(sigmas[index], sample)
                ), exit_index
            sample = ode_euler_step(sample, prediction, sigmas[index], sigmas[index + 1])
        raise RuntimeError("Student sampling reached no exit prediction.")

    def score_inputs(self, generated):
        """Construct detached flow corruption with discrete or continuous sigma sampling."""
        cfg = self.dmd_config
        if cfg.score_discrete_steps:
            total = self.scheduler.config.num_train_timesteps
            if cfg.score_discrete_steps != total:
                raise ValueError("score_discrete_steps must equal the model scheduler's num_train_timesteps.")
            timestep = torch.randint(
                total,
                (generated.shape[0],),
                device=generated.device,
                generator=self.generator("score_sigma", generated.device),
            )
            sigma = timestep_shift(timestep, total, cfg.score_timestep_shift) / total
            sigma = sigma.clamp(cfg.score_sigma_min, cfg.score_sigma_max)
        else:
            sigma = torch.rand(
                generated.shape[0], device=generated.device, generator=self.generator("score_sigma", generated.device)
            )
            sigma = cfg.score_sigma_min + (cfg.score_sigma_max - cfg.score_sigma_min) * sigma
        noise = self.noise(generated.shape, generated.device, "score_noise")
        expanded = self.expand_sigma(sigma, generated)
        return (1 - expanded) * generated.detach().float() + expanded * noise, noise, sigma

    def prepare_model_inputs(self, micro_batch, step=None):
        """Delegate geometry and frozen conditioning to the registered architecture."""
        device = torch.device(get_device_id())
        dtype = getattr(self.module, "dtype", torch.bfloat16)
        condition, negative = self.condition_provider.encode(
            micro_batch, device=device, dtype=dtype, require_negative=self.active_stage == "student"
        )
        for item in (condition, negative):
            if item is None:
                continue
            for key, value in item.items():
                item[key] = value.to(device)
            if self.use_ulysses_sp:
                item["prompt_embeds"], item["prompt_embeds_mask"] = self._pad_embeds_for_sp(
                    item["prompt_embeds"], item["prompt_embeds_mask"], self.ulysses_sequence_parallel_size
                )
        shape, geometry = self.model_adapter.latent_geometry(self.module, self.model_config, micro_batch)
        return condition, negative, shape, geometry

    def prepare_model_outputs(self, output, micro_batch):
        """Keep graph-bearing student outputs and detached scoring inputs explicit."""
        return output

    def forward_step(self, micro_batch, loss_function, forward_only=False, step=None):
        """Compute one DMD2 microbatch through the registered diffusion loss."""
        timings = {}
        start = time.perf_counter()
        condition, negative, shape, geometry = self.prepare_model_inputs(micro_batch)
        timings["perf/condition_encode_s"] = time.perf_counter() - start
        initial = self.model_adapter.pack_latents(self.noise(shape, condition["prompt_embeds"].device, "initial_noise"))
        start = time.perf_counter()
        with torch.profiler.record_function("dmd/student_rollout"):
            generated, exit_index = self.student_sample(
                initial, condition, geometry, grad_enabled=self.active_stage == "student" and not forward_only
            )
        timings["perf/student_rollout_s"] = time.perf_counter() - start
        noisy, noise, sigma = self.score_inputs(generated)
        start = time.perf_counter()
        prediction = self.predict(
            "fake_score",
            noisy,
            sigma,
            condition,
            geometry,
            grad_enabled=self.active_stage == "fake_score" and not forward_only,
        )
        timings["perf/fake_score_s"] = time.perf_counter() - start
        if self.active_stage == "student":
            start = time.perf_counter()
            positive = self.predict("teacher_score", noisy, sigma, condition, geometry)
            negative_prediction = self.predict("teacher_score", noisy, sigma, negative, geometry)
            guided = standard_cfg(
                positive, negative_prediction, self.dmd_config.teacher_guidance_scale, self.dmd_config.cfg_norm
            )
            timings["perf/teacher_score_s"] = time.perf_counter() - start
            expanded = self.expand_sigma(sigma, generated)
            output = {
                "generated_x0": generated,
                "fake_x0": self.model_adapter.prediction_to_x0(noisy, prediction, expanded),
                "teacher_x0": self.model_adapter.prediction_to_x0(noisy, guided, expanded),
            }
        else:
            output = {"generated_x0": generated.detach(), "noise_pred": prediction, "noise": noise}
        output = self.prepare_model_outputs(output, micro_batch)
        loss, metrics = loss_function(model_output=output, data=micro_batch, dp_group=self.get_data_parallel_group())
        metrics.update({"dmd/rollout_exit": float(exit_index), **timings})
        return loss, metrics

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        """Use the existing DP-aware microbatch splitter with sample-weighted means."""
        stage = tu.get_non_tensor_data(data, "dmd_stage", default="student")
        if stage != self.active_stage:
            raise ValueError("Select the DMD2 optimizer before entering its train context.")
        self.module.eval()
        tu.assign_non_tensor(data, use_dynamic_bsz=False, sp_size=self.ulysses_sequence_parallel_size)
        micro_size = tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", default=None)
        if len(data) % micro_size:
            # verl's wrapper requires divisibility; native TensorDict splitting preserves the dense tail.
            micro_batches = data.split(micro_size)
        else:
            micro_batches, _ = prepare_micro_batches(
                data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
            )
        losses, metrics = [], {}
        self.forward_finite = True
        for micro in micro_batches:
            micro = micro.to(get_device_id())
            tu.assign_non_tensor(
                micro,
                gradient_accumulation_steps=len(data) / len(micro),
                dmd_normalization_epsilon=self.dmd_config.normalization_epsilon,
            )
            loss, values = self.forward_step(micro, loss_function, forward_only)
            loss_value = loss.detach().item()
            self.forward_finite = self.forward_finite and math.isfinite(loss_value)
            if not forward_only:
                start = time.perf_counter()
                loss.backward()
                values["perf/backward_s"] = time.perf_counter() - start
            losses.append(loss_value)
            for key, value in values.items():
                summed = key.endswith("_s") or key.endswith("/active_elements") or key.endswith("/nonfinite")
                weight = 1.0 if summed else len(micro) / len(data)
                value = value.aggregate() if isinstance(value, Metric) else value
                metrics[key] = metrics.get(key, 0.0) + float(value) * weight
        metrics = Metric.from_dict(metrics, aggregation="mean")
        device = get_torch_device()
        metrics["perf/max_memory_allocated_gib"] = Metric("max", device.max_memory_allocated() / 1024**3)
        metrics["perf/max_memory_reserved_gib"] = Metric("max", device.max_memory_reserved() / 1024**3)
        return {"loss": losses, "metrics": metrics, "model_output": {}}

    def optimizer_step(self):
        """Agree on numerical skips before stepping, and update only the owning scheduler/EMA."""
        for stage, parameters in self.role_parameters.items():
            if stage != self.active_stage and any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError(f"Gradient leaked into inactive DMD2 role {stage}.")
        norm = float(self.clip_grad_norm())
        finite = torch.tensor(int(math.isfinite(norm) and self.forward_finite), device=get_device_id())
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        self.last_step_succeeded = bool(finite.item())
        if self.last_step_succeeded:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer_steps[self.active_stage] += 1
            if self.active_stage == "student" and self.optimizer_steps["student"] >= self.dmd_config.ema_start_step:
                self.ema_update_adapter("default", "student_ema", self.dmd_config.ema_decay)
        else:
            self.skipped_steps[self.active_stage] += 1
        self.optimizer.zero_grad()
        return norm

    def lr_scheduler_step(self):
        """Schedulers already advance atomically with successful optimizer updates."""
        return self.lr_scheduler.get_last_lr()[0]

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        """Save the physical model/student optimizer once, plus DMD-specific state."""
        if hdfs_path is not None:
            raise ValueError("DMD2 atomic checkpoints currently require a local shared directory.")
        previous = self.active_stage
        self.select_stage("student")
        try:
            super().save_checkpoint(local_path, global_step=global_step)
            torch.save(
                {
                    "version": 1,
                    "world_size": torch.distributed.get_world_size(),
                    "fake_optimizer": self.optimizers["fake_score"].state_dict(),
                    "fake_scheduler": self.schedulers["fake_score"].state_dict(),
                    "optimizer_steps": self.optimizer_steps,
                    "skipped_steps": self.skipped_steps,
                    "generators": {
                        **self.pending_generator_states,
                        **{name: generator.get_state() for name, generator in self.generators.items()},
                    },
                },
                Path(local_path) / f"dmd_state_rank_{self.rank}.pt",
            )
            torch.distributed.barrier()
        finally:
            self.select_stage(previous)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        """Reject old prototype/incomplete state, then restore both optimization clocks."""
        if hdfs_path is not None or del_local_after_load:
            raise ValueError("DMD2 resume preserves its local atomic checkpoint.")
        state = torch.load(Path(local_path) / f"dmd_state_rank_{self.rank}.pt", map_location="cpu", weights_only=False)
        if state.get("version") != 1 or state.get("world_size") != torch.distributed.get_world_size():
            raise ValueError("Incompatible DMD2 checkpoint version or world size.")
        if set(state.get("optimizer_steps", {})) != {"student", "fake_score"}:
            raise ValueError("Missing DMD2 optimizer counters.")
        self.select_stage("student")
        super().load_checkpoint(local_path, del_local_after_load=False)
        self.optimizers["fake_score"].load_state_dict(state["fake_optimizer"])
        self.schedulers["fake_score"].load_state_dict(state["fake_scheduler"])
        self.optimizer_steps = state["optimizer_steps"]
        self.skipped_steps = state["skipped_steps"]
        self.generators.clear()
        self.pending_generator_states = state["generators"]
        return dict(self.optimizer_steps)

    def export_student(self, directory, role="student"):
        """Export only the chosen inference adapter using existing per-tensor collection."""
        if role not in {"student", "student_ema"}:
            raise ValueError("Only student or student_ema can be exported.")
        adapter = self.adapter_names[role]
        with self.use_adapter(adapter):
            expected = torch.tensor(
                sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad),
                device=get_device_id(),
            )
        if self.engine_config.strategy == "fsdp":
            torch.distributed.all_reduce(expected)
        tensors, peft_config = self.get_per_tensor_param(base_sync_done=True, adapter_name=adapter)
        state = {name.removeprefix("transformer."): value.detach().cpu().contiguous() for name, value in tensors}
        if sum(value.numel() for value in state.values()) != int(expected.item()):
            raise ValueError("Incomplete LoRA export: fsdp_layer_prefixes must cover every selected adapter parameter.")
        if not state or any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Inference export requires nonempty finite adapter weights.")
        if self.rank == 0:
            from safetensors.torch import save_file

            os.makedirs(directory, exist_ok=False)
            save_file(state, str(Path(directory) / "adapter_model.safetensors"))
            with open(Path(directory) / "adapter_config.json", "w") as file:
                json.dump(peft_config, file, indent=2, default=list)
        torch.distributed.barrier()
