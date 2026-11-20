# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Omni Megatron engine using pipeline-selected BSHD model forwards."""

from functools import partial

import torch
import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id
from verl.utils.megatron_utils import (
    get_megatron_module_device,
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
)
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.workers.utils.parameter_revision import ParameterTrackingMixin
from verl_omni.workers.utils.training_output import detach_microbatch_outputs


@EngineRegistry.register(model_type="omni_model", backend="megatron")
class OmniMegatronEngine(ParameterTrackingMixin, MegatronEngineWithLMHead):
    """Use verl's LM flow with the registered pipeline's Megatron model call."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_adapter_cls = OmniModelBase.get_class(model_config)
        model_config = self.model_adapter_cls.prepare_megatron_config(model_config, engine_config)
        self.model_forward = self.model_adapter_cls.get_megatron_forward()
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ) -> None:
        """Preserve native checkpoint ordering while tracking device moves."""
        tracker = getattr(self, "_parameter_revision", None)
        if tracker is None:
            return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep, **kwargs)
        # The native checkpoint entry calls free movement helpers, bypassing
        # the tracked to() method. Storage errors alone do not poison weights.
        with tracker.access():
            origin_module_device = get_megatron_module_device(self.module)
            if self._is_offload_param or origin_module_device == "cpu":
                with tracker.device_transition():
                    load_megatron_model_to_gpu(self.module, load_grad=True)
            self.checkpoint_mananager.save_checkpoint(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )
            torch.distributed.barrier()
            if self._is_offload_param:
                with tracker.device_transition():
                    offload_megatron_model_to_cpu(self.module)

    def supports_detached_training_output(self):
        """Qualify the tested engine and its supported training configuration.

        A subclass must explicitly opt in after qualifying its own output
        semantics; inheriting this implementation alone is not sufficient.
        """
        return (
            type(self) is OmniMegatronEngine
            and not self.engine_config.forward_only
            and not self.engine_config.dynamic_context_parallel
        )

    def postprocess_micro_batch_func(self, output, data, forward_only, loss_function, local_cp_size=None):
        result = super().postprocess_micro_batch_func(
            output, data, forward_only, loss_function, local_cp_size=local_cp_size
        )
        if not forward_only and tu.get_non_tensor_data(data, "return_model_output", default=False):
            return detach_microbatch_outputs(result)
        return result

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        """Use verl's non-fused LM flow, selecting Omni's direct BSHD model call."""
        batch = next(batch_iter).to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(
            batch, key="use_fused_kernels", default=self.engine_config.use_fused_kernels
        )
        if use_fused_kernels:
            raise ValueError("Omni Megatron BSHD forward does not support per-batch fused kernels.")
        if tu.get_non_tensor_data(batch, key="local_cp_size", default=None) is not None:
            raise ValueError("Omni Megatron BSHD forward does not support dynamic CP.")

        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(batch, key="calculate_sum_pi_squared", default=False)
        distillation_use_topk = tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)
        distillation_only = tu.get_non_tensor_data(batch, key="distillation_only", default=False)
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        if pad_mode != DatasetPadMode.NO_PADDING:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        loss_mask = model_inputs["loss_mask"]
        temperature = batch["temperature"]
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)
        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]
        temperature = verl_F.expand_as_nested(temperature, input_ids)

        logits_processor = partial(
            self._lm_head_logits_processor,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            distillation_only=distillation_only,
            logits_processor_func=logits_processor_func,
            batch=batch,
            data_format="bshd",
        )
        response_attention_mask = None
        if attention_mask is not None and not loss_mask.is_nested:
            response_attention_mask = attention_mask[:, -loss_mask.shape[-1] :]
        output = self.model_forward(
            model,
            input_ids,
            model_inputs["multi_modal_inputs"],
            logits_processor=logits_processor,
            logits_processor_args={
                "label": input_ids.clone(),
                "temperature": temperature,
                "loss_mask": loss_mask,
                "response_attention_mask": response_attention_mask,
            },
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            forced_max_seqlen=tu.get_non_tensor_data(batch, key="forced_max_seqlen", default=None),
        )
        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=None)
