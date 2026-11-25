# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Omni Megatron engine using pipeline-selected BSHD model forwards."""

from functools import partial

import torch
import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.workers.engine.megatron.router_replay import build_bshd_router_replay_targets


@EngineRegistry.register(model_type="omni_model", backend="megatron")
class OmniMegatronEngine(MegatronEngineWithLMHead):
    """Use verl's LM flow with the registered pipeline's Megatron model call."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_adapter_cls = OmniModelBase.get_class(model_config)
        model_config = self.model_adapter_cls.prepare_megatron_config(model_config, engine_config)
        self.model_forward = self.model_adapter_cls.get_megatron_forward()
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

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
        unwrapped_model = unwrap_model(model)
        router_replay_active = self._prepare_router_replay(unwrapped_model, batch, model_inputs)
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
        self._finish_router_replay(unwrapped_model, router_replay_active)
        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=None)

    def _prepare_router_replay(self, model, batch, model_inputs) -> bool:
        """Point the model's routers at this micro-batch's rollout routes.

        Returns whether the forward runs with replayed routes.
        """
        if not self.enable_routing_replay:
            return False
        import megatron.core.parallel_state as mpu
        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
        from verl.utils.megatron.router_replay_patch import RouterReplayAction
        from verl.utils.megatron.router_replay_utils import (
            RouterReplayHelper,
            is_moe_layer,
            iter_model_routers,
            set_model_router_replay_action,
        )

        vp_rank = getattr(model, "vp_stage", 0)
        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            # The previous micro-batch's backward recompute ran; back to forward replay.
            for router in RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank):
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            set_model_router_replay_action(model, RouterReplayAction.REPLAY_FORWARD)
        if not RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            return False
        if self.engine_config.router_replay.mode != "R3":
            raise NotImplementedError("Omni Megatron router replay supports mode='R3' only.")

        targets, replay_mask = build_bshd_router_replay_targets(
            model_inputs["routed_experts"],
            model_inputs["input_ids"],
            model_inputs["loss_mask"],
            self._bshd_padded_len(model, batch, model_inputs["input_ids"]),
        )
        if mpu.get_tensor_model_parallel_world_size() > 1 and getattr(self.tf_config, "sequence_parallel", False):
            targets = scatter_to_sequence_parallel_region(targets).contiguous()
            replay_mask = scatter_to_sequence_parallel_region(replay_mask).contiguous()

        index_by_layer = targets.shape[1] == self.tf_config.num_layers
        for layer_number, router in iter_model_routers(model):
            idx = layer_number - 1
            if not index_by_layer:
                idx = sum(1 for i in range(idx) if is_moe_layer(self.tf_config, i))
            if 0 <= idx < targets.shape[1]:
                router.set_target_indices(targets[:, idx].contiguous().to(torch.int64), replay_mask=replay_mask)
        return True

    def _finish_router_replay(self, model, active: bool) -> None:
        """Switch to replay-backward so activation recompute pops its own routes."""
        if not active:
            return
        from verl.utils.megatron.router_replay_patch import RouterReplayAction
        from verl.utils.megatron.router_replay_utils import RouterReplayHelper, set_model_router_replay_action

        vp_rank = getattr(model, "vp_stage", 0)
        for router in RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank):
            router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        set_model_router_replay_action(model, RouterReplayAction.REPLAY_BACKWARD)

    def _bshd_padded_len(self, model, batch, input_ids) -> int:
        """Sequence length the adapter's BSHD forward pads ``input_ids`` to."""
        from verl.models.mcore.util import build_vlm_attn_mask_bshd, preprocess_bshd_engine

        forced_max_seqlen = tu.get_non_tensor_data(batch, key="forced_max_seqlen", default=None)
        if hasattr(self.model_config.hf_config, "vision_config"):
            _, attention_mask = build_vlm_attn_mask_bshd(
                input_ids,
                input_ids.shape[0],
                self.model_config.tokenizer.pad_token_id,
                forced_max_seqlen=forced_max_seqlen,
            )
        else:
            _, attention_mask, _ = preprocess_bshd_engine(
                input_ids,
                pre_process=model.pre_process,
                use_fp8_padding=model.config.fp8 in ("e4m3", "hybrid"),
                forced_max_seqlen=forced_max_seqlen,
            )
        return attention_mask.shape[-1]
