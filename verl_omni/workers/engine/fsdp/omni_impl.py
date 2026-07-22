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
"""FSDP/FSDP2 engine for omni models (online RL and direct preference).

Model loading follows PR #258: ``AutoModelForMultimodalLM`` plus
``OmniModelBase.configure_model``.  Direct preference adds token log-prob
``infer_batch`` / ``train_batch`` on top of the shared FSDP stack.
"""

import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    get_init_weight_context_manager,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.model import convert_weight_keys
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine.utils import prepare_micro_batches

from verl_omni.pipelines.utils import prepare_omni_model_inputs
from verl_omni.utils.fsdp_utils import collect_lora_params
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_NON_MODEL_KEYS = {
    "average_log_prob",
    "compute_loss",
    "disable_auto_offload",
    "global_token_num",
    "gradient_accumulation_steps",
    "max_token_len_per_gpu",
    "micro_batch_size_per_gpu",
    "mini_batch_size",
    "num_mini_batch",
    "reference_chosen_logps",
    "reference_rejected_logps",
    "sample_level_rewards",
    "sample_level_scores",
    "sp_size",
    "update_lr_scheduler",
    "use_dynamic_bsz",
    "use_fused_kernels",
    "use_remove_padding",
}

# Text tensors rebuilt by the packed->dense unpack step, and the multimodal
# placeholder masks consumed there. These are excluded from the dict handed to
# the HF forward and replaced with the per-branch (2N-row) tensors.
_UNPACK_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "labels",
    "image_mask",
    "video_mask",
    "audio_mask",
}


@EngineRegistry.register(model_type="omni", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(LoRAAdapterMixin, FSDPEngine):
    """FSDP/FSDP2 omni model engine for direct preference and LoRA training."""

    def __init__(
        self,
        model_config: OmniModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        self._is_lora = self.model_config.lora_rank > 0 or self.model_config.lora_adapter_path is not None
        self._placeholder_token_ids: Optional[dict[str, Optional[int]]] = None

    def _build_module(self):
        from transformers import AutoModelForMultimodalLM

        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_config: OmniModelConfig
        architecture = self.model_config.architecture

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Umbrella config delegates tie_word_embeddings to sub-configs.
        if not hasattr(self.model_config.hf_config, "tie_word_embeddings"):
            self.model_config.hf_config.tie_word_embeddings = False

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            module = AutoModelForMultimodalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            adapter_cls = OmniModelBase.get_class_by_name(
                architecture,
                self.model_config.model_stage,
                self.model_config.get("external_lib"),
            )
            module = adapter_cls.configure_model(module, self.model_config)

            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_fsdp_module(self, module):
        saved_lora_rank = self.model_config.lora_rank
        if self._is_lora and saved_lora_rank <= 0:
            self.model_config.lora_rank = 1
        try:
            return super()._build_fsdp_module(module)
        finally:
            self.model_config.lora_rank = saved_lora_rank

    def _model_module(self):
        return getattr(self.module, "_fsdp_wrapped_module", self.module)

    @contextmanager
    def disable_adapter(self):
        module = self._model_module()
        if not hasattr(module, "disable_adapters"):
            yield
            return
        module.disable_adapters()
        try:
            yield
        finally:
            module.enable_adapters()

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def lr_scheduler_step(self):
        if self.lr_scheduler is None:
            return None
        return super().lr_scheduler_step()

    def _get_placeholder_token_ids(self) -> dict[str, Optional[int]]:
        """Multimodal placeholder token ids, matching the dataset transform.

        The dataset zeroes the multimodal placeholder ids in
        ``input_ids`` and tracks their positions in ``image_mask``/``video_mask``/
        ``audio_mask``. The HF forward instead locates placeholders via
        ``input_ids == config.<x>_token_id``, so we restore the real ids here.
        """
        if self._placeholder_token_ids is None:
            processor = self.model_config.get_processor()
            tokenizer = getattr(processor, "tokenizer", processor)
            vocab = tokenizer.get_vocab()
            self._placeholder_token_ids = {
                "image": vocab.get("<|image_pad|>", vocab.get("<|IMAGE|>")),
                "video": vocab.get("<|video_pad|>", vocab.get("<|VIDEO|>")),
                "audio": vocab.get("<|audio_pad|>", vocab.get("<|AUDIO|>")),
            }
        return self._placeholder_token_ids

    def _unpack_paired_rows(
        self, micro_batch: TensorDict | dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split packed ``[chosen; rejected]`` rows into a dense ``2N``-row batch.

        The dataset packs each preference pair into a single row (chosen then
        rejected along the sequence dim) for the VeOmni varlen forward. The HF
        forward used here has no varlen isolation, so we split each row into two
        independent branches (detected via per-branch ``position_ids`` resets),
        restore the multimodal placeholder ids, and re-pad into ``[chosen0,
        rejected0, chosen1, rejected1, ...]`` order. That order matches the
        multimodal feature order produced by the adapter, so ``masked_scatter``
        in the HF forward consumes features correctly without manual routing.
        """
        input_ids = micro_batch["input_ids"].clone()
        attention_mask = micro_batch["attention_mask"]
        labels = micro_batch["labels"]
        position_ids = micro_batch["position_ids"]

        token_ids = self._get_placeholder_token_ids()
        for name, mask_key in (("image", "image_mask"), ("video", "video_mask"), ("audio", "audio_mask")):
            mask = micro_batch.get(mask_key, None)
            token_id = token_ids.get(name)
            if mask is not None and token_id is not None:
                input_ids[mask.bool()] = token_id

        has_mrope = position_ids.dim() == 3  # (B, 3, L)
        pos_reset_signal = position_ids.sum(dim=1) if has_mrope else position_ids  # (B, L)

        seg_input_ids: list[torch.Tensor] = []
        seg_labels: list[torch.Tensor] = []
        seg_attn: list[torch.Tensor] = []
        seg_pos: list[torch.Tensor] = []
        for b in range(input_ids.shape[0]):
            valid = attention_mask[b].bool()
            valid_len = int(valid.sum().item())
            starts = torch.nonzero((pos_reset_signal[b] == 0) & valid, as_tuple=False).flatten().tolist()
            if len(starts) != 2:
                raise ValueError(
                    "OmniFSDPEngine expects exactly 2 packed sub-sequences (chosen, rejected) per row, "
                    f"but found {len(starts)} position-id resets. Check the paired-preference dataset."
                )
            for start, end in ((starts[0], starts[1]), (starts[1], valid_len)):
                seg_input_ids.append(input_ids[b, start:end])
                seg_labels.append(labels[b, start:end])
                seg_attn.append(attention_mask[b, start:end])
                seg_pos.append(position_ids[b, :, start:end] if has_mrope else position_ids[b, start:end])

        max_len = max(int(seg.shape[-1]) for seg in seg_input_ids)

        def _pad_1d(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
            if tensor.shape[-1] == max_len:
                return tensor
            out = tensor.new_full((max_len,), pad_value)
            out[: tensor.shape[-1]] = tensor
            return out

        def _pad_pos(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[-1] == max_len:
                return tensor
            shape = (*tensor.shape[:-1], max_len)
            out = tensor.new_zeros(shape)
            out[..., : tensor.shape[-1]] = tensor
            return out

        new_input_ids = torch.stack([_pad_1d(seg, 0) for seg in seg_input_ids], dim=0)
        new_labels = torch.stack([_pad_1d(seg, -100) for seg in seg_labels], dim=0)
        new_attn = torch.stack([_pad_1d(seg, 0) for seg in seg_attn], dim=0)
        # HF mrope rotary expects position_ids as (3, batch, seq); the dataset
        # collates them as (batch, 3, seq), so stack the per-branch (3, seq) segments
        # along the new batch axis (dim=1) to yield (3, 2N, seq).
        padded_pos = [_pad_pos(seg) for seg in seg_pos]
        new_pos = torch.stack(padded_pos, dim=1) if has_mrope else torch.stack(padded_pos, dim=0)
        return new_input_ids, new_attn, new_pos, new_labels

    def _prepare_model_inputs(self, micro_batch: TensorDict | dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
        new_input_ids, new_attention_mask, new_position_ids, labels = self._unpack_paired_rows(micro_batch)
        branch_items = {
            key: value for key, value in micro_batch.items() if key not in _NON_MODEL_KEYS and key not in _UNPACK_KEYS
        }
        branch_items["input_ids"] = new_input_ids
        branch_items["attention_mask"] = new_attention_mask
        branch_items["position_ids"] = new_position_ids
        branch_items["labels"] = labels
        branch_batch = TensorDict.from_dict(branch_items, batch_size=new_input_ids.shape[0])
        model_inputs = prepare_omni_model_inputs(
            self.model_config,
            branch_batch,
            dtype=next(self.module.parameters()).dtype,
        )
        return model_inputs, labels

    @staticmethod
    def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:].contiguous()
        loss_mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~loss_mask, 0)
        token_logps = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        seq_logps = (token_logps * loss_mask).sum(dim=-1)
        if average_log_prob:
            seq_logps = seq_logps / loss_mask.sum(dim=-1).clamp(min=1)
        return seq_logps

    def _concatenated_forward(self, model, micro_batch: TensorDict | dict[str, Any]):
        model_inputs, labels = self._prepare_model_inputs(micro_batch)
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        all_logps = self._sequence_logps(
            logits,
            labels,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )
        return all_logps[0::2], all_logps[1::2]

    def _forward_backward_micro_batch(self, micro_batch: TensorDict, loss_function: Callable):
        micro_batch = micro_batch.to(get_device_id())
        tu.assign_non_tensor(
            micro_batch,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )
        policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.module, micro_batch)
        model_output = {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": micro_batch["reference_chosen_logps"],
            "reference_rejected_logps": micro_batch["reference_rejected_logps"],
        }
        loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        loss.backward()
        return loss.detach(), metrics

    def train_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        if loss_function is None:
            raise ValueError("OmniFSDPEngine.train_batch requires a loss_function.")
        config = getattr(loss_function, "keywords", {}).get("config")
        loss_config = getattr(config, "omni_loss", None)
        tu.assign_non_tensor(data, average_log_prob=getattr(loss_config, "average_log_prob", False))
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        gradient_accumulation_steps = len(micro_batches)
        losses = []
        metrics: dict[str, list[torch.Tensor]] = {}
        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
            loss, micro_metrics = self._forward_backward_micro_batch(micro_batch, loss_function)
            losses.append(loss.item())
            for key, value in micro_metrics.items():
                metrics.setdefault(key, []).append(value)
        grad_norm = self.optimizer_step()
        self.optimizer_zero_grad()
        metrics = {key: torch.stack(value).mean().item() for key, value in metrics.items()}
        metrics["grad_norm"] = grad_norm
        return {"model_output": {}, "loss": losses, "metrics": metrics}

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        del loss_function
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        chosen_logps = []
        rejected_logps = []
        with torch.no_grad():
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                chosen, rejected = self._concatenated_forward(self.module, micro_batch)
                chosen_logps.append(chosen)
                rejected_logps.append(rejected)
        return {
            "model_output": {
                "chosen_logps": torch.cat(chosen_logps, dim=0),
                "rejected_logps": torch.cat(rejected_logps, dim=0),
            },
            "loss": [0.0],
            "metrics": {},
        }

    def get_per_tensor_param(
        self,
        layered_summon=False,
        base_sync_done=False,
        adapter_name: str | None = None,
        **kwargs,
    ):
        load_fsdp_model_to_gpu(self.module)
        peft_config = None
        peft_model = self._model_module()
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            adapter_ctx = self.use_adapter(adapter_name) if adapter_name is not None else nullcontext()
            with adapter_ctx:
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                    is_diffusers=False,
                    adapter_name=adapter_name or "default",
                    layer_prefixes=self.model_config.fsdp_layer_prefixes,
                )
        else:
            params = self.module.state_dict()
        params = convert_weight_keys(params, peft_model)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        device = get_device_id()
        export_dtype = PrecisionType.to_dtype(self.engine_config.model_dtype)

        def param_generator():
            for name, param in params.items():
                tensor = param.full_tensor() if isinstance(param, DTensor) else param
                tensor = tensor.to(device, non_blocking=True)
                if tensor.is_floating_point() and export_dtype is not None and tensor.dtype != export_dtype:
                    tensor = tensor.to(export_dtype, non_blocking=True)
                yield name, tensor

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return param_generator(), peft_config_dict
