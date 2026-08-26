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

from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import WorkerGroup
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict
from verl_omni.workers.config.diffusion import DiffusionDistillationConfig, DiffusionDistillationTeacherModelConfig
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding


class DiffusionTeacherManager:
    """Routes rollout batches to frozen diffusion teachers and scores them."""

    def __init__(
        self,
        distillation_config: DiffusionDistillationConfig,
        model_config: DictConfig,
        teacher_wg: dict[str, WorkerGroup],
        infer_micro_batch_size_per_gpu: Optional[int] = None,
    ):
        self.distillation_config = distillation_config
        self.infer_micro_batch_size_per_gpu = infer_micro_batch_size_per_gpu
        self.teacher_key: str = distillation_config.teacher_key
        self.teacher_model_configs: dict[str, DiffusionDistillationTeacherModelConfig] = (
            distillation_config.teacher_models
        )
        expected = set(self.teacher_model_configs)
        if set(teacher_wg.keys()) != expected:
            raise ValueError(
                f"teacher worker group keys {sorted(teacher_wg.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_wg = teacher_wg
        self.model_config = model_config

    def _resolve_teacher_keys(self, batch: DataProto) -> np.ndarray:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return np.full(len(batch), next(iter(self.teacher_model_configs)), dtype=object)
        if self.teacher_key not in batch.non_tensor_batch:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        routing_keys = batch.non_tensor_batch[self.teacher_key]
        unknown = sorted(set(routing_keys) - set(self.teacher_model_configs))
        if unknown:
            raise ValueError(
                f"No teacher configured for routing key {unknown}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_keys

    def _infer(self, batch: DataProto, teacher_key: str):
        batch_td = _to_diffusion_worker_tensordict(batch)
        batch_td = embeds_padding_2_no_padding(batch_td)
        tu.assign_non_tensor(
            batch_td,
            compute_loss=False,
            height=self.model_config.pipeline.height,
            width=self.model_config.pipeline.width,
            vae_scale_factor=self.model_config.get("vae_scale_factor", 8),
            teacher_key=teacher_key,
        )
        return self.teacher_wg[teacher_key].infer_teacher_batch(batch_td)

    @staticmethod
    def _to_dataproto(output) -> DataProto:
        prev_sample_mean = tu.get(output, "prev_sample_mean")
        teacher_output = tu.get_tensordict({"teacher_prev_sample_mean": prev_sample_mean.float()})
        return DataProto.from_tensordict(teacher_output)

    def compute_prev_sample_mean(self, batch: DataProto) -> DataProto:
        """Score ``batch`` with its teachers, returning ``teacher_prev_sample_mean`` in the input row order."""
        routing_keys = self._resolve_teacher_keys(batch)
        if len(self.teacher_model_configs) == 1:
            return self._to_dataproto(self._infer(batch, routing_keys[0]).get())
        # dispatch per teacher from the driver so every DP rank of a teacher sees a non-empty shard
        # that splits evenly into forward micro-batches
        order, pending = [], []
        for teacher_key, wg in self.teacher_wg.items():
            idxs = np.flatnonzero(routing_keys == teacher_key)
            if len(idxs) == 0:
                continue
            size_divisor = wg.world_size * (self.infer_micro_batch_size_per_gpu or 1)
            padded, pad_size = pad_dataproto_to_divisor(batch.select_idxs(idxs), size_divisor)
            pending.append((self._infer(padded, teacher_key), pad_size))
            order.append(idxs)
        outputs = [unpad_dataproto(self._to_dataproto(future.get()), pad_size) for future, pad_size in pending]
        teacher_output = DataProto.concat(outputs)
        teacher_output.reorder(torch.from_numpy(np.argsort(np.concatenate(order))))
        return teacher_output
