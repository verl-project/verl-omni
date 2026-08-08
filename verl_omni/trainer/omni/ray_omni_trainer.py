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
"""Omni trainer — a PPOTrainerSync subclass registered via ``@register_trainer("omni_sync")``."""

from typing import Any

import numpy as np
import torch
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.workers.config import OmniModelConfig


@register_trainer("omni_sync")
class OmniPPOTrainerSync(PPOTrainerSync):
    """``PPOTrainerSync`` subclass that wires tokenizer/processor from ``OmniModelConfig``."""

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _compute_reward_colocate(self, batch, metrics: dict[str, Any] | None = None):
        """Run a generative reward model on the actor's colocated resource pool.

        The verl revision pinned by this repository leaves the V1 bridge
        unimplemented. Besides backporting that bridge, retain the dataset
        metadata required by multimodal rewards such as OmniVideo-R1 QI.
        """
        del metrics

        import transfer_queue as tq
        from tensordict import TensorDict
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu

        fields = ["prompts", "responses", "raw_prompt", "data_source", "reward_model", "extra_info"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        padding_id = self.tokenizer.pad_token_id or 0
        prompts = data["prompts"].to_padded_tensor(padding=padding_id)
        responses = data["responses"].to_padded_tensor(padding=padding_id)
        attention_mask = torch.cat(
            [
                self._lengths_to_mask(prompt_lengths, prompts.size(1)),
                self._lengths_to_mask(response_lengths, responses.size(1)),
            ],
            dim=1,
        )

        non_tensor_batch = {
            field: self._as_object_array(data[field])
            for field in ("raw_prompt", "data_source", "reward_model", "extra_info")
        }
        rm_input = DataProto(
            batch=TensorDict(
                {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
                batch_size=len(batch),
            ),
            non_tensor_batch=non_tensor_batch,
        )
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[index, : response_lengths[index]] for index in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(write_back),
        )
        return batch

    @staticmethod
    def _as_object_array(values) -> np.ndarray:
        """Preserve nested prompts and dictionaries as one object per sample."""
        values = list(values)
        result = np.empty(len(values), dtype=object)
        result[:] = values
        return result

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Build a right-padded attention mask from sequence lengths."""
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).to(torch.int64)
