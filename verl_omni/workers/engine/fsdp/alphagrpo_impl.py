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

"""Accumulate one text objective and all image steps into one policy update."""

from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import EngineRegistry

from .diffusers_impl import PPODiffusersFSDPEngine


@EngineRegistry.register(model_type="diffusion_alphagrpo_model", backend=["fsdp", "fsdp2"], device=["cuda"])
class AlphaGRPOFSDPEngine(PPODiffusersFSDPEngine):
    """Reuse diffusion accumulation with a text update at the first denoising step."""

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        """Keep the text loss independent of image timestep count; see BAGEL README Gotchas."""
        if tu.get_non_tensor_data(data, "enable_timestep_staging", default=False):
            raise ValueError("AlphaGRPO does not support timestep staging.")
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError("AlphaGRPO currently requires sequence parallel size 1.")
        return super().forward_backward_batch(data, loss_function, forward_only)

    def forward_step(self, micro_batch, loss_function, forward_only, step):
        """Add exact masked thinking likelihoods only once per micro-batch."""
        if loss_function is None or step != 0:
            return super().forward_step(micro_batch, loss_function, forward_only, step)

        def joint_loss(model_output, data, dp_group):
            text_log_probs = self.module(
                thinking_input_ids=micro_batch["thinking_input_ids"],
                thinking_attention_mask=micro_batch["thinking_attention_mask"],
                thinking_labels=micro_batch["thinking_labels"],
                thinking_mask=micro_batch["thinking_mask"],
                thinking_temperature=self.model_config.pipeline.think_temperature,
            )
            model_output["text_log_probs"] = text_log_probs
            data["old_text_log_probs"] = micro_batch["old_text_log_probs"]
            data["thinking_mask"] = micro_batch["thinking_mask"]
            tu.assign_non_tensor(data, text_loss_scale=micro_batch["all_timesteps"].shape[1])
            return loss_function(model_output=model_output, data=data, dp_group=dp_group)

        return super().forward_step(micro_batch, joint_loss, forward_only, step)
