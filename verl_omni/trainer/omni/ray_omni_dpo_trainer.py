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

"""Omni direct-preference Ray trainer."""

from __future__ import annotations

import logging
from typing import Optional

from verl.protocol import DataProto
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import rename_dict

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    BaseRayDiffusionTrainer,
    DirectPreferenceRayTrainer,
)

sys_logger = logging.getLogger(__name__)

__all__ = ["OmniDirectPreferenceRayTrainer"]


class OmniDirectPreferenceRayTrainer(DirectPreferenceRayTrainer):
    """Omni AR direct-preference trainer on the shared Ray preference loop.

    Supports ref-in-actor (LoRA base weights as reference) and an optional
    external ref worker when ``lora_rank == 0``.
    """

    def __init__(self, config, *args, **kwargs):
        BaseRayDiffusionTrainer.__init__(self, config, *args, **kwargs)
        self.is_offline = config.algorithm.get("sample_source", "online") == "offline"
        if not self.is_offline:
            raise NotImplementedError(
                "OmniDirectPreferenceRayTrainer currently supports algorithm.sample_source=offline only."
            )
        if config.actor_rollout_ref.model.get("model_type", "language_model") != "omni_model":
            raise ValueError("OmniDirectPreferenceRayTrainer requires actor_rollout_ref.model.model_type=omni_model.")
        loss_mode = config.actor_rollout_ref.actor.omni_loss.loss_mode
        if loss_mode != "dpo":
            raise NotImplementedError("OmniDirectPreferenceRayTrainer currently supports omni_loss.loss_mode=dpo only.")
        self.use_reference_policy = need_reference_policy(self.config) or (loss_mode == "dpo")
        self._has_old_adapter = "old" in tuple(
            config.actor_rollout_ref.model.get("policy_state_adapters", ("default",))
        )
        if self._has_old_adapter:
            raise NotImplementedError("OmniDirectPreferenceRayTrainer does not support old-policy adapters yet.")
        self._loss_fn = None

    def _infer_reference_policy(self, batch: DataProto) -> Optional[DataProto]:
        """Compute reference-policy log-probs for chosen/rejected pairs."""
        batch_td = batch.to_tensordict()
        metadata = {
            "compute_loss": False,
            "average_log_prob": self.config.actor_rollout_ref.actor.omni_loss.average_log_prob,
            "use_dynamic_bsz": False,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.infer_actor_batch(batch_td)
        else:
            output = self.ref_policy_wg.infer_ref_batch(batch_td)
        if output is None:
            return None

        ref_logps = tu.get_tensordict(
            {
                "reference_chosen_logps": tu.get(output, "chosen_logps").float(),
                "reference_rejected_logps": tu.get(output, "rejected_logps").float(),
            }
        )
        return DataProto.from_tensordict(ref_logps)

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch_td = batch.to_tensordict()

        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        if self.config.algorithm.get("paired_preference", False) and shuffle:
            sys_logger.warning(
                "Shuffle is not supported for omni direct preference because chosen/rejected "
                "branches are packed together. Setting shuffle to False."
            )
            shuffle = False

        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        if (actor_mfu := actor_output.pop("actor/mfu", None)) is not None:
            actor_output["perf/mfu/actor"] = actor_mfu
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
