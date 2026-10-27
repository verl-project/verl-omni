# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 AlphaGRPO Authors.
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

"""Two-stage synchronous AlphaGRPO rollout using the configured reward workers."""

from copy import deepcopy

import numpy as np
import ray
import torch
from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop

from .reflection import REFLECTION_PROMPT, penalize_failed_refinements, select_initial_images


class BagelReflectionAgentLoop(DiffusionSingleTurnAgentLoop):
    """Use native framing for initial generation and image-conditioned reflection."""

    async def run(self, sampling_params, **kwargs):
        """Generate one stage; selection and reward comparisons belong to the manager."""
        sampling_params = dict(sampling_params)
        if kwargs.get("condition_images") is None:
            sampling_params.update(
                alphagrpo_stage1=True,
                noise_level=0.0,
                logprobs=False,
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
                cfg_interval=(0.4, 1.0),
                cfg_renorm_type="global",
                cfg_renorm_min=0.0,
                num_inference_steps=self.rollout_config.val_kwargs.pipeline.num_inference_steps,
            )
        else:
            ids = list(kwargs["prompt_token_ids"])
            caption = kwargs["reward_model"]["ground_truth"]
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("Reflection requires the original caption in reward_model.ground_truth.")
            kwargs["prompt_token_ids"] = [
                ids[0],
                *self.tokenizer.encode(REFLECTION_PROMPT.format(caption), add_special_tokens=False),
                ids[-1],
            ]
            if sampling_params.get("seed") is not None:
                sampling_params["seed"] = (int(sampling_params["seed"]) + 2**31) % (2**63 - 1)
        return await super().run(sampling_params, **kwargs)


class BagelReflectionAgentLoopManager(AgentLoopManager):
    """Select initial images across all workers before dispatching refinement groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.trainer.get("use_v1", True):
            raise ValueError("AlphaGRPO reflection currently requires trainer.use_v1=false (synchronous batches).")
        if not self.reward_loop_worker_handles:
            raise ValueError("Reflection requires streaming reward workers on a separate reward resource pool.")
        if (
            self.model_config.algorithm != "alphagrpo"
            or self.rollout_config.pipeline.get("output_type", "image") != "image"
        ):
            raise ValueError("Reflection requires the AlphaGRPO adapter and decoded image output.")
        if self.config.reward.reward_model.rollout.free_cache_engine:
            raise ValueError(
                "Reflection requires a resident judge: set reward.reward_model.rollout.free_cache_engine=false."
            )
        if self.config.reward.custom_reward_function.name != "compute_score_dvreward":
            raise ValueError("Reflection requires raw DVReward; format bonuses must not affect image comparisons.")
        if self.rollout_config.n < 2:
            raise ValueError("Reflection training needs at least two refinements per prompt.")
        self.agent_loop_workers_class = ray.remote(DiffusionAgentLoopWorker)

    @auto_await
    async def generate_sequences(self, prompts):
        """Return refined trajectories with FPR-adjusted rewards and preserved source prompts."""
        if "prompt_token_ids" not in prompts.non_tensor_batch or "uid" not in prompts.non_tensor_batch:
            raise ValueError("Reflection requires native prompt_token_ids and prompt UIDs.")
        uids = prompts.non_tensor_batch["uid"]
        group_size = (
            self.rollout_config.val_kwargs.n if prompts.meta_info.get("validate", False) else self.rollout_config.n
        )
        select_initial_images(torch.zeros(len(prompts)), uids, group_size)
        candidate_count = max(group_size // 2, 1)
        groups = {}
        for row, uid in enumerate(uids):
            groups.setdefault(str(uid), []).append(row)
        initial_rows = [row for rows in groups.values() for row in rows[:candidate_count]]
        initial_prompts = deepcopy(prompts[initial_rows])
        if len(initial_prompts) % len(self.agent_loop_workers):
            raise ValueError(
                "Initial candidate count must divide evenly across agent workers; reduce agent.num_workers."
            )
        initial = await super().generate_sequences(initial_prompts)
        if "rm_scores" not in initial.batch:
            raise ValueError("Initial images must be scored before refinement.")
        initial_rewards = initial.batch["rm_scores"].squeeze(-1)
        candidate_uids = initial_prompts.non_tensor_batch["uid"]
        selected_candidates = select_initial_images(initial_rewards, candidate_uids, candidate_count)
        selected_by_uid = {str(uid): int(index) for uid, index in zip(candidate_uids, selected_candidates, strict=True)}
        selected = torch.tensor([selected_by_uid[str(uid)] for uid in uids], dtype=torch.long)
        images = initial.batch["responses"]
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Reflection requires initial images as uint8 RGB [B, 3, H, W].")
        refine = deepcopy(prompts)
        conditions = np.empty(len(prompts), dtype=object)
        for row, source in enumerate(selected.tolist()):
            conditions[row] = [Image.fromarray(images[source].permute(1, 2, 0).cpu().numpy())]
        refine.non_tensor_batch["condition_images"] = conditions
        refined = await super().generate_sequences(refine)
        rewards = refined.batch["rm_scores"].squeeze(-1)
        baselines = initial_rewards[selected]
        adjusted, no_improvement = penalize_failed_refinements(rewards, baselines, uids)
        if not prompts.meta_info.get("validate", False):
            refined.batch["rm_scores"] = adjusted[:, None]
            zero_advantages = all(bool((adjusted[rows] == adjusted[rows[0]]).all()) for rows in groups.values())
            refined.meta_info["skip_actor_update"] = no_improvement or zero_advantages
        for key in ("prompts", "attention_mask"):
            if key in initial.batch:
                refined.batch[key] = initial.batch[key][selected]
        refined.non_tensor_batch.pop("condition_images", None)
        for key, value in prompts.non_tensor_batch.items():
            refined.non_tensor_batch[key] = value
        refined.non_tensor_batch["initial_dvreward"] = baselines.cpu().numpy()
        refined.non_tensor_batch["refined_dvreward"] = rewards.cpu().numpy()
        refined.non_tensor_batch["reflection_improved"] = (rewards > baselines).float().cpu().numpy()
        refined.non_tensor_batch["__num_turns__"] = np.full(len(prompts), 4, dtype=np.int32)
        refined.meta_info["reward_extra_keys"] = sorted(
            set(refined.meta_info.get("reward_extra_keys", []))
            | {"initial_dvreward", "refined_dvreward", "reflection_improved"}
        )
        refined.meta_info["timing"] = {
            **{f"initial/{key}": value for key, value in initial.meta_info.get("timing", {}).items()},
            **{f"refinement/{key}": value for key, value in refined.meta_info.get("timing", {}).items()},
        }
        return refined
