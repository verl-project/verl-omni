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

import inspect

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score

from verl_omni.utils.reward_score import default_compute_score_image


def _validate_visual_response(response_visual, config, *, is_validate: bool) -> None:
    rollout_config = config.actor_rollout_ref.rollout
    pipeline_config = rollout_config.val_kwargs.pipeline if is_validate else rollout_config.pipeline
    output_type = pipeline_config.get("output_type", "image")

    if output_type == "latent":
        if not isinstance(response_visual, torch.Tensor) or not response_visual.dtype.is_floating_point:
            dtype = getattr(response_visual, "dtype", type(response_visual))
            raise ValueError(f"Expected floating-point latent responses, got {dtype}.")
    elif not isinstance(response_visual, torch.Tensor) or response_visual.dtype != torch.uint8:
        dtype = getattr(response_visual, "dtype", type(response_visual))
        raise ValueError(f"Expected uint8 pixel responses for output_type={output_type!r}, got {dtype}.")


class VisualRewardManager(RewardManagerBase):
    """The reward manager for visual response."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)

        if compute_score is None or compute_score is _upstream_default_compute_score:
            self.compute_score = default_compute_score_image
        else:
            self.compute_score = compute_score

        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Per-sample image rewards: ``rm_scores`` has shape ``(batch_size, 1)``."""
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_visual = data_item.batch["responses"]
        _validate_visual_response(response_visual, self.config, is_validate=data_item.meta_info.get("validate", False))
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        rm_rollout = self.config.reward.reward_model.rollout
        reward_model_cfg = self.config.reward.reward_model
        sampling_params = {
            "temperature": rm_rollout.temperature,
            "top_k": rm_rollout.top_k,
            "top_p": rm_rollout.top_p,
            "max_tokens": getattr(rm_rollout, "response_length", None) or 4096,
        }
        if reward_model_cfg.get("full_determinism", False):
            sampling_params["seed"] = reward_model_cfg.get("seed", 42)

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
                "model_name": self.config.reward.reward_model.model_path,
                "sampling_params": sampling_params,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_image=response_visual,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_image=response_visual,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                if key == "score":
                    continue
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
