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
"""Multi-reward manager that aggregates multiple reward functions via weighted sum."""

import inspect
import logging

from verl import DataProto
from verl.utils.import_utils import load_extern_object

from .visual import VisualRewardManager

logger = logging.getLogger(__name__)


def _multi_reward_placeholder(**kwargs):
    """Sentinel function used as the upstream custom_reward_function placeholder.

    This is never called directly; MultiVisualRewardManager overrides run_single.
    """
    raise RuntimeError("_multi_reward_placeholder should never be called directly")


def _filter_kwargs(all_kwargs: dict, sig: inspect.Signature) -> dict:
    """Filter kwargs to only those declared in the function signature.

    If the function accepts **kwargs, all arguments are passed through.
    """
    params = sig.parameters
    # Check if the function accepts **kwargs
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return all_kwargs
    # Only pass declared parameters
    return {k: v for k, v in all_kwargs.items() if k in params}


class MultiVisualRewardManager(VisualRewardManager):
    """Reward manager that loads and aggregates multiple reward functions.

    Each sub-reward function is called with filtered kwargs (based on its signature),
    and the final reward is a weighted sum of all sub-rewards.

    NOTE: All sub-reward functions that need a reward model share the same
    reward_router_address (single RM instance). Deploying separate RM instances
    per reward function is not supported. If a future use case requires multiple
    distinct reward models, this architecture will need to be extended.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        # Initialize parent with the placeholder (never actually called)
        super().__init__(config, tokenizer, _multi_reward_placeholder, reward_router_address, reward_model_tokenizer)

        reward_functions_cfg = config.reward.reward_functions
        if not reward_functions_cfg:
            raise ValueError("MultiVisualRewardManager requires non-empty reward.reward_functions config")

        self._sub_rewards = []
        total_weight = 0.0
        _reserved_keys = {"path", "name", "weight"}
        for key, entry in reward_functions_cfg.items():
            path = entry["path"]
            name = entry["name"]
            weight = float(entry.get("weight", 1.0))
            total_weight += weight

            # Collect extra config fields (beyond path/name/weight) to pass to compute_score
            extra_args = {k: v for k, v in entry.items() if k not in _reserved_keys}

            fn = load_extern_object(path, name)
            sig = inspect.signature(fn)
            is_async = inspect.iscoroutinefunction(fn)

            self._sub_rewards.append(
                {
                    "key": key,
                    "fn": fn,
                    "weight": weight,
                    "sig": sig,
                    "is_async": is_async,
                    "extra_args": extra_args,
                }
            )
            logger.info(f"Loaded sub-reward '{key}': {path}:{name} (weight={weight}, async={is_async})")

        if total_weight <= 0:
            raise ValueError(
                f"Total weight of reward functions must be > 0, got {total_weight}. "
                f"Check reward.reward_functions config."
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_visual = data_item.batch["responses"]
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

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
                "model_name": self.config.reward.reward_model.model_path,
            }
            if self.reward_router_address is not None
            else {}
        )

        # Build the full kwargs dict that any sub-function might need
        all_kwargs = {
            "data_source": data_source,
            "solution_image": response_visual,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            **extra_reward_kwargs,
        }

        combined_score = 0.0
        reward_extra_info = {}

        for sub in self._sub_rewards:
            key = sub["key"]
            fn = sub["fn"]
            weight = sub["weight"]
            sig = sub["sig"]
            is_async = sub["is_async"]
            extra_args = sub["extra_args"]

            # Merge per-reward extra config fields into kwargs
            sub_kwargs = {**all_kwargs, **extra_args}
            filtered_kwargs = _filter_kwargs(sub_kwargs, sig)

            try:
                if is_async:
                    result = await fn(**filtered_kwargs)
                else:
                    result = await self.loop.run_in_executor(None, lambda f=fn, kw=filtered_kwargs: f(**kw))

                if isinstance(result, dict):
                    score = float(result["score"])
                    for rk, rv in result.items():
                        if rk == "score":
                            continue
                        reward_extra_info[f"reward/{key}/{rk}"] = rv
                else:
                    score = float(result)

            except Exception as e:
                logger.error(f"Sub-reward '{key}' raised an exception: {e}. Contributing 0 to weighted sum.")
                score = 0.0

            reward_extra_info[f"reward/{key}"] = score
            combined_score += weight * score

        reward_extra_info["reward/combined"] = combined_score
        return {"reward_score": combined_score, "reward_extra_info": reward_extra_info}
