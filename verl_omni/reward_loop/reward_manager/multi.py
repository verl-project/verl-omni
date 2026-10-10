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
"""Input-agnostic multi-reward execution and weighted aggregation."""

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from collections import ChainMap

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.import_utils import load_extern_object

from verl_omni.workers.config.reward import get_reward_model_entries, resolve_reward_model_name

from .media import _reward_extra_info
from .visual import _sampling_params_from_rollout, _validate_visual_response

logger = logging.getLogger(__name__)


def _multi_reward_placeholder(**kwargs):
    """Sentinel function used as the upstream custom_reward_function placeholder.

    This is never called directly; MultiRewardManager overrides run_single.
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


class MultiRewardManager(RewardManagerBase, ABC):
    """Load and aggregate reward functions without owning an input modality.

    Each sub-reward function is called with filtered kwargs (based on its signature),
    and the final reward is a weighted sum of all sub-rewards.

    A sub-reward may reference a named model. The selected executor supplies
    inference access, while the configured reward function owns score semantics.
    Input-specific managers implement :meth:`_build_reward_kwargs` and delegate
    scorer execution and aggregation to this class.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        RewardManagerBase.__init__(self, config, tokenizer, _multi_reward_placeholder)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        self._engine_reward_executors = {}
        self._native_reward_executors = {}
        concurrency = config.reward.get("multi_reward_concurrency", 1)
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
            raise ValueError("reward.multi_reward_concurrency must be a positive integer")
        self._multi_reward_concurrency = concurrency
        self._multi_reward_semaphore = asyncio.Semaphore(concurrency) if concurrency > 1 else None

        reward_functions_cfg = config.reward.reward_functions
        reward_models_cfg = get_reward_model_entries(config)
        if not reward_functions_cfg:
            raise ValueError("MultiRewardManager requires non-empty reward.reward_functions config")

        self._sub_rewards = []
        total_weight = 0.0
        _reserved_keys = {"path", "name", "weight", "required", "model", "independent", "use_rollout_sampling_params"}
        for key, entry in reward_functions_cfg.items():
            model_name = resolve_reward_model_name(key, entry, reward_models_cfg)
            use_rollout_sampling_params = entry.get("use_rollout_sampling_params", False)
            if not isinstance(use_rollout_sampling_params, bool):
                raise TypeError("use_rollout_sampling_params must be a boolean")
            if use_rollout_sampling_params and (
                model_name is None or reward_models_cfg[model_name].get("backend") != "engine"
            ):
                raise ValueError("use_rollout_sampling_params requires a named engine reward model")
            path = entry.get("path")
            name = entry.get("name")
            if (path is None) != (name is None):
                raise ValueError(f"Reward function {key!r} must set both path and name")
            if model_name is None and path is None:
                raise ValueError(f"Reward function {key!r} requires path/name")
            if model_name is not None and path is None:
                raise ValueError(f"Model-backed reward function {key!r} requires path/name")
            weight = float(entry.get("weight", 1.0))
            required_value = entry.get("required", False)
            if isinstance(required_value, str):
                normalized = required_value.lower()
                if normalized not in {"true", "false"}:
                    raise ValueError(f"Invalid required value: {required_value!r}")
                required = normalized == "true"
            elif isinstance(required_value, bool):
                required = required_value
            else:
                raise TypeError(f"required must be a boolean, got {type(required_value).__name__}")
            independent = entry.get("independent", False)
            if not isinstance(independent, bool):
                raise TypeError(f"Reward function {key!r} independent must be a boolean")
            total_weight += weight

            # Collect non-manager fields to pass to compute_score.
            extra_args = {k: v for k, v in entry.items() if k not in _reserved_keys}

            fn = load_extern_object(path, name) if path is not None else None
            sig = inspect.signature(fn) if fn is not None else None
            is_async = inspect.iscoroutinefunction(fn) if fn is not None else True
            if independent and (
                not is_async
                or (model_name is not None and reward_models_cfg.get(model_name, {}).get("backend") != "engine")
            ):
                raise ValueError(
                    f"Reward function {key!r} independent requires an async scorer with no named model "
                    "or an engine-backed model"
                )

            self._sub_rewards.append(
                {
                    "key": key,
                    "fn": fn,
                    "weight": weight,
                    "required": required,
                    "independent": independent,
                    "sig": sig,
                    "is_async": is_async,
                    "extra_args": extra_args,
                    "model": model_name,
                    "use_rollout_sampling_params": use_rollout_sampling_params,
                }
            )
            logger.info(
                "Loaded sub-reward '%s': %s (weight=%s, required=%s, async=%s)",
                key,
                model_name or f"{path}:{name}",
                weight,
                required,
                is_async,
            )

        if total_weight <= 0:
            raise ValueError(
                f"Total weight of reward functions must be > 0, got {total_weight}. "
                f"Check reward.reward_functions config."
            )

    def set_reward_executors(self, engine_reward_executors: dict | None, native_reward_executors: dict | None) -> None:
        """Attach per-worker executors for configured engine/native models."""
        self._engine_reward_executors = engine_reward_executors or {}
        self._native_reward_executors = native_reward_executors or {}

    @abstractmethod
    async def _build_reward_kwargs(self, data_item: DataProto) -> dict:
        """Build scorer kwargs for one sample in a modality-specific subclass."""

    async def _run_multi_reward(self, all_kwargs: dict) -> dict:
        """Execute configured reward terms and preserve their weighted outputs."""
        combined_score = 0.0
        reward_extra_info = {}
        admitted_tasks = []

        def prepare_term(sub):
            sig = sub["sig"]
            extra_args = sub["extra_args"]
            model_name = sub["model"]

            # Merge per-reward extra config fields into kwargs
            sub_kwargs = {**all_kwargs, **extra_args}
            filtered_kwargs = _filter_kwargs(sub_kwargs, sig) if sig is not None else {}

            if model_name is not None:
                executor = self._engine_reward_executors.get(model_name)
                if executor is not None:
                    if sub["use_rollout_sampling_params"] and "sampling_params" not in extra_args:
                        model = get_reward_model_entries(self.config)[model_name]
                        rollout = ChainMap(
                            model.get("rollout") or {}, self.config.reward.reward_model.get("rollout") or {}
                        )
                        sub_kwargs["sampling_params"] = _sampling_params_from_rollout(rollout)
                else:
                    executor = self._native_reward_executors.get(model_name)
                if executor is None:
                    raise RuntimeError(f"Reward model {model_name!r} is not available in this worker")
                reward_kwargs = getattr(executor, "reward_kwargs", None)
                if reward_kwargs is None:
                    raise RuntimeError(f"Reward model {model_name!r} cannot be used with a reward function")
                filtered_kwargs = _filter_kwargs({**sub_kwargs, **reward_kwargs()}, sig)
            return filtered_kwargs

        async def run_term(sub, filtered_kwargs):
            fn = sub["fn"]

            async def invoke():
                if sub["is_async"]:
                    return await fn(**filtered_kwargs)
                future = self.loop.run_in_executor(None, lambda f=fn, kw=filtered_kwargs: f(**kw))
                try:
                    return await asyncio.shield(future)
                except asyncio.CancelledError:
                    # The thread keeps running after its await is cancelled.
                    # Retain ownership (and any semaphore permit) until it exits.
                    settled = asyncio.gather(future, return_exceptions=True)
                    while not settled.done():
                        try:
                            await asyncio.shield(settled)
                        except asyncio.CancelledError:
                            pass
                    raise

            if self._multi_reward_semaphore is None:
                return await invoke()
            async with self._multi_reward_semaphore:
                return await invoke()

        async def drain_tasks():
            for task in admitted_tasks:
                task.cancel()
            if admitted_tasks:
                settled = asyncio.gather(*admitted_tasks, return_exceptions=True)
                while not settled.done():
                    try:
                        await asyncio.shield(settled)
                    except asyncio.CancelledError:
                        # Repeated caller cancellation must not release accepted tasks.
                        for task in admitted_tasks:
                            task.cancel()

        try:
            index = 0
            while index < len(self._sub_rewards):
                sub = self._sub_rewards[index]
                if sub["independent"]:
                    window = []
                    while (
                        index < len(self._sub_rewards)
                        and self._sub_rewards[index]["independent"]
                        and len(window) < self._multi_reward_concurrency
                    ):
                        window.append(self._sub_rewards[index])
                        index += 1
                    prepared = [prepare_term(term) for term in window]
                    admitted_tasks = []
                    for term, kwargs in zip(window, prepared, strict=True):
                        admitted_tasks.append(asyncio.create_task(run_term(term, kwargs)))
                    results = await asyncio.gather(*admitted_tasks, return_exceptions=True)
                    admitted_tasks = []
                else:
                    window = [sub]
                    index += 1
                    filtered_kwargs = prepare_term(sub)
                    try:
                        results = [await run_term(sub, filtered_kwargs)]
                    except Exception as exc:
                        results = [exc]

                for term, result in zip(window, results, strict=True):
                    key = term["key"]
                    if isinstance(result, BaseException):
                        if not isinstance(result, Exception):
                            raise result
                        if term["required"]:
                            raise RuntimeError(f"Required sub-reward '{key}' failed: {result}") from result
                        logger.error(
                            "Sub-reward '%s' raised an exception: %s. Contributing 0 to weighted sum.",
                            key,
                            result,
                            exc_info=(type(result), result, result.__traceback__),
                        )
                        reward_extra_info[f"reward/{key}/errors"] = 1
                        score = 0.0
                    else:
                        try:
                            if isinstance(result, dict):
                                score = float(result["score"])
                                for rk, rv in result.items():
                                    if rk != "score":
                                        reward_extra_info[f"reward/{key}/{rk}"] = rv
                            else:
                                score = float(result)
                        except Exception as exc:
                            if term["required"]:
                                raise RuntimeError(f"Required sub-reward '{key}' failed: {exc}") from exc
                            logger.exception(
                                "Sub-reward '%s' raised an exception: %s. Contributing 0 to weighted sum.",
                                key,
                                exc,
                            )
                            reward_extra_info[f"reward/{key}/errors"] = 1
                            score = 0.0
                    reward_extra_info[f"reward/{key}"] = score
                    combined_score += term["weight"] * score
        except BaseException:
            await drain_tasks()
            raise

        reward_extra_info["reward/combined"] = combined_score
        return {"reward_score": combined_score, "reward_extra_info": reward_extra_info}

    async def run_single(self, data: DataProto) -> dict:
        """Prepare one sample through the subclass input contract, then aggregate."""
        if len(data) != 1:
            raise ValueError(f"{type(self).__name__} scores one sample at a time, got batch size {len(data)}.")
        return await self._run_multi_reward(await self._build_reward_kwargs(data[0]))


class MultiVisualRewardManager(MultiRewardManager):
    """Visual input contract backed by the shared multi-reward aggregator."""

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Keep the historical per-sample visual score layout."""
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    async def _build_reward_kwargs(self, data_item: DataProto) -> dict:
        """Preserve the existing visual manager input and router contract."""
        response_visual = data_item.batch["responses"]
        _validate_visual_response(
            response_visual,
            self.config,
            is_validate=data_item.meta_info.get("validate", False),
        )
        batch = data_item.non_tensor_batch
        extra_info = _reward_extra_info(data_item)
        extra_info["num_turns"] = batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = batch.get("reward_scores", {})

        reward_kwargs = {
            "data_source": batch["data_source"],
            "solution_image": response_visual,
            "ground_truth": batch["reward_model"]["ground_truth"],
            "extra_info": extra_info,
        }
        if self.reward_router_address is not None:
            rm_rollout = self.config.reward.reward_model.rollout
            sampling_params = _sampling_params_from_rollout(rm_rollout)
            reward_kwargs.update(
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
                model_name=self.config.reward.reward_model.model_path,
                sampling_params=sampling_params,
            )
        return reward_kwargs
