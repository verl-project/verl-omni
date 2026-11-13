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
from functools import partial
from pathlib import Path

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score

from verl_omni.pipelines.rollout_media import PRIMARY_ARTIFACT, ArtifactContractError
from verl_omni.utils.reward_score import default_compute_score_image

from .media import _reward_extra_info

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UNIFIED_REWARD_PKG_PATHS = frozenset(
    {
        "pkg://verl_omni.utils.reward_score.vlm_as_a_judge",
        "pkg://verl_omni.utils.reward_score.unified_reward",
    }
)
_UNIFIED_REWARD_SOURCE_FILES = frozenset(
    {
        (_REPO_ROOT / "verl_omni" / "utils" / "reward_score" / "vlm_as_a_judge.py").resolve(),
        (_REPO_ROOT / "verl_omni" / "utils" / "reward_score" / "unified_reward.py").resolve(),
    }
)


def _is_repo_unified_reward_path(path) -> bool:
    """Whether a configured reward path names this repository's UnifiedReward module."""
    if not isinstance(path, str) or not path:
        return False
    if path in _UNIFIED_REWARD_PKG_PATHS:
        return True
    if path.startswith("file://"):
        path = path[len("file://") :]
    candidate = Path(path).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [candidate, _REPO_ROOT / candidate]
    return any(item.resolve() in _UNIFIED_REWARD_SOURCE_FILES for item in candidates)


def _partial_sampling_params(compute_score) -> dict | None:
    """Return sampling params supplied by a standard ``functools.partial``, if any."""
    scorer = compute_score
    while isinstance(scorer, partial):
        if "sampling_params" in scorer.keywords:
            return scorer.keywords["sampling_params"]
        scorer = scorer.func
    return None


def _is_legacy_unified_reward_entry(compute_score, config) -> bool:
    """Whether a callable is the repository UnifiedReward scorer for this compat entry.

    Direct injection is identified by the scorer's defining module and name.
    Config-loaded callables are identified by path plus function name so a
    third-party file exporting the same name keeps the general rollout sampling
    behavior.
    """
    scorer = compute_score
    while isinstance(scorer, partial):
        scorer = scorer.func
    if (
        getattr(scorer, "__module__", None) == "verl_omni.utils.reward_score.vlm_as_a_judge"
        and getattr(scorer, "__name__", None) == "compute_score_unified_reward"
    ):
        return True
    reward_fn_config = config.reward.get("custom_reward_function") or {}
    return reward_fn_config.get("name") == "compute_score_unified_reward" and _is_repo_unified_reward_path(
        reward_fn_config.get("path")
    )


def _sampling_params_from_rollout(rollout_config) -> dict:
    """Build judge generation overrides from a reward rollout config.

    The response length is always forwarded. Temperature and top-p are
    forwarded together once either is tuned away from the upstream rollout
    defaults, and top-k is forwarded when it is set. Configs that keep the
    upstream defaults retain the scorer's own sampling setup.
    """
    params = {"max_tokens": rollout_config.get("response_length") or 4096}
    top_k = rollout_config.get("top_k", -1)
    if top_k != -1:
        params["top_k"] = top_k
    temperature = rollout_config.get("temperature", 1.0)
    top_p = rollout_config.get("top_p", 1.0)
    if temperature != 1.0 or top_p != 1.0:
        params["temperature"] = temperature
        params["top_p"] = top_p
    if rollout_config.get("full_determinism", False):
        params["seed"] = rollout_config.get("seed", 42)
    return params


def _validate_visual_response(response_visual, config, *, is_validate: bool, extra_info=None) -> None:
    if extra_info is not None and "media_artifacts" in extra_info:
        artifacts = extra_info["media_artifacts"]
        primary = extra_info.get(PRIMARY_ARTIFACT)
        if primary not in artifacts:
            raise ArtifactContractError(f"Reward entry requires an explicit primary artifact, got {primary!r}")
        artifact = artifacts[primary]
        if not isinstance(response_visual, torch.Tensor) or (
            response_visual.dtype != artifact.data.dtype or response_visual.shape != artifact.data.shape
        ):
            raise ArtifactContractError(
                f"{artifact.context}, artifact={primary!r}: responses projection does not match declared primary"
            )
        try:
            # Exact values, but NaN positions match and the device is not part of the contract.
            torch.testing.assert_close(
                response_visual,
                artifact.data,
                rtol=0,
                atol=0,
                equal_nan=True,
                check_device=False,
            )
        except AssertionError as error:
            raise ArtifactContractError(
                f"{artifact.context}, artifact={primary!r}: responses projection does not match declared primary"
            ) from error
        return
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
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = _reward_extra_info(data_item)
        _validate_visual_response(
            response_visual,
            self.config,
            is_validate=data_item.meta_info.get("validate", False),
            extra_info=extra_info,
        )

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        rm_rollout = self.config.reward.reward_model.rollout
        # Unset rollout sampling fields keep the scorer's own tuned defaults.
        sampling_params = _sampling_params_from_rollout(rm_rollout)
        if _is_legacy_unified_reward_entry(self.compute_score, self.config):
            # The compat entry keeps UnifiedReward's own 512-token default. Its
            # explicit reward_kwargs still win through get_custom_reward_fn.
            sampling_params.pop("max_tokens", None)
            if _partial_sampling_params(self.compute_score) is not None:
                sampling_params = None

        extra_reward_kwargs = {}
        if self.reward_router_address is not None:
            extra_reward_kwargs = {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
                "model_name": self.config.reward.reward_model.model_path,
            }
            if sampling_params is not None:
                extra_reward_kwargs["sampling_params"] = sampling_params
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
