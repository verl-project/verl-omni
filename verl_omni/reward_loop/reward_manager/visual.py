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
import logging

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score

from verl_omni.pipelines.rollout_media import PRIMARY_ARTIFACT, ArtifactContractError
from verl_omni.utils.reward_score import default_compute_score_image

from .media import _reward_extra_info

logger = logging.getLogger(__name__)


def _optional_row_field(data_item, key: str, default):
    """Read a ``non_tensor_batch`` field that not every lane populates.

    The episode batch is assembled by ``agent_loop._compute_score``, which only
    guarantees ``__num_turns__`` plus whatever the agent loop itself attached. A
    Hermes-tool-call lane (Bagel Co-RL) carries neither ``data_source`` nor
    ``reward_model``, and indexing them directly killed the run with
    ``KeyError: 'data_source'`` *inside the reward worker* -- after the rollout had
    already produced its artifacts, so the whole step was lost and the trainer died
    (measured 2026-09-22, ``ENABLE_RM=1``, ``outputs/bagel_corl_20260922_084240``).

    Reading them optionally is safe for the pixel lanes too: they always supply the
    fields, so this only changes behaviour where the old code would have crashed.
    The episode scorer discards ``data_source`` outright (``del data_source`` in
    ``agentic_multidim_reward.compute_score``), so a default cannot misroute it.
    """
    batch = getattr(data_item, "non_tensor_batch", None) or {}
    value = batch.get(key, default)
    return default if value is None else value


def _sampling_params_from_rollout(rollout_config) -> dict:
    """Build the OCR-compatible generation overrides from a rollout config."""
    params = {"max_tokens": rollout_config.get("response_length") or 4096}
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


def _is_token_response(response) -> bool:
    """True for a token-id trajectory, as opposed to pixels or a latent.

    The omni tree feeds this manager two different kinds of ``responses``:

    * **pixels** — ``uint8`` for ``output_type="image"``, floating point for ``"latent"``.
      That is what ``_validate_visual_response`` describes, and what pure image-output
      lanes (Qwen-Image FlowGRPO and friends) produce.
    * **token ids** — ``int32``/``int64``. The Bagel Joint-Training lane's episode is a
      Hermes *text* trajectory whose images live in the trajectory artifacts, so its
      ``responses`` are ids.

    Only the first kind is a picture to dtype-check; the second is a trajectory to decode.
    Treating an id tensor as pixels raised
    ``Expected uint8 pixel responses for output_type='image', got torch.int64`` inside the
    reward worker (measured 2026-09-22 with ``ENABLE_RM=1``), which failed every episode
    and took the run down through the sync replay buffer.
    """
    return isinstance(response, torch.Tensor) and response.dtype in (torch.int32, torch.int64)


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
        data_source = _optional_row_field(data_item, "data_source", "")
        ground_truth = _optional_row_field(data_item, "reward_model", {}).get("ground_truth", "")
        extra_info = _reward_extra_info(data_item)
        # Hand the scorer whichever input its branch expects. A token trajectory is decoded
        # into ``solution_str`` (the branch this manager historically could not reach); a
        # pixel/latent response keeps riding ``solution_image`` under the dtype/artifact guard.
        if _is_token_response(response_visual):
            solution_str = self.tokenizer.decode(response_visual.reshape(-1).tolist(), skip_special_tokens=True)
            score_input = {"solution_str": solution_str}
        else:
            _validate_visual_response(
                response_visual,
                self.config,
                is_validate=data_item.meta_info.get("validate", False),
                extra_info=extra_info,
            )
            score_input = {"solution_image": response_visual}

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores
        if not ground_truth and "bagel_corl" not in (extra_info or {}):
            # Not fatal (the episode scorer returns a zero result for a missing
            # ``task_type``), but a fleet of silent zeros looks exactly like a model that
            # never learns, so say it once and let the metric surface it.
            #
            # Scoped on purpose: the mid-loop scoring row is built by
            # ``bagel_corl_rm._default_data_builder``, which carries the images in
            # ``extra_info['bagel_corl']`` and *deliberately* leaves ``ground_truth``
            # empty, so warning there fired on every scored image (11 times per poll,
            # measured 2026-09-22) and buried the real signal.
            logger.warning(
                "VisualRewardManager: episode row carries no reward_model.ground_truth "
                "(data_source=%r); the episode scorer will return a zero reward for it.",
                data_source,
            )

        rm_rollout = self.config.reward.reward_model.rollout
        # Keep the scorer's own sampling defaults and only override the length/seed.
        sampling_params = _sampling_params_from_rollout(rm_rollout)

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
                ground_truth=ground_truth,
                extra_info=extra_info,
                **score_input,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **score_input,
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
