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
"""Reward manager for audio waveforms produced by omni rollouts."""

import asyncio
import inspect
import math
from collections.abc import Mapping
from functools import partial

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score


class AudioRewardManager(RewardManagerBase):
    """Route one validated waveform per rollout to a custom reward function."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        if (
            compute_score is None
            or compute_score is _upstream_default_compute_score
            or (isinstance(compute_score, partial) and compute_score.func is _upstream_default_compute_score)
        ):
            raise ValueError("AudioRewardManager requires reward.custom_reward_function.")
        super().__init__(config, tokenizer, compute_score)
        self.is_async_reward_score = inspect.iscoroutinefunction(compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    @staticmethod
    def _mapping(value):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"Audio reward metadata must be a mapping, got {type(value).__name__}.")
        return dict(value)

    @classmethod
    def _extract_audio(cls, extra_info):
        audio = extra_info.get("audio")
        sample_rate = extra_info.get("audio_sample_rate")
        if audio is None:
            raise KeyError("Audio reward requires extra_info['audio'] from the rollout.")
        if sample_rate is None:
            raise KeyError("Audio reward requires extra_info['audio_sample_rate'] from the rollout.")

        try:
            if isinstance(audio, np.ndarray) and audio.dtype == object:
                audio = np.asarray(audio, dtype=np.float32)
            waveform = torch.as_tensor(audio).detach().float().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Audio reward could not convert the waveform to numeric samples.") from exc
        while waveform.ndim > 1 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected one mono waveform with shape (T,) or leading singleton dimensions, "
                f"got {tuple(waveform.shape)}."
            )
        if waveform.numel() == 0:
            raise ValueError("Audio reward received an empty waveform.")
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio reward received a waveform containing NaN or infinity.")

        if isinstance(sample_rate, np.ndarray | torch.Tensor):
            sample_rate_count = sample_rate.size if isinstance(sample_rate, np.ndarray) else sample_rate.numel()
            if sample_rate_count != 1:
                raise ValueError("Audio reward requires one scalar sample rate per waveform.")
            sample_rate = sample_rate.item()
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | float):
            raise TypeError(f"Audio sample rate must be numeric, got {type(sample_rate).__name__}.")
        if not math.isfinite(float(sample_rate)) or float(sample_rate) <= 0 or float(sample_rate) != int(sample_rate):
            raise ValueError(f"Audio sample rate must be a positive integer, got {sample_rate!r}.")
        return waveform.numpy().astype(np.float32, copy=False), int(sample_rate)

    async def run_single(self, data: DataProto) -> dict:
        if len(data) != 1:
            raise ValueError(f"AudioRewardManager scores one sample at a time, got batch size {len(data)}.")
        item = data[0]
        batch = item.non_tensor_batch
        extra_info = self._mapping(batch.get("extra_info", {}))
        extra_info.update(self._mapping(batch.get("tool_extra_fields")))
        for key in ("audio", "audio_sample_rate"):
            if key in batch and batch[key] is not None:
                extra_info[key] = batch[key]
        if "__num_turns__" in batch:
            extra_info["num_turns"] = batch["__num_turns__"]
        if "global_steps" in batch:
            extra_info["global_steps"] = batch["global_steps"]
        ground_truth = batch["reward_model"]["ground_truth"]
        audio = await asyncio.to_thread(self._extract_audio, extra_info)
        kwargs = {
            "data_source": batch["data_source"],
            "solution_audio": audio,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
        }
        if self.is_async_reward_score:
            result = await self.compute_score(**kwargs)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        if isinstance(result, dict):
            if "score" not in result:
                raise ValueError("Audio reward result dictionary is missing 'score'.")
            score = float(result["score"])
            reward_extra_info = {key: value for key, value in result.items() if key != "score"}
        else:
            score = float(result)
            reward_extra_info = {"acc": score}
        if not math.isfinite(score):
            raise ValueError(f"Audio reward must be finite, got {score!r}.")
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
