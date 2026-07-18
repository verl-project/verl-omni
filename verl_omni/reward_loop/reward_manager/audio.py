# Copyright 2026 Gulp AI Inc and/or its affiliates
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
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score as _upstream_default_compute_score

from verl_omni.utils.reward_score import default_compute_score_audio


def _load_qwen_tts_decoder(model_path, device):
    """Load the qwen-tts speech tokenizer (code2wav vocoder). Returns (decode_fn, codebook_max)."""
    import os

    from qwen_tts import Qwen3TTSTokenizer
    from transformers.utils import cached_file

    cfg = cached_file(model_path, "speech_tokenizer/config.json")
    dec = Qwen3TTSTokenizer.from_pretrained(os.path.dirname(cfg), device_map=device, dtype=torch.bfloat16)
    codebook_size = getattr(getattr(dec.model.config, "decoder_config", None), "codebook_size", 2048)

    def decode(codes):
        wavs, sr = dec.decode([{"audio_codes": codes}])
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)

    return decode, int(codebook_size) - 1


# Codec decoders dispatched on reward.audio.codec; add new codec models here.
_CODEC_DECODERS = {"qwen_tts": _load_qwen_tts_decoder}


class AudioRewardManager(RewardManagerBase):
    """The reward manager for audio response.

    The rollout surfaces either a decoded waveform or codec tokens; codec tokens are decoded
    reward-side with the decoder selected by reward.audio.codec, and the (wav, sr) tuple is
    passed to compute_score as solution_audio.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)

        if compute_score is None or compute_score is _upstream_default_compute_score:
            self.compute_score = default_compute_score_audio
        else:
            self.compute_score = compute_score

        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        audio_cfg = getattr(getattr(config, "reward", None), "audio", None) or {}
        codec = audio_cfg.get("codec", "qwen_tts")
        if codec not in _CODEC_DECODERS:
            raise ValueError(f"Unknown reward.audio.codec {codec!r}, available: {sorted(_CODEC_DECODERS)}")
        self._codec = codec
        self._codec_eos = int(audio_cfg.get("codec_eos_token_id", 2150))
        self._decode = None
        self._codebook_max = None
        self._decoder_lock = threading.Lock()
        self._device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        model_cfg = getattr(getattr(config, "actor_rollout_ref", None), "model", None)
        self._decoder_model_path = audio_cfg.get("decoder_model_path") or getattr(model_cfg, "path", None)
        # Scoring runs in a small bounded pool: the decode and feature-extraction paths are
        # CPU-bound and not thread-safe under CUDA, so parallelism comes from reward.num_workers
        # processes rather than threads.
        self._score_executor = ThreadPoolExecutor(max_workers=int(audio_cfg.get("score_threads", 1)))

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Per-sample audio rewards: ``rm_scores`` has shape ``(batch_size, 1)``."""
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    def _get_decoder(self):
        if self._decode is None:
            with self._decoder_lock:
                if self._decode is None:
                    loader = _CODEC_DECODERS[self._codec]
                    self._decode, self._codebook_max = loader(self._decoder_model_path, self._device)
        return self._decode

    def _decode_codes(self, extra_info):
        """Decode rollout codec tokens (T, 16) from extra_info into (wav, sr), or None."""
        codes = extra_info.get("tts_audio_codes")
        if codes is None:
            return None
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2:
            return None
        # Trim frames after codec eos, then clamp: out-of-range indices device-assert in decode.
        eos = (codes[:, 0] == self._codec_eos).nonzero().flatten()
        if len(eos):
            codes = codes[: int(eos[0])]
        if codes.shape[0] == 0:
            return None
        decode = self._get_decoder()
        return decode(codes.clamp_(0, self._codebook_max))

    def _extract_audio(self, data_item, extra_info):
        """The generated audio as a (wav, sr) tuple, or None when nothing decodes."""
        nb = data_item.non_tensor_batch
        audio = nb.get("audio")
        sr = nb.get("sr")
        if audio is None and isinstance(nb.get("multimodal_output"), dict):
            mm = nb["multimodal_output"]
            audio, sr = mm.get("audio"), mm.get("sr")
        if audio is None:
            return self._decode_codes(extra_info)
        if isinstance(audio, list | tuple):
            audio = torch.cat([a if torch.is_tensor(a) else torch.as_tensor(a) for a in audio], dim=-1)
        if torch.is_tensor(audio):
            audio = audio.float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None
        if isinstance(sr, list | tuple):
            sr = sr[-1]
        sr = int(sr.item()) if hasattr(sr, "item") else int(sr or 24000)
        return audio, sr

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
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
        if self.is_async_reward_score:
            solution_audio = await self.loop.run_in_executor(
                self._score_executor, lambda: self._extract_audio(data_item, extra_info)
            )
            result = await self.compute_score(
                data_source=data_source,
                solution_audio=solution_audio,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                self._score_executor,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_audio=self._extract_audio(data_item, extra_info),
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
