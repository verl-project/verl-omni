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

import asyncio
import logging

from verl import DataProto

from .audio import AudioRewardManager

logger = logging.getLogger(__name__)


class AudioJudgeRewardManager(AudioRewardManager):
    """Online-DPO reward: score a prompt's rollout candidates by direct LLM-judge A-vs-B comparison.

    Reuses AudioRewardManager's codec decode path but skips the in-process reward functions. A prompt's
    rollout.n candidates are buffered by uid and judged against each other in one /rank call; sj_score is
    each candidate's rating. The actor pairs chosen/rejected within the uid group by sj_score for
    tts_dpo_loss (adv_estimator grpo consumes reward_score but the DPO loss ignores advantages).

    REQUIRES reward.num_workers=1: a prompt's siblings must rendezvous in one worker's event loop. Judge
    endpoints come from reward.judge_urls (any /rank server, e.g. examples/qwen3_tts_dpo/judge_server.py).
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
        reward_cfg = config.reward
        num_workers = int(reward_cfg.get("num_workers", 1) or 1)
        if num_workers != 1:
            raise ValueError(
                "AudioJudgeRewardManager requires reward.num_workers=1: a prompt's rollout.n candidates "
                f"rendezvous in one worker's event loop for the pairwise judge, got {num_workers}."
            )
        urls = reward_cfg.get("judge_urls", None) or ["http://localhost:8901"]
        self._judge_urls = [str(u).rstrip("/") for u in urls]
        self._judge_debias = bool(reward_cfg.get("judge_debias", True))
        self._judge_timeout_s = float(reward_cfg.get("judge_timeout_s", 1800))
        self._group_size = int(config.actor_rollout_ref.rollout.n)
        self._groups: dict = {}
        self._group_lock = asyncio.Lock()

    def _decode_wav_bytes(self, data_item, extra_info):
        """Decode this candidate to in-memory PCM16 WAV bytes for the judge, or None on synth failure."""
        import io

        res = self._extract_audio(data_item, extra_info)
        if res is None or res[0] is None or res[0].size == 0:
            return None
        import soundfile as sf

        wav, sr = res
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _judge_group(self, text, blobs):
        """Direct A-vs-B judging of a prompt's candidates. POSTs base64 WAV bytes to /rank and returns one
        score per blob. Synth-failed blobs (None) get a worst score so they become the rejected side; a
        judge failure returns a neutral score so the group contributes no signal this step."""
        import base64
        import json
        import urllib.request

        scores = [-10.0] * len(blobs)
        valid = [(i, b) for i, b in enumerate(blobs) if b]
        if not valid:
            return scores
        if len(valid) == 1:
            scores[valid[0][0]] = 5.0  # lone survivor beats its failed sibling(s)
            return scores
        idxs = [i for i, _ in valid]
        wavs_b64 = [base64.b64encode(b).decode("ascii") for _, b in valid]
        url = self._judge_urls[hash(wavs_b64[0][:64]) % len(self._judge_urls)]
        body = json.dumps({"text": text, "wavs_b64": wavs_b64, "debias": self._judge_debias}).encode()
        req = urllib.request.Request(url + "/rank", data=body, headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=self._judge_timeout_s).read())
            # strict: a score-count mismatch raises here and falls through to the neutral 0.0 below,
            # rather than silently mis-assigning candidates.
            for j, sc in zip(idxs, r["scores"], strict=True):
                scores[j] = float(sc)
        except Exception as e:  # noqa: BLE001
            logger.warning("audio judge call failed (%s); group gives no signal this step", e)
            for j in idxs:
                scores[j] = 0.0
        return scores

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())
        text = extra_info.get("text") or data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
        uid = extra_info.get("id") or data_item.non_tensor_batch.get("uid")

        my_blob = await self.loop.run_in_executor(
            self._score_executor, lambda: self._decode_wav_bytes(data_item, extra_info)
        )

        # Rendezvous this candidate with its uid siblings; whichever coroutine completes the group runs
        # the single direct-comparison judge call and releases scores to all members.
        async with self._group_lock:
            g = self._groups.get(uid)
            if g is None:
                g = {"text": text, "blobs": [], "scores": None, "event": asyncio.Event()}
                self._groups[uid] = g
            my_idx = len(g["blobs"])
            g["blobs"].append(my_blob)
            complete = len(g["blobs"]) >= self._group_size

        if complete:
            g["scores"] = await self.loop.run_in_executor(
                self._score_executor, self._judge_group, g["text"], g["blobs"]
            )
            # Drop the group so its wav bytes are freed; every sibling already holds its own g ref.
            self._groups.pop(uid, None)
            g["event"].set()
        else:
            try:
                await asyncio.wait_for(g["event"].wait(), timeout=self._judge_timeout_s)
            except asyncio.TimeoutError:  # a sibling never arrived (aborted rollout): judge what we have
                async with self._group_lock:
                    if g["scores"] is None:
                        g["scores"] = await self.loop.run_in_executor(
                            self._score_executor, self._judge_group, g["text"], g["blobs"]
                        )
                        g["event"].set()
                    self._groups.pop(uid, None)

        scores = g["scores"] or []
        my_score = float(scores[my_idx]) if my_idx < len(scores) else 0.0
        synth_ok = 1.0 if my_blob else 0.0
        return {"reward_score": my_score, "reward_extra_info": {"sj_score": my_score, "synth_ok": synth_ok}}
