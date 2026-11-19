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
"""Reward terms for ``MultiVisualRewardManager`` with ``reward.aggregation=weighted_sum``.

The recipe weights them ``0.5 * term_pickscore + 0.2 * term_ocr_guarded + 0.3 * term_hpsv2``.

- ``term_ocr_guarded``: VLM OCR score (``genrm_ocr``) times a layout guard that is 0 when any PP-OCR text box touches
  the frame and ``min(1, OCR_GUARD_AREA_REF / text_area)`` otherwise. Without the guard the policy enlarges the text
  until it fills or leaves the canvas; the preference terms do not see that. A failed PP-OCR request raises
  ``RuntimeError`` so the OCR term contributes 0 and records ``reward/ocr/errors=1`` (or fails the step when
  ``required=true``) rather than leaving the guard at 1.
- ``term_pickscore`` / ``term_hpsv2``: PickScore (logit / 26) and HPSv2.1 of the image against the full generation
  prompt. Both terms share one scorer process and cache per reward-loop worker via a ``sys.modules`` state object.

Each reward-loop worker process owns its own PP-OCR pool (``OCR_GUARD_PROCS`` processes) and one scorer process on
``PREFERENCE_SCORER_GPU``. A child process that exits or gives no reply within ``REWARD_WORKER_TIMEOUT_S`` is killed,
its request raises, and a new process takes its place.

The full generation prompt is looked up by ``(extra_info.get("tier"), extra_info["split"], extra_info["index"])`` in
``TRAIN_FILES`` and ``VAL_FILES`` (or taken from ``extra_info["prompt"]`` when the rows carry it).
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from types import SimpleNamespace

import torch

from verl_omni.utils.reward_score.genrm_ocr import compute_score_ocr
from verl_omni.utils.reward_score.image_preference_utils import decoded_reward_image

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
AREA_REF = float(os.environ.get("OCR_GUARD_AREA_REF", "0.08"))
POOL_SIZE = int(os.environ.get("OCR_GUARD_PROCS", "8"))
SCORER_GPU = os.environ.get("PREFERENCE_SCORER_GPU", "3")
TIMEOUT_S = float(os.environ.get("REWARD_WORKER_TIMEOUT_S", "300"))
TMP = os.path.join(os.environ.get("OCR_REWARD_TMPDIR") or tempfile.gettempdir(), "ocr_preference_reward")

# load_extern_object executes this file once per registered reward name without caching in sys.modules.
# Keep process/cache state in a single sys.modules sentinel so term_pickscore and term_hpsv2 share one scorer.
_STATE_KEY = f"_verl_omni_minimax_h3_ocr_preference_state:{os.path.abspath(__file__)}"


def _new_state() -> SimpleNamespace:
    return SimpleNamespace(
        loop=None,
        guard_pool=None,
        guard_lock=None,
        scorer=None,
        scorer_lock=None,
        prompts=None,
        last=(None, None, None),
    )


def _get_state() -> SimpleNamespace:
    state = sys.modules.get(_STATE_KEY)
    if state is None:
        state = _new_state()
        sys.modules[_STATE_KEY] = state
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and state.loop is not None and state.loop is not loop:
        state.guard_pool = None
        state.guard_lock = None
        state.scorer = None
        state.scorer_lock = None
    if loop is not None:
        state.loop = loop
    return state


async def _kill(proc) -> None:
    if proc.returncode is None:
        proc.kill()
    await proc.wait()


async def _spawn(script: str, env: dict):
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        os.path.join(HERE, script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        line = await asyncio.wait_for(proc.stdout.readline(), TIMEOUT_S)
    except TimeoutError:
        await _kill(proc)
        raise RuntimeError(f"{script} did not start within {TIMEOUT_S:g} s") from None
    if not line or not json.loads(line).get("ready"):
        await _kill(proc)
        raise RuntimeError(f"{script} failed to start (exit code {proc.returncode})")
    return proc


async def _request(proc, script: str, payload: dict) -> dict:
    """Send one request; a worker that exits or hangs is killed and the request raises."""
    try:
        proc.stdin.write((json.dumps(payload) + "\n").encode())
        await proc.stdin.drain()
        line = await asyncio.wait_for(proc.stdout.readline(), TIMEOUT_S)
    except TimeoutError:
        await _kill(proc)
        raise RuntimeError(f"{script} gave no reply within {TIMEOUT_S:g} s") from None
    except (BrokenPipeError, ConnectionResetError):
        line = b""
    if not line:
        await _kill(proc)
        raise RuntimeError(f"{script} exited (exit code {proc.returncode})")
    return json.loads(line)


def _save(solution_image, extra_info: dict, scorer: str) -> str:
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, f"{uuid.uuid4().hex}.png")
    decoded_reward_image(solution_image, extra_info, scorer=scorer).save(path)
    return path


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _prompt_key(info: dict) -> tuple[str | None, str, int]:
    tier = info.get("tier")
    return (None if tier is None else str(tier), str(info["split"]), int(info["index"]))


def full_prompt(extra_info: dict) -> str:
    state = _get_state()
    if extra_info.get("prompt"):
        return extra_info["prompt"]
    if state.prompts is None:
        import pyarrow.parquet as pq

        table = {}
        for key in ("TRAIN_FILES", "VAL_FILES"):
            for path in filter(None, os.environ.get(key, "").split(",")):
                for row in pq.read_table(path, columns=["prompt", "extra_info"]).to_pylist():
                    row_key = _prompt_key(row["extra_info"])
                    if row_key in table:
                        raise ValueError(f"Duplicate prompt key {row_key} in TRAIN_FILES / VAL_FILES.")
                    table[row_key] = row["prompt"][-1]["content"]
        state.prompts = table
    key = _prompt_key(extra_info)
    if key not in state.prompts:
        raise KeyError(f"No prompt for {key} in TRAIN_FILES / VAL_FILES.")
    return state.prompts[key]


async def _layout(solution_image, extra_info: dict) -> dict:
    state = _get_state()
    if state.guard_lock is None:
        state.guard_lock = asyncio.Lock()
    async with state.guard_lock:
        if state.guard_pool is None:
            procs = await asyncio.gather(
                *[_spawn("ocr_layout_worker.py", dict(os.environ)) for _ in range(POOL_SIZE)], return_exceptions=True
            )
            failed = [proc for proc in procs if isinstance(proc, BaseException)]
            if failed:
                for proc in procs:
                    if not isinstance(proc, BaseException):
                        await _kill(proc)
                raise failed[0]
            state.guard_pool = asyncio.Queue()
            for proc in procs:
                state.guard_pool.put_nowait(proc)
    path = _save(solution_image, extra_info, "Layout guard")
    try:
        proc = await asyncio.wait_for(state.guard_pool.get(), TIMEOUT_S)
    except TimeoutError:
        _remove(path)
        raise RuntimeError(f"ocr_layout_worker.py pool had no idle worker within {TIMEOUT_S:g} s") from None
    try:
        if proc is None or proc.returncode is not None:
            proc = await _spawn("ocr_layout_worker.py", dict(os.environ))
        return await _request(proc, "ocr_layout_worker.py", {"path": path})
    finally:
        _remove(path)
        if proc is not None and proc.returncode is not None:
            try:
                proc = await _spawn("ocr_layout_worker.py", dict(os.environ))
            except Exception:
                logger.exception("Could not replace a dead PP-OCR layout worker")
                proc = None
        state.guard_pool.put_nowait(proc)


async def _preference(solution_image, extra_info: dict) -> dict:
    state = _get_state()
    prompt = full_prompt(extra_info)
    if state.last[0] is solution_image and state.last[1] == prompt:
        return state.last[2]
    if state.scorer_lock is None:
        state.scorer_lock = asyncio.Lock()
    async with state.scorer_lock:
        if state.last[0] is solution_image and state.last[1] == prompt:
            return state.last[2]
        path = _save(solution_image, extra_info, "Preference scorer")
        try:
            if state.scorer is None or state.scorer.returncode is not None:
                env = dict(os.environ)
                if env.get("H3_IMAGE_DEVICE") == "npu":
                    env["ASCEND_RT_VISIBLE_DEVICES"] = SCORER_GPU
                elif torch.version.hip:
                    # ROCm aborts on conflicting visibility; the inherited values name the trainer GPUs.
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                    env.pop("ROCR_VISIBLE_DEVICES", None)
                    env["HIP_VISIBLE_DEVICES"] = SCORER_GPU
                else:
                    env["CUDA_VISIBLE_DEVICES"] = SCORER_GPU
                state.scorer = await _spawn("preference_worker.py", env)
            reply = await _request(state.scorer, "preference_worker.py", {"path": path, "prompt": prompt})
        finally:
            _remove(path)
        if "error" in reply:
            raise RuntimeError(f"Preference scorer failed: {reply['error']}")
        state.last = (solution_image, prompt, reply)
        return reply


def guard_value(geometry: dict) -> float:
    if geometry.get("error"):
        raise RuntimeError(f"PP-OCR layout worker failed: {geometry['error']}")
    if geometry["border"]:
        return 0.0
    return min(1.0, AREA_REF / geometry["area"]) if geometry["area"] > 0 else 1.0


async def term_ocr_guarded(data_source, solution_image, ground_truth, extra_info, **kwargs):
    result, geometry = await asyncio.gather(
        compute_score_ocr(data_source, solution_image, ground_truth, extra_info, **kwargs),
        _layout(solution_image, extra_info),
    )
    guard = guard_value(geometry)
    ocr = float(result["score"])
    return {
        **result,
        "score": ocr * guard,
        "ocr_vlm": ocr,
        "guard": guard,
        "guard_error": 0.0,
        "text_area": float(geometry["area"]),
        "border_touch": float(geometry["border"]),
        "line_height": float(geometry["line_h"]),
    }


async def term_pickscore(data_source, solution_image, ground_truth, extra_info, **kwargs):
    reply = await _preference(solution_image, extra_info)
    return {"score": float(reply["pickscore"])}


async def term_hpsv2(data_source, solution_image, ground_truth, extra_info, **kwargs):
    reply = await _preference(solution_image, extra_info)
    return {"score": float(reply["hpsv2"])}
