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
  until it fills or leaves the canvas; the preference terms do not see that. A failed PP-OCR request leaves the
  guard at 1 (its identity) and reports ``guard_error=1``.
- ``term_pickscore`` / ``term_hpsv2``: PickScore (logit / 26) and HPSv2.1 of the image against the full generation
  prompt. Both terms share one scorer call per image.

Each reward-loop worker process owns its own PP-OCR pool (``OCR_GUARD_PROCS`` processes) and one scorer process on
``PREFERENCE_SCORER_GPU``. A child process that exits or gives no reply within ``REWARD_WORKER_TIMEOUT_S`` is killed,
its request raises, and a new process takes its place.

The full generation prompt is looked up by ``extra_info["split"]`` / ``extra_info["index"]`` in ``TRAIN_FILES`` and
``VAL_FILES`` (or taken from ``extra_info["prompt"]`` when the rows carry it).
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid

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

_guard_pool = None
_guard_lock = None
_scorer = None
_scorer_lock = None
_prompts = None
_last = (None, None)


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


def full_prompt(extra_info: dict) -> str:
    global _prompts
    if extra_info.get("prompt"):
        return extra_info["prompt"]
    if _prompts is None:
        import pyarrow.parquet as pq

        table = {}
        for key in ("TRAIN_FILES", "VAL_FILES"):
            for path in filter(None, os.environ.get(key, "").split(",")):
                for row in pq.read_table(path, columns=["prompt", "extra_info"]).to_pylist():
                    info = row["extra_info"]
                    table[(str(info["split"]), int(info["index"]))] = row["prompt"][-1]["content"]
        _prompts = table
    key = (str(extra_info["split"]), int(extra_info["index"]))
    if key not in _prompts:
        raise KeyError(f"No prompt for {key} in TRAIN_FILES / VAL_FILES.")
    return _prompts[key]


async def _layout(solution_image, extra_info: dict) -> dict:
    global _guard_pool, _guard_lock
    if _guard_lock is None:
        _guard_lock = asyncio.Lock()
    async with _guard_lock:
        if _guard_pool is None:
            procs = await asyncio.gather(
                *[_spawn("ocr_layout_worker.py", dict(os.environ)) for _ in range(POOL_SIZE)], return_exceptions=True
            )
            failed = [proc for proc in procs if isinstance(proc, BaseException)]
            if failed:
                for proc in procs:
                    if not isinstance(proc, BaseException):
                        await _kill(proc)
                raise failed[0]
            _guard_pool = asyncio.Queue()
            for proc in procs:
                _guard_pool.put_nowait(proc)
    path = _save(solution_image, extra_info, "Layout guard")
    proc = await _guard_pool.get()
    try:
        return await _request(proc, "ocr_layout_worker.py", {"path": path})
    finally:
        _remove(path)
        if proc.returncode is not None:
            try:
                proc = await _spawn("ocr_layout_worker.py", dict(os.environ))
            except Exception:
                logger.exception("Could not replace a dead PP-OCR layout worker")
                proc = None
        if proc is not None:
            _guard_pool.put_nowait(proc)


async def _preference(solution_image, extra_info: dict) -> dict:
    global _scorer, _scorer_lock, _last
    if _last[0] is solution_image:
        return _last[1]
    if _scorer_lock is None:
        _scorer_lock = asyncio.Lock()
    path = _save(solution_image, extra_info, "Preference scorer")
    try:
        async with _scorer_lock:
            if _scorer is None or _scorer.returncode is not None:
                env = dict(os.environ)
                if torch.version.hip:
                    # ROCm aborts on conflicting visibility; the inherited values name the trainer GPUs.
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                    env.pop("ROCR_VISIBLE_DEVICES", None)
                    env["HIP_VISIBLE_DEVICES"] = SCORER_GPU
                else:
                    env["CUDA_VISIBLE_DEVICES"] = SCORER_GPU
                _scorer = await _spawn("preference_worker.py", env)
            reply = await _request(_scorer, "preference_worker.py", {"path": path, "prompt": full_prompt(extra_info)})
    finally:
        _remove(path)
    if "error" in reply:
        raise RuntimeError(f"Preference scorer failed: {reply['error']}")
    _last = (solution_image, reply)
    return reply


def guard_value(geometry: dict) -> float:
    if geometry.get("error"):
        return 1.0
    if geometry["border"]:
        return 0.0
    return min(1.0, AREA_REF / geometry["area"]) if geometry["area"] > 0 else 1.0


async def term_ocr_guarded(data_source, solution_image, ground_truth, extra_info, **kwargs):
    result, geometry = await asyncio.gather(
        compute_score_ocr(data_source, solution_image, ground_truth, extra_info, **kwargs),
        _layout(solution_image, extra_info),
    )
    if geometry.get("error"):
        logger.warning("PP-OCR layout request failed, guard left at 1: %s", geometry["error"])
    ocr = float(result["score"])
    guard = guard_value(geometry)
    return {
        **result,
        "score": ocr * guard,
        "ocr_vlm": ocr,
        "guard": guard,
        "guard_error": float(bool(geometry.get("error"))),
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
