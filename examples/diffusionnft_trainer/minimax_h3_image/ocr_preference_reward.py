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
  until it fills or leaves the canvas; the preference terms do not see that.
- ``term_pickscore`` / ``term_hpsv2``: PickScore (logit / 26) and HPSv2.1 of the image against the full generation
  prompt. Both terms share one scorer call per image.

The full generation prompt is looked up by ``extra_info["split"]`` / ``extra_info["index"]`` in ``TRAIN_FILES`` and
``VAL_FILES`` (or taken from ``extra_info["prompt"]`` when the rows carry it).
"""

import asyncio
import json
import os
import sys
import tempfile
import uuid

from verl_omni.utils.reward_score.genrm_ocr import _to_pil, compute_score_ocr

HERE = os.path.dirname(os.path.abspath(__file__))
AREA_REF = float(os.environ.get("OCR_GUARD_AREA_REF", "0.08"))
POOL_SIZE = int(os.environ.get("OCR_GUARD_PROCS", "8"))
SCORER_GPU = os.environ.get("PREFERENCE_SCORER_GPU", "3")
TMP = os.path.join(os.environ.get("OCR_REWARD_TMPDIR") or tempfile.gettempdir(), "ocr_preference_reward")

_guard_pool = None
_guard_lock = None
_scorer = None
_scorer_lock = None
_prompts = None
_last = (None, None)


async def _spawn(script: str, env: dict):
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        os.path.join(HERE, script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=env,
    )
    line = await proc.stdout.readline()
    if not line or not json.loads(line).get("ready"):
        raise RuntimeError(f"{script} failed to start (exit code {await proc.wait()})")
    return proc


async def _request(proc, script: str, payload: dict) -> dict:
    proc.stdin.write((json.dumps(payload) + "\n").encode())
    await proc.stdin.drain()
    line = await proc.stdout.readline()
    if not line:
        raise RuntimeError(f"{script} exited (exit code {await proc.wait()})")
    return json.loads(line)


def _save(image) -> str:
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, f"{uuid.uuid4().hex}.png")
    _to_pil(image).save(path)
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


async def _layout(image) -> dict:
    global _guard_pool, _guard_lock
    if _guard_lock is None:
        _guard_lock = asyncio.Lock()
    async with _guard_lock:
        if _guard_pool is None:
            procs = await asyncio.gather(*[_spawn("ocr_layout_worker.py", dict(os.environ)) for _ in range(POOL_SIZE)])
            _guard_pool = asyncio.Queue()
            for proc in procs:
                _guard_pool.put_nowait(proc)
    path = _save(image)
    proc = await _guard_pool.get()
    try:
        return await _request(proc, "ocr_layout_worker.py", {"path": path})
    finally:
        _guard_pool.put_nowait(proc)
        _remove(path)


async def _preference(image, extra_info) -> dict:
    global _scorer, _scorer_lock, _last
    if _last[0] is image:
        return _last[1]
    if _scorer_lock is None:
        _scorer_lock = asyncio.Lock()
    path = _save(image)
    try:
        async with _scorer_lock:
            if _scorer is None:
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = SCORER_GPU
                _scorer = await _spawn("preference_worker.py", env)
            reply = await _request(_scorer, "preference_worker.py", {"path": path, "prompt": full_prompt(extra_info)})
    finally:
        _remove(path)
    if "error" in reply:
        raise RuntimeError(f"Preference scorer failed: {reply['error']}")
    _last = (image, reply)
    return reply


def guard_value(geometry: dict) -> float:
    if geometry["border"]:
        return 0.0
    return min(1.0, AREA_REF / geometry["area"]) if geometry["area"] > 0 else 1.0


async def term_ocr_guarded(data_source, solution_image, ground_truth, extra_info, **kwargs):
    result, geometry = await asyncio.gather(
        compute_score_ocr(data_source, solution_image, ground_truth, extra_info, **kwargs),
        _layout(solution_image),
    )
    ocr = float(result["score"])
    guard = guard_value(geometry)
    return {
        **result,
        "score": ocr * guard,
        "ocr_vlm": ocr,
        "guard": guard,
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
