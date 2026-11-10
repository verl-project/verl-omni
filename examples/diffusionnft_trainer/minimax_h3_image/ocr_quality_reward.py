"""OCR reward with a layout guard and an HPSv2.1 quality gate.

``score = ocr * guard * gate``

- ``ocr``: VLM transcription score from ``verl_omni.utils.reward_score.genrm_ocr``.
- ``guard``: 0 when any PP-OCR text box touches the frame, otherwise ``min(1, OCR_GUARD_AREA_REF / text_area)``.
  It removes the reward for enlarging the text until it fills or leaves the canvas.
- ``gate``: ``min(1, hpsv2(image, full prompt) / QUALITY_GATE_BASE)``. ``QUALITY_GATE_BASE`` is the mean HPSv2.1 of the
  untrained model on the validation prompts. It removes the reward for washing out the scene behind the text.

The full generation prompt is read from ``extra_info["prompt"]`` (written by ``prepare_dense_text_data.py``).
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
GATE_BASE = float(os.environ.get("QUALITY_GATE_BASE", "0.1971"))
GATE_GPU = os.environ.get("QUALITY_GATE_GPU", "")
TMP = os.environ.get("OCR_REWARD_TMPDIR") or tempfile.mkdtemp(prefix="ocr_reward_")

_guard_pool = None
_guard_lock = None
_gate_proc = None
_gate_lock = None


async def _spawn(script: str, env: dict):
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        os.path.join(HERE, script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    ready = json.loads(await proc.stdout.readline())
    if not ready.get("ready"):
        raise RuntimeError(f"{script} failed to start")
    return proc


async def _request(proc, payload: dict) -> dict:
    proc.stdin.write((json.dumps(payload) + "\n").encode())
    await proc.stdin.drain()
    return json.loads(await proc.stdout.readline())


def _save(image) -> str:
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, f"{uuid.uuid4().hex}.png")
    _to_pil(image).save(path)
    return path


async def _layout(path: str) -> dict:
    global _guard_pool, _guard_lock
    if _guard_lock is None:
        _guard_lock = asyncio.Lock()
    async with _guard_lock:
        if _guard_pool is None:
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            procs = await asyncio.gather(*[_spawn("ocr_layout_worker.py", env) for _ in range(POOL_SIZE)])
            _guard_pool = asyncio.Queue()
            for proc in procs:
                _guard_pool.put_nowait(proc)
    proc = await _guard_pool.get()
    try:
        return await _request(proc, {"path": path})
    finally:
        _guard_pool.put_nowait(proc)


async def _hpsv2(path: str, prompt: str) -> float:
    global _gate_proc, _gate_lock
    if _gate_lock is None:
        _gate_lock = asyncio.Lock()
    async with _gate_lock:
        if _gate_proc is None:
            env = dict(os.environ)
            if GATE_GPU:
                env["CUDA_VISIBLE_DEVICES"] = GATE_GPU
            _gate_proc = await _spawn("hpsv2_worker.py", env)
        reply = await _request(_gate_proc, {"path": path, "prompt": prompt})
    if "error" in reply:
        raise RuntimeError(f"HPSv2.1 scorer failed: {reply['error']}")
    return float(reply["hpsv2"])


def guard_value(geometry: dict) -> float:
    if geometry["border"]:
        return 0.0
    return min(1.0, AREA_REF / geometry["area"]) if geometry["area"] > 0 else 1.0


async def compute_score(data_source, solution_image, ground_truth, extra_info, *args, **kwargs):
    path = _save(solution_image)
    try:
        result, geometry, hpsv2 = await asyncio.gather(
            compute_score_ocr(data_source, solution_image, ground_truth, extra_info, *args, **kwargs),
            _layout(path),
            _hpsv2(path, extra_info["prompt"]),
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    ocr = float(result["score"])
    guard = guard_value(geometry)
    gate = max(0.0, min(1.0, hpsv2 / GATE_BASE))
    return {
        **result,
        "score": ocr * guard * gate,
        "ocr_vlm": ocr,
        "guard": guard,
        "gate": gate,
        "hpsv2": hpsv2,
        "text_area": float(geometry["area"]),
        "border_touch": float(geometry["border"]),
        "line_height": float(geometry["line_h"]),
    }
