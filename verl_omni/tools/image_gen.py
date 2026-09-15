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

"""Frozen agentic function tools for verl's stock ``ToolAgentLoop``.

Module path: ``verl_omni/tools/image_gen.py``.
Bound automatically via ``OmniAgentLoopWorker`` (``function_tool_path``).

Provides ``generate_image`` + ``judge_image``. Agentic LLM RL keeps image
generation **outside** the actor optimizer. GRPO trains the actor as the
tool-calling agent while a frozen image-gen sidecar produces candidate PNGs.
A separate frozen VL sidecar (``judge_image``) scores those images; the actor
only sees text tool observations (scores / findings / ``path=``) and then
writes ``Reflection:`` / ``Done.`` or a rewritten ``generate_image``.

Config: Hydra ``agentic_image_gen`` (see
``verl_omni/trainer/config/agentic/image_gen_tools.yaml``) is the source of
truth. ``OmniAgentLoopWorker`` / ``OmniAgentLoopManager`` call
``bind_agentic_image_gen`` so FunctionTool bodies (``asyncio.to_thread``) read
the bound process-local knobs — no AGENTIC_* env.

Backends (first match wins):
  1. ``agentic_image_gen.vllm_omni_url`` — vLLM-Omni OpenAI image generations
     (``/v1/images/generations``).
  2. ``agentic_image_gen.qwen_image_url`` — bundled Qwen-Image HTTP service
     (POST ``{"prompt"}`` → base64 image JSON).
  3. ``agentic_image_gen.diffusion_tool_url`` — generic service with the same
     response contract.
  4. Else text-only stub (acceptance smoke when no gen service is up).

Observation modality is always text: PNGs are written under the rollout image
dir for the sidecar judge; they are never attached to the actor context.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image
from verl.tools.function_tool import function_tool
from verl.tools.schemas import ToolResponse

from verl_omni.tools.trajectory import (
    active_trajectory_relpath,
    active_user_prompt,
    build_artifact_id,
    count_live_generate_artifacts_for_active_rollout,
    get_active_rollout_id,
    get_good_enough_yes_reached,
    get_latest_generate_prompt_for_active_rollout,
    register_tool_artifact,
    resolve_rollout_images_root,
    resolve_tool_image_path,
    set_good_enough_yes_reached,
    set_latest_tool_image_path,
)
from verl_omni.tools.trajectory import paths as traj_paths
from verl_omni.tools.trajectory.hydra_env import (
    agentic_get,
    agentic_get_bool,
    agentic_get_float,
    agentic_get_int,
    agentic_get_str,
)
from verl_omni.utils.agentic.max_passes import max_generate_passes
from verl_omni.utils.agentic.vllm_chat import post_vllm_chat
from verl_omni.utils.agentic_image_judge_parse import (
    build_judge_prompt,
    format_judge_observation,
    format_judge_parse_error,
    parse_judge_json,
)

logger = logging.getLogger(__file__)


def _decode_images(payload: dict) -> list[Image.Image]:
    """Decode the endpoint's optional base64 image fields."""
    encoded = payload.get("images_base64")
    if encoded is None and payload.get("image_base64") is not None:
        encoded = [payload["image_base64"]]
    if not encoded:
        return []
    if not isinstance(encoded, list):
        logger.warning(
            "diffusion tool response 'images_base64' must be a list, got %s",
            type(encoded).__name__,
        )
        return []
    images: list[Image.Image] = []
    for item in encoded:
        try:
            images.append(Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB"))
        except Exception as exc:  # noqa: BLE001 — soft-fail external tool payloads
            logger.warning("failed to decode diffusion image payload: %s", exc)
    return images


def _next_call_dir(root: Path) -> Path:
    """Fallback when no trajectory is bound: ``call_<ts>_<uuid>/``."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    call_dir = root / f"call_{stamp}_{uuid.uuid4().hex[:10]}"
    call_dir.mkdir(parents=True, exist_ok=True)
    return call_dir


def _next_image_index(traj_dir: Path) -> int:
    """Next ``image_XX`` index under a trajectory folder."""
    idxs: list[int] = []
    for path in traj_dir.glob("image_*.png"):
        m = re.match(r"image_(\d+)", path.stem, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            idxs.append(int(m.group(1)))
        except ValueError:
            continue
    return (max(idxs) + 1) if idxs else 0


def _call_meta_fields(prompt: str, *, user_prompt: str) -> dict:
    """Default meta.json call-entry fields (reflection provenance is not bound)."""
    return {
        "call_role": "initial",
        "controlled_by_reflection": False,
        "reflection": "",
        "prev_tool_prompt": "",
        "source_image_for_reflection": "",
        "rewritten_prompt": "",
        "image_generated_from_reflected_prompt": False,
        "tool_prompt_equals_rewritten_prompt": False,
        "content_source": "initial",
        "llm_reflection": "",
        "llm_prompt": "",
        "model_decode": "",
        "user_prompt": user_prompt,
        "tool_prompt": prompt,
    }


def _update_traj_meta(traj_dir: Path, entry: dict) -> None:
    meta_path = traj_dir / "meta.json"
    meta: dict
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    else:
        meta = {}
    meta.setdefault("trajectory", traj_dir.name)
    meta.setdefault("experiment", traj_paths.run_name or "agentic_run")
    user_prompt = entry.get("user_prompt") or active_user_prompt.get() or ""
    if user_prompt:
        meta["user_prompt"] = user_prompt
        entry.setdefault("user_prompt", user_prompt)
    calls = list(meta.get("calls") or [])
    calls.append(entry)
    meta["calls"] = calls
    meta["num_images"] = len(calls)
    meta["reflection_controlled_image_files"] = [c.get("file") for c in calls if c.get("controlled_by_reflection")]
    meta["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def _image_call_entry(
    idx: int,
    path: Path,
    *,
    prompt: str,
    backend: str,
    stubbed: bool,
    provenance: dict,
) -> dict:
    """One ``meta.json`` call row for a saved generate_image artifact."""
    return {
        "index": idx,
        "file": path.name,
        "path": str(path),
        "prompt": prompt,
        "backend": backend,
        "tool_stubbed": stubbed,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **provenance,
    }


def _save_images(images: list[Image.Image], prompt: str, *, backend: str, tool_stubbed: bool) -> list[str]:
    """Persist tool images under the active stock-loop request path.

    When no trajectory is bound (e.g. standalone smoke), falls back to
    ``rollout_images/call_<ts>_<uuid>/``.
    """
    root = resolve_rollout_images_root()
    root.mkdir(parents=True, exist_ok=True)
    relpath = active_trajectory_relpath.get()
    user_prompt = active_user_prompt.get() or ""
    provenance = _call_meta_fields(prompt, user_prompt=user_prompt)
    paths: list[str] = []

    if relpath:
        traj_dir = root / relpath
        traj_dir.mkdir(parents=True, exist_ok=True)
        start_idx = _next_image_index(traj_dir)
        artifact_ids: list[str] = []
        if images:
            for offset, img in enumerate(images):
                idx = start_idx + offset
                aid = build_artifact_id(relpath=relpath, index=idx, prompt=prompt)
                path = traj_dir / f"image_{idx:02d}_{aid}.png"
                img.save(path)
                paths.append(str(path))
                artifact_ids.append(aid)
                meta_entry = _image_call_entry(
                    idx, path, prompt=prompt, backend=backend, stubbed=tool_stubbed, provenance=provenance
                )
                meta_entry["artifact_id"] = aid
                meta_entry["rollout_id"] = get_active_rollout_id()
                _update_traj_meta(traj_dir, meta_entry)
        else:
            aid = build_artifact_id(relpath=relpath, index=start_idx, prompt=prompt)
            stub_path = traj_dir / f"STUB_NO_IMAGE_{start_idx:02d}_{aid}.txt"
            stub_path.write_text(
                "No PNG produced (text stub or empty tool response).\n"
                f"user_prompt={user_prompt!r}\n"
                f"tool_prompt={prompt!r}\n"
                f"controlled_by_reflection={provenance.get('controlled_by_reflection')}\n"
                f"reflection={provenance.get('reflection')!r}\n"
                f"backend={backend}\n"
                f"artifact_id={aid}\n"
                "Set agentic_image_gen.qwen_image_url (or vllm_omni_url / diffusion_tool_url) "
                "for real images.\n"
            )
            paths.append(str(stub_path))
            artifact_ids.append(aid)
            meta_entry = _image_call_entry(
                start_idx, stub_path, prompt=prompt, backend=backend, stubbed=True, provenance=provenance
            )
            meta_entry["artifact_id"] = aid
            meta_entry["rollout_id"] = get_active_rollout_id()
            _update_traj_meta(traj_dir, meta_entry)
        logger.info(
            "diffusion tool artifacts (%d image(s), stub=%s, reflect_ctrl=%s) -> %s",
            len(images),
            tool_stubbed,
            provenance.get("controlled_by_reflection"),
            traj_dir,
        )
        register_tool_artifact(
            prompt=prompt,
            paths=paths,
            backend=backend,
            tool_stubbed=tool_stubbed,
            artifact_id=artifact_ids[0] if artifact_ids else None,
            trajectory_relpath=relpath,
            rollout_id=get_active_rollout_id(),
        )
        return paths

    # Legacy fallback (no active trajectory context).
    call_dir = _next_call_dir(root)
    meta = {
        **provenance,
        "backend": backend,
        "tool_stubbed": tool_stubbed,
        "num_images": len(images),
        "experiment": traj_paths.run_name or "agentic_run",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for i, img in enumerate(images):
        path = call_dir / f"image_{i:02d}.png"
        img.save(path)
        paths.append(str(path))
    if not images:
        stub_path = call_dir / "STUB_NO_IMAGE.txt"
        stub_path.write_text(
            "No PNG produced (text stub or empty tool response).\n"
            f"user_prompt={user_prompt!r}\n"
            f"tool_prompt={prompt!r}\n"
            f"backend={backend}\n"
            "Set agentic_image_gen.qwen_image_url (or vllm_omni_url / diffusion_tool_url) "
            "for real images.\n"
        )
        paths.append(str(stub_path))
    (call_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    logger.info(
        "diffusion tool artifacts (%d image(s), stub=%s) -> %s",
        len(images),
        tool_stubbed,
        call_dir,
    )
    register_tool_artifact(prompt=prompt, paths=paths, backend=backend, tool_stubbed=tool_stubbed)
    return paths


def _pack_response(
    prompt: str,
    text: str,
    images: list[Image.Image],
    reward: float,
    *,
    backend: str,
    tool_stubbed: bool,
) -> tuple[ToolResponse, float, dict]:
    paths = _save_images(images, prompt, backend=backend, tool_stubbed=tool_stubbed)
    metrics: dict = {
        "tool_stubbed": tool_stubbed,
        "diffusion_backend": backend,
        "image_paths": paths,
        "num_images": len(images),
        "prompt": prompt,
        "artifact_dir": str(Path(paths[0]).parent) if paths else "",
    }
    ok = 1 if (images and not tool_stubbed) else 0
    # Machine-readable markers for agentic_reward (R_tool / R_result) — survive
    # decode of the multi-turn response including tool-obs tokens.
    prompt_snip = (prompt or "").replace("\n", " ")[:240]
    png0 = next((p for p in paths if str(p).endswith(".png")), None)
    artifact_id = ""
    if png0:
        m = re.search(r"image_\d+_([0-9a-f]{12})\.png$", Path(png0).name, re.IGNORECASE)
        if m:
            artifact_id = m.group(1)
        set_latest_tool_image_path(png0)
    marker = (
        f"agentic_tool ok={ok} stub={1 if tool_stubbed else 0} images={len(images)} "
        f"backend={backend} prompt={prompt_snip!r}"
    )
    if artifact_id:
        marker = f"{marker} artifact={artifact_id}"
    rid = get_active_rollout_id()
    if rid:
        marker = f"{marker} rollout={rid}"
    if paths and "path=" not in text:
        text = f"{text} path={paths[0]}"
    text = f"{text} {marker}"
    # Always text-only to the actor; PNGs stay on disk for the sidecar judge.
    return ToolResponse(text=text), reward, metrics


def _call_generic_http(
    prompt: str,
    endpoint: str,
    *,
    backend: str = "http",
) -> tuple[ToolResponse, float, dict]:
    headers = {"Content-Type": "application/json"}
    token = agentic_get_str("diffusion_tool_token", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        endpoint,
        data=json.dumps({"prompt": prompt}).encode(),
        headers=headers,
        method="POST",
    )
    # Offloaded Qwen-Image requests may queue behind other rollout workers on
    # the single frozen-tool GPU.
    timeout = agentic_get_float("diffusion_tool_timeout", 900.0)
    try:
        with urlopen(request, timeout=timeout) as result:  # noqa: S310 - endpoint is operator-configured
            payload = json.loads(result.read())
    except Exception as exc:  # noqa: BLE001 - return failure as an observable tool result
        err = f"{backend} request failed: {exc}"
        logger.error(err)
        return _pack_response(
            prompt,
            err,
            images=[],
            reward=0.0,
            backend=f"{backend}_error",
            tool_stubbed=True,
        )

    images = _decode_images(payload)
    text = payload.get("text") or "The frozen diffusion tool generated the requested image."
    reward = float(payload.get("reward", 0.0))
    if not images:
        return _pack_response(
            prompt,
            text or f"{backend} returned no image",
            images=[],
            reward=0.0,
            backend=f"{backend}_empty",
            tool_stubbed=True,
        )
    return _pack_response(prompt, text, images, reward, backend=backend, tool_stubbed=False)


def _blocked_generate_after_yes(prompt: str) -> tuple[ToolResponse, float, dict]:
    """Hydra hard-stop: refuse generate_image after good_enough=YES (no diffusion call)."""
    prompt_snip = (prompt or "").replace("\n", " ")[:240]
    text = (
        "generate_image blocked: a prior judge_image already returned good_enough=YES. "
        "Emit Reflection summarizing the VL feedback and end with Done. — do not rewrite. "
        f"agentic_block_generate_after_yes=1 agentic_tool ok=0 stub=0 images=0 "
        f"backend=blocked_after_yes prompt={prompt_snip!r}"
    )
    metrics = {
        "tool_stubbed": False,
        "diffusion_backend": "blocked_after_yes",
        "image_paths": [],
        "num_images": 0,
        "prompt": prompt,
        "blocked_after_yes": 1,
    }
    logger.info("Blocked generate_image after good_enough=YES (prompt=%r)", prompt_snip[:120])
    return ToolResponse(text=text), 0.0, metrics


def _blocked_generate_after_max_passes(prompt: str, *, n_gen: int, max_passes: int) -> tuple[ToolResponse, float, dict]:
    """Hydra hard-stop: refuse further generate_image after the pass cap."""
    prompt_snip = (prompt or "").replace("\n", " ")[:240]
    text = (
        f"generate_image blocked: already completed {n_gen}/{max_passes} successful "
        "generate_image passes. Emit Reflection summarizing the latest VL feedback and "
        "end with Done. — do not rewrite again. "
        f"agentic_block_generate_after_max_passes=1 agentic_tool ok=0 stub=0 images=0 "
        f"backend=blocked_after_max_passes prompt={prompt_snip!r}"
    )
    metrics = {
        "tool_stubbed": False,
        "diffusion_backend": "blocked_after_max_passes",
        "image_paths": [],
        "num_images": 0,
        "prompt": prompt,
        "blocked_after_max_passes": 1,
        "generate_passes": n_gen,
        "max_generate_passes": max_passes,
    }
    logger.info(
        "Blocked generate_image after max passes (%d/%d, prompt=%r)",
        n_gen,
        max_passes,
        prompt_snip[:120],
    )
    return ToolResponse(text=text), 0.0, metrics


@function_tool("generate_image")
def generate_image(prompt: str) -> tuple[ToolResponse, float, dict]:
    """Generate an image with the frozen diffusion model.

    After each generation, call ``judge_image`` on that image before deciding
    Done. or rewrite.

    Args:
        prompt: The complete prompt to send to the diffusion model.

    Returns:
        ``(ToolResponse, reward, metrics)``. Reward is unused (frozen tool).
    """
    if agentic_get_bool("block_generate_after_yes", True) and get_good_enough_yes_reached():
        return _blocked_generate_after_yes(prompt)

    if agentic_get_bool("block_generate_after_max_passes", True):
        max_passes = max_generate_passes()
        n_gen = count_live_generate_artifacts_for_active_rollout()
        if n_gen >= max_passes:
            return _blocked_generate_after_max_passes(prompt, n_gen=n_gen, max_passes=max_passes)

    # vLLM-omni (continuous batching) — preferred.
    vllm_omni_url = agentic_get_str("vllm_omni_url")
    if vllm_omni_url:
        return _call_vllm_omni(prompt, vllm_omni_url)

    qwen_image_url = agentic_get_str("qwen_image_url")
    if qwen_image_url:
        return _call_generic_http(prompt, qwen_image_url, backend="qwen_image")

    endpoint = agentic_get_str("diffusion_tool_url")
    if endpoint:
        return _call_generic_http(prompt, endpoint)

    logger.warning("agentic_image_gen image URLs unset; using text-only stub diffusion tool (acceptance smoke only)")
    text = f"[stub diffusion result] No image service is configured. The requested prompt was: {prompt}"
    return _pack_response(prompt, text, images=[], reward=0.0, backend="stub", tool_stubbed=True)


def _require_hydra_int(key: str, default: int) -> int:
    raw = agentic_get(key, default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic_image_gen.{key} must be an integer, got {raw!r}") from exc


def _require_hydra_float(key: str, default: float) -> float:
    raw = agentic_get(key, default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic_image_gen.{key} must be a float, got {raw!r}") from exc


def _qwen_image_seed(prompt: str) -> int | None:
    raw = agentic_get("qwen_image_seed", None)
    if raw is None or raw == "":
        return None
    try:
        base_seed = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic_image_gen.qwen_image_seed must be an integer or null, got {raw!r}") from exc
    relpath = active_trajectory_relpath.get() or get_active_rollout_id()
    if not relpath or not agentic_get_bool("qwen_image_diversify_seed", True):
        return base_seed
    generate_pass = count_live_generate_artifacts_for_active_rollout()
    material = f"{relpath}|{generate_pass}|{prompt}".encode()
    offset = int.from_bytes(hashlib.blake2s(material, digest_size=4).digest(), "big")
    return (base_seed + offset) % (2**31)


def _call_vllm_omni(
    prompt: str,
    vllm_omni_url: str,
) -> tuple[ToolResponse, float, dict]:
    """Call vLLM-Omni's OpenAI-compatible image-generation endpoint."""
    base = vllm_omni_url.rstrip("/")
    height = _require_hydra_int("qwen_image_height", 512)
    width = _require_hydra_int("qwen_image_width", 512)
    steps = _require_hydra_int("qwen_image_steps", 20)
    cfg = _require_hydra_float("qwen_image_true_cfg_scale", 4.0)
    seed = _qwen_image_seed(prompt)

    payload: dict = {
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
        "num_inference_steps": steps,
        "true_cfg_scale": cfg,
    }
    if seed is not None:
        # Stable per-rollout/per-pass derivation preserves reproducibility while
        # giving GRPO non-identical candidates within each prompt group.
        payload["seed"] = seed

    headers = {"Content-Type": "application/json"}
    token = agentic_get_str("diffusion_tool_token", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = agentic_get_float("diffusion_tool_timeout", 900.0)
    try:
        req = Request(
            f"{base}/v1/images/generations",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=timeout) as result:  # noqa: S310
            data = json.loads(result.read().decode())
    except Exception as exc:  # noqa: BLE001
        err = f"vLLM-omni request failed: {exc}"
        logger.error(err)
        return _pack_response(prompt, err, images=[], reward=0.0, backend="vllm_omni_error", tool_stubbed=True)

    images: list[Image.Image] = []
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        b64_data = item.get("b64_json")
        if b64_data:
            images.append(Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB"))

    if not images:
        err = f"vLLM-omni returned no image for prompt={prompt!r}"
        logger.error("%s", err)
        return _pack_response(prompt, err, images=[], reward=0.0, backend="vllm_omni_empty", tool_stubbed=True)

    text = "vLLM-Omni generated the requested image."
    return _pack_response(prompt, text, images, 0.0, backend="vllm_omni", tool_stubbed=False)


def _expand_judge_user_request(user_request: str) -> str:
    """Expand compact / truncated judge args to the bound live user task."""
    potential_bound_holders = {
        "",
        "same",
        "same as user",
        "same as user message",
        "same as user task",
        "same as the user message",
        "same as the user task",
        "user",
        "user request",
        "user_request",
        "last",
        "previous",
    }
    raw = (user_request or "").strip()
    bound = (active_user_prompt.get() or "").strip()
    if not bound:
        return raw
    low = re.sub(r"\s+", " ", raw.lower()).rstrip(".")
    if low in potential_bound_holders:
        return bound
    # Truncated paste of the full task (common when response budget runs out).
    if len(raw) < max(80, int(0.55 * len(bound))) and bound.lower().startswith(raw[:48].lower()):
        return bound
    if len(raw) <= 120 and raw.lower() in bound.lower():
        return bound
    return raw


def _expand_judge_image_prompt(image_prompt: str) -> str:
    """Expand compact / truncated image_prompt to the latest live generate prompt."""
    potential_bound_holders = {"", "last", "latest", "same", "previous", "prior", "image_prompt"}
    raw = (image_prompt or "").strip()
    latest = (get_latest_generate_prompt_for_active_rollout() or "").strip()
    low = re.sub(r"\s+", " ", raw.lower()).rstrip(".")
    if low in potential_bound_holders:
        return latest or raw
    if latest and len(raw) < max(40, int(0.55 * len(latest))) and latest.lower().startswith(raw[:40].lower()):
        return latest
    return raw or latest


def _call_judge_vlm(
    user_request: str,
    image_prompt: str,
) -> tuple[str, dict]:
    """Call the frozen image-judge sidecar to judge the last generated image.

    Requires Hydra ``agentic_image_gen.vllm_url`` (OpenAI ``/v1/chat/completions``).

    Returns ``(text, meta)`` where *text* is formatted for the agent to read
    and *meta* carries per-dimension scores for logging.
    """
    user_request = _expand_judge_user_request(user_request)
    image_prompt = _expand_judge_image_prompt(image_prompt)
    vllm_url = agentic_get_str("vllm_url")
    if not vllm_url:
        return (
            "[judge stub] agentic_image_gen.vllm_url unset — cannot score the image. "
            "Start run_judge_image_tool_server.sh and set agentic_image_gen.vllm_url.",
            {"stub": True},
        )
    return _call_judge_vllm(user_request, image_prompt, vllm_url)


# ── vLLM judge path (OpenAI /v1/chat/completions, continuous batching) ──────


def _call_judge_vllm(
    user_request: str,
    image_prompt: str,
    vllm_url: str,
) -> tuple[str, dict]:
    """Judge via vLLM's OpenAI-compatible ``/v1/chat/completions`` with parse retry."""
    image_path = resolve_tool_image_path(image_prompt=image_prompt)
    if not image_path:
        msg = (
            "[judge error] no image on disk for this generate_image call "
            f"(image_prompt={image_prompt[:120]!r}). "
            "Refusing to call the VL sidecar without pixels."
        )
        logger.warning("judge_image aborted: missing image path (prompt=%r)", image_prompt[:160])
        return msg, {"error": "missing_image_path", "parse_ok": 0}

    try:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    except OSError as exc:
        msg = f"[judge error] cannot read image at {image_path}: {exc}"
        logger.error("%s", msg)
        return msg, {"error": str(exc), "image_path": image_path, "parse_ok": 0}

    base_tokens = agentic_get_int("reflect_max_new_tokens", 1024)
    max_retries = max(0, agentic_get_int("judge_parse_retries", 1))

    last_raw = ""
    last_err = None
    for attempt in range(max_retries + 1):
        strict = attempt > 0
        prompt_text = build_judge_prompt(user_request, image_prompt, strict_json=strict)
        # Give truncated JSON more room on retry.
        tokens = base_tokens if attempt == 0 else max(base_tokens, 1536)
        raw_text, err = post_vllm_chat(
            vllm_url=vllm_url,
            image_b64=image_b64,
            prompt_text=prompt_text,
            max_tokens=tokens,
            model=agentic_get_str("vllm_model"),
            timeout=agentic_get_float("reflect_vlm_timeout"),
            enable_thinking=agentic_get_bool("judge_enable_thinking"),
        )
        if err is not None:
            last_err = err
            logger.warning("vLLM judge call failed (attempt=%d): %s", attempt, err)
            continue
        assert raw_text is not None
        last_raw = raw_text
        parsed = parse_judge_json(raw_text)
        if parsed is not None:
            return format_judge_observation(
                image_path=image_path,
                parsed=parsed,
                backend="vllm",
                parse_retries=attempt,
            )
        logger.warning(
            "vLLM judge unparseable (attempt=%d/%d): %.200s",
            attempt,
            max_retries,
            raw_text,
        )

    if last_err and not last_raw:
        return (
            f"[judge error] vLLM request failed ({last_err}). "
            "Retry judge_image or rewrite the diffusion prompt and generate again.\n"
            f"  path={image_path}\n"
            f"  agentic_judge ok=0 parse_ok=0 stub=0 backend=vllm",
            {"error": last_err, "image_path": image_path, "parse_ok": 0, "parse_retries": max_retries},
        )
    return format_judge_parse_error(
        image_path=image_path,
        raw_text=last_raw,
        backend="vllm",
        parse_retries=max_retries,
    )


@function_tool("judge_image")
def judge_image(user_request: str, image_prompt: str) -> tuple[ToolResponse, float, dict]:
    """Judge the last generated image with a frozen vision model.

    Returns structured feedback: correctness/aesthetics scores, findings,
    suggested prompt fixes, and a good_enough verdict. Call this after every
    ``generate_image``. Keep arguments short: prefer
    ``user_request='same as user message'`` and ``image_prompt='last'``.

    Args:
        user_request: Compact task tag only. Prefer exactly ``same as user
            message``; do not paste the full user task. The server expands this
            to the bound user request.
        image_prompt: Prefer exactly ``last``, or a short echo of the diffusion
            prompt. Do not re-paste long prompts.

    Returns:
        ``(ToolResponse, reward, metrics)``. Reward is unused (frozen tool).
    """
    text, meta = _call_judge_vlm(user_request, image_prompt)
    # Env hard-stop latch: after YES, later generate_image calls are refused.
    if meta.get("good_enough") and meta.get("parse_ok", 1) != 0 and not meta.get("error"):
        set_good_enough_yes_reached(True)
    metrics = {
        "tool": "judge_image",
        "judge_stub": meta.get("stub", False),
        "judge_error": meta.get("error", ""),
    }
    for key in (
        "correctness",
        "aesthetics",
        "good_enough",
        "findings",
        "suggested_fixes",
    ):
        if key in meta:
            metrics[f"judge_{key}"] = meta[key]
    return ToolResponse(text=text), 0.0, metrics
