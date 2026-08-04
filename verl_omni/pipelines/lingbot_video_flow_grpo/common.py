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

"""Small dependency-free helpers shared by the LingBot dense adapters."""

from __future__ import annotations

import ctypes
import importlib
import json
import logging
import site
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FA3_RUNTIME_DISABLED_REASON: str | None = None

# Kept in sync with Robbyant/lingbot-video at a638721.  Keeping the template
# here lets the dataset-side agent loop run before the optional model package is
# imported by a rollout worker.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative":{"visual_quality":["low quality","worst quality","blurry",'
    '"pixelated","jpeg artifacts","low resolution","unstable color","color flicker",'
    '"underexposed","overexposed","invisible subject","subject hidden in darkness"],'
    '"artistic_style":["painting","illustration","drawing","cartoon","3d render",'
    '"cgi","sketch","digital art"],"composition_and_content":["text","watermark",'
    '"signature","logo","subtitles","pillarboxed","side bars","portrait image in landscape frame"],'
    '"temporal_and_motion_stability":["flickering","jittery","motion blur",'
    '"temporal inconsistency","warping","morphing","incoherent motion","unnatural movement",'
    '"static object with sudden jump","frame-to-frame inconsistency"],"material_and_structure":'
    '["plastic-like glass","unrealistic texture","deformed bottle","liquid freezing improperly",'
    '"distorted reflections"]}}'
)


def install_flash_attn_interface_compat(*, prefer_fa3: bool = True) -> bool:
    """Expose LingBot's expected varlen attention entrypoint.

    Upstream ``lingbot-video`` imports
    ``flash_attn_interface.flash_attn_varlen_func`` for packed-batch attention
    (used when diffusion micro-batch size is greater than 1).  Some
    environments provide the compatible symbol from ``fa3_fwd_interface`` rather
    than the legacy module, and the FA3 wheel can import successfully while its
    first CUDA call fails because the wheel bundles a newer CUDA runtime than the
    host driver supports.

    Install an in-process shim so the optional LingBot dependency stays
    unmodified.  The shim tries the real FA3 implementation when requested, but
    falls back to a differentiable PyTorch SDPA varlen implementation if FA3 is
    absent or hits the known CUDA runtime/driver mismatch at execution time.
    """

    backend_func = None
    backend_file = None

    if prefer_fa3:
        existing = sys.modules.get("flash_attn_interface")
        if existing is not None and hasattr(existing, "flash_attn_varlen_func"):
            backend_func = existing.flash_attn_varlen_func
            backend_file = getattr(existing, "__file__", None)

        if backend_func is None:
            try:
                module = importlib.import_module("flash_attn_interface")
            except Exception:
                module = None
            if module is not None and hasattr(module, "flash_attn_varlen_func"):
                backend_func = module.flash_attn_varlen_func
                backend_file = getattr(module, "__file__", None)

        if backend_func is None:
            _preload_bundled_cuda13_runtime()
            try:
                fa3_module = importlib.import_module("fa3_fwd_interface")
                backend_func = fa3_module.flash_attn_varlen_func
                backend_file = getattr(fa3_module, "__file__", None)
            except Exception:
                backend_func = None

    shim = types.ModuleType("flash_attn_interface")
    shim.flash_attn_varlen_func = _make_flash_attn_varlen_func(backend_func)
    shim.__file__ = backend_file
    shim.__doc__ = "verl-omni LingBot flash_attn_interface compatibility shim."
    shim.__verl_omni_fa3_shim__ = True
    shim.__verl_omni_fa3_backend__ = backend_file if backend_func is not None else None
    sys.modules["flash_attn_interface"] = shim

    # If LingBot was imported before the shim was installed, patch its cached
    # module global too.  The normal training path calls this before importing
    # LingBot, but this keeps interactive/debug imports recoverable.
    transformer_module = sys.modules.get("lingbot_video.transformer_lingbot_video")
    if transformer_module is not None:
        transformer_module.flash_attn_varlen_func_v3 = shim.flash_attn_varlen_func
    return True


def _make_flash_attn_varlen_func(backend_func):
    def _compat_flash_attn_varlen_func(*args, **kwargs):
        global _FA3_RUNTIME_DISABLED_REASON
        if backend_func is not None and _FA3_RUNTIME_DISABLED_REASON is None:
            try:
                return backend_func(*args, **kwargs)
            except RuntimeError as exc:
                if not _is_known_fa3_runtime_mismatch(exc):
                    raise
                _FA3_RUNTIME_DISABLED_REASON = str(exc)
                logger.warning(
                    "LingBot FA3 varlen attention failed at runtime (%s); "
                    "falling back to PyTorch SDPA varlen attention in this process.",
                    _FA3_RUNTIME_DISABLED_REASON,
                )
        return _torch_flash_attn_varlen_fallback(*args, **kwargs)

    _compat_flash_attn_varlen_func.__name__ = "flash_attn_varlen_func"
    _compat_flash_attn_varlen_func.__doc__ = "FA3-backed LingBot varlen attention with PyTorch SDPA fallback."
    _compat_flash_attn_varlen_func.__verl_omni_fa3_backend__ = backend_func is not None
    return _compat_flash_attn_varlen_func


def _is_known_fa3_runtime_mismatch(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "CUDA driver version is insufficient for CUDA runtime version" in message
        or "cudaGetDeviceProperties failed" in message
    )


def _torch_flash_attn_varlen_fallback(*args, **kwargs) -> torch.Tensor:
    """Small SDPA fallback for LingBot's packed self-attention path.

    It implements the subset of ``flash_attn_varlen_func`` used by
    ``lingbot_video.transformer_lingbot_video``: flattened ``(tokens, heads,
    head_dim)`` q/k/v tensors with cumulative sequence lengths and no attention
    windowing.  The implementation is slower than FA3 but differentiable and
    keeps ``micro_batch_size > 1`` functional when the FA3 wheel is ABI/runtime
    incompatible with the host.
    """

    names = (
        "q",
        "k",
        "v",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
    )
    params = {name: value for name, value in zip(names, args, strict=False)}
    params.update(kwargs)

    q = params["q"]
    k = params["k"]
    v = params["v"]
    cu_seqlens_q = params["cu_seqlens_q"]
    cu_seqlens_k = params["cu_seqlens_k"]
    causal = bool(params.get("causal", False))
    dropout_p = float(params.get("dropout_p", 0.0) or 0.0)
    softmax_scale = params.get("softmax_scale", None)
    window_size = params.get("window_size", None)

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "LingBot SDPA varlen fallback expects q/k/v shaped "
            f"(tokens, heads, head_dim), got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}."
        )
    if window_size not in (None, (-1, -1)):
        raise NotImplementedError("LingBot SDPA varlen fallback does not support local attention windows.")
    if params.get("return_attn_probs", False):
        raise NotImplementedError("LingBot SDPA varlen fallback does not return attention probabilities.")

    outputs: list[torch.Tensor] = []
    num_sequences = int(cu_seqlens_q.numel()) - 1
    for index in range(num_sequences):
        q_start = int(cu_seqlens_q[index].item())
        q_end = int(cu_seqlens_q[index + 1].item())
        k_start = int(cu_seqlens_k[index].item())
        k_end = int(cu_seqlens_k[index + 1].item())
        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0).contiguous()


def _preload_bundled_cuda13_runtime() -> None:
    """Preload the CUDA 13 runtime wheel used by ``fa3_fwd_interface`` if present."""

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for site_packages in site.getsitepackages():
        libcudart = Path(site_packages) / "nvidia" / "cu13" / "lib" / "libcudart.so.13"
        if libcudart.exists():
            try:
                ctypes.CDLL(str(libcudart), mode=mode)
            except OSError:
                pass


def caption_to_json(caption: Any) -> str:
    """Serialize a structured LingBot caption with the official compact JSON form.

    The model is trained on structured captions.  Rejecting plain text here is
    intentional: silently accepting it would make an incorrectly preprocessed
    dataset look like a valid LingBot rollout.
    """

    if isinstance(caption, str):
        try:
            caption = json.loads(caption)
        except json.JSONDecodeError as exc:
            raise ValueError("LingBot T2V prompts must be valid structured JSON.") from exc
    if not isinstance(caption, dict | list):
        raise TypeError(f"LingBot T2V prompts must be a mapping, list, or JSON string; got {type(caption).__name__}.")
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def apply_prompt_template(caption: str) -> str:
    """Apply the official LingBot text-only T2V template."""

    return PROMPT_TEMPLATE.format(caption)


def validate_t2v_dimensions(height: int, width: int, num_frames: int) -> None:
    """Validate the official Dense T2V spatial and temporal constraints."""

    if num_frames != 1 and (num_frames - 1) % 4 != 0:
        raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"`height` and `width` must be multiples of 16, got {height}x{width}.")


def shifted_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
    """Return the LingBot/FlowMatch shifted sigma schedule without terminal zero."""

    if num_inference_steps <= 0:
        raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
    if shift <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}.")
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    return sigmas[:-1].cpu().numpy()


def apply_cfg(noise_pred: torch.Tensor, negative_noise_pred: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Apply LingBot's standard classifier-free guidance rule."""

    return negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)


def guidance_scale(pipeline_config: Any) -> float:
    """Resolve the user-facing LingBot guidance value from shared pipeline config."""

    value = getattr(pipeline_config, "guidance_scale", None)
    if value is None and hasattr(pipeline_config, "get"):
        value = pipeline_config.get("guidance_scale", None)
    if value is None:
        value = getattr(pipeline_config, "true_cfg_scale", 1.0)
        if hasattr(pipeline_config, "get"):
            value = pipeline_config.get("true_cfg_scale", value)
    return float(value)
