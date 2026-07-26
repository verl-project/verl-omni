from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from tensordict.tensorclass import NonTensorStack
from verl.utils import tensordict_utils as tu

BOOGU_VAE_SCALE_FACTOR = 8


def apply_boogu_text_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    text_guidance_scale: float,
) -> torch.Tensor:
    return negative_noise_pred + text_guidance_scale * (noise_pred - negative_noise_pred)


def _get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _time_shift_v1(t_np: np.ndarray, mu: float, sigma: float = 1.0) -> np.ndarray:
    eps = 1e-8
    t1 = 1.0 - t_np
    t1 = np.clip(t1, eps, 1.0 - eps)
    num = math.exp(mu)
    denom = num + np.power(1.0 / t1 - 1.0, sigma)
    y = num / denom
    out = 1.0 - y
    return out.astype(np.float32)


def _time_shift_v2(t_np: np.ndarray, m: float) -> np.ndarray:
    return (t_np / (m - m * t_np + t_np)).astype(np.float32)


def build_boogu_sigmas(
    *,
    num_inference_steps: int,
    num_tokens: int,
    scheduler_config: Any,
) -> list[float]:
    t_arr = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]

    if not scheduler_config.get("do_shift", True):
        return t_arr.tolist()

    dynamic_time_shift = scheduler_config.get("dynamic_time_shift", True)
    time_shift_version = scheduler_config.get("time_shift_version", "v1")

    if dynamic_time_shift:
        if time_shift_version == "v1":
            tokens_reduced = max(1, int(num_tokens) // 4)
            lin = _get_lin_function(
                y1=scheduler_config.get("base_shift", 0.5),
                y2=scheduler_config.get("max_shift", 1.15),
            )
            mu = lin(tokens_reduced)
            t_arr = _time_shift_v1(t_arr, mu, sigma=1.0)
        elif time_shift_version == "v2":
            scaling_factor = scheduler_config.get("time_shift_v2_half_scaling_factor", 60.0) * 2
            m = float(np.sqrt(num_tokens)) / scaling_factor
            t_arr = _time_shift_v2(t_arr, m)
    else:
        seq_len = scheduler_config.get("seq_len")
        if seq_len is not None and seq_len > 0:
            if time_shift_version == "v1":
                lin = _get_lin_function(
                    y1=scheduler_config.get("base_shift", 0.5),
                    y2=scheduler_config.get("max_shift", 1.15),
                )
                mu = lin(int(seq_len))
                t_arr = _time_shift_v1(t_arr, mu, sigma=1.0)
            elif time_shift_version == "v2":
                scaling_factor = scheduler_config.get("time_shift_v2_half_scaling_factor", 60.0) * 2
                m = float(np.sqrt(seq_len)) / scaling_factor
                t_arr = _time_shift_v2(t_arr, m)

    return t_arr.tolist()


def build_boogu_freqs_cis(
    axes_dim: tuple[int, int, int],
    axes_lens: tuple[int, int, int],
    *,
    theta: int = 10000,
) -> list[torch.Tensor]:
    freqs_cis = []
    freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
    for dim, length in zip(axes_dim, axes_lens):
        freqs_cis.append(get_1d_rotary_pos_embed(dim, length, theta=theta, freqs_dtype=freqs_dtype))
    return freqs_cis


def _normalize_ref_sample(sample: Any) -> Optional[list[torch.Tensor]]:
    if sample is None:
        return None
    if isinstance(sample, torch.Tensor):
        if sample.ndim == 3:
            return [sample]
        if sample.ndim == 4:
            return [img for img in sample]
        raise ValueError(
            "condition_image_latents tensor per sample must have shape (C,H,W) or (N,C,H,W), "
            f"got {tuple(sample.shape)}."
        )
    if isinstance(sample, (list, tuple)):
        normalized: list[torch.Tensor] = []
        for item in sample:
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    "condition_image_latents list entries must be torch.Tensor instances; "
                    f"got {type(item)!r}."
                )
            if item.ndim != 3:
                raise ValueError(
                    "condition_image_latents nested tensor entries must have shape (C,H,W), "
                    f"got {tuple(item.shape)}."
                )
            normalized.append(item)
        return normalized
    raise TypeError(
        "condition_image_latents per-sample value must be None, Tensor, list, or tuple; "
        f"got {type(sample)!r}."
    )


def normalize_ref_image_hidden_states(value: Any) -> Optional[list[Optional[list[torch.Tensor]]]]:
    if value is None:
        return None
    if isinstance(value, NonTensorStack):
        value = [tu.unwrap_non_tensor_data(value[i]) for i in range(len(value))]
    else:
        value = tu.unwrap_non_tensor_data(value)

    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            return [[sample] for sample in value]
        if value.ndim == 5:
            return [[img for img in sample] for sample in value]
        raise ValueError(
            "condition_image_latents batch tensor must have shape (B,C,H,W) or (B,N,C,H,W), "
            f"got {tuple(value.shape)}."
        )

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "condition_image_latents batch value must be Tensor, list, tuple, or NonTensorStack; "
            f"got {type(value)!r}."
        )

    return [_normalize_ref_sample(sample) for sample in value]
