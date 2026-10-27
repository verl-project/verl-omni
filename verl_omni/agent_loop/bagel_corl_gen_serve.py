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

"""Live GEN serving helpers: FlowGRPO sampling params + traj stash from vLLM-Omni."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_AR_ONLY_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "min_p",
        "n",
    }
)

# The values an AR request carries when nobody thought about this lane: vLLM's untruncated decoding defaults, plus the
# ``repetition_penalty=1.0`` that ``verl``'s ``AgentLoopWorkerTQ.generate_sequences`` *hardcodes*  sampling_params =
# dict(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, repetition_penalty=1.0, ...)  and never
# reads ``actor_rollout_ref.rollout.repetition_penalty`` from.
UNTRAINED_AR_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
}

# Minimally-mitigated UND sampling, used only for the knobs above that are unset *and* absent from the config.
UND_AR_SAMPLING_FALLBACK: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
}


# Where the AR knobs can live, most specific first. ``generate`` is handed the *trainer*
# config, and the values that actually decide the UND decode are the ones the AR engine was
# launched with -- the recipe's ``actor_rollout_ref.rollout`` section, which
# ``_build_und_ar_entrypoint_config`` copies into the replica config. The remaining entries
# accept a caller that passes just the rollout section, or a bare mapping.
_AR_SAMPLING_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("actor_rollout_ref", "rollout"),
    ("rollout",),
    (),
)


def _read_knob(src: Any, name: str) -> Any:
    """Read one key off a DictConfig / plain dict / namespace, or ``None`` when absent."""
    if src is None:
        return None
    if isinstance(src, dict):
        return src.get(name)
    value = getattr(src, name, None)
    if value is None and hasattr(src, "get"):
        try:
            return src.get(name)
        except Exception:  # noqa: BLE001 -- a config probe must never fail a rollout
            return None
    return value


def config_sampling_value(config: Any, name: str) -> Any:
    """Read one sampling knob off the rollout config, whatever shape the caller passed."""
    for path in _AR_SAMPLING_CONFIG_PATHS:
        src = config
        for key in path:
            src = _read_knob(src, key)
            if src is None:
                break
        value = _read_knob(src, name)
        if value is not None:
            return value
    return None


def fill_und_ar_sampling_params(params: dict[str, Any], config: Any = None) -> dict[str, Any]:
    """Pin the UND AR knobs to the configured recipe whenever the caller left them at the
    untruncated defaults."""
    for name, fallback in UND_AR_SAMPLING_FALLBACK.items():
        current = params.get(name)
        if current is not None and current != UNTRAINED_AR_SAMPLING[name]:
            continue
        value = config_sampling_value(config, name)
        params[name] = fallback if value is None else value
    return params


# ``sde`` (default) keeps the trained FlowGRPO kernel from ``rollout.algo`` (``noise_level > 0`` + a bounded
# ``sde_window_*``).
GEN_DECODE_SDE = "sde"
GEN_DECODE_ODE = "ode"
# Env knob (not a Hydra key) so it cannot leak into the engine sampling params, which
# are forwarded verbatim to vLLM-Omni. Set by the recipe as ``GEN_DECODE``.
GEN_DECODE_ENV = "BAGEL_CORL_GEN_DECODE"
GEN_ODE_STEPS = 50
GEN_ODE_NOISE_LEVEL = 0.0


def resolve_gen_decode_mode() -> str:
    """Return the GEN decode kernel: ``sde`` (default) or ``ode``.

    A typo is a hard error: silently picking ``ode`` would disable the GEN loss, and
    silently picking ``sde`` without a window would grain the rollouts. Neither is a
    failure mode to guess through.
    """
    mode = os.environ.get(GEN_DECODE_ENV, GEN_DECODE_SDE).strip().lower()
    if mode not in (GEN_DECODE_SDE, GEN_DECODE_ODE):
        raise ValueError(f"{GEN_DECODE_ENV}={mode!r} is invalid; expected {GEN_DECODE_SDE!r} or {GEN_DECODE_ODE!r}")
    return mode


def config_to_sampling_dict(config: Any) -> dict[str, Any]:
    """Flatten OmegaConf / BaseConfig / dict / namespace into sampling kwargs."""
    if config is None:
        return {}
    items = None
    if isinstance(config, dict):
        items = config.items()
    elif hasattr(config, "items"):
        try:
            items = config.items()
        except (TypeError, AttributeError):
            items = None
    if items is not None:
        return {k: v for k, v in items if not str(k).startswith("_") and v is not None}
    if hasattr(config, "__dict__"):
        return {
            k: v for k, v in vars(config).items() if not str(k).startswith("_") and v is not None and not callable(v)
        }
    return {}


def build_gen_sampling_params(
    rollout_config: Any,
    *,
    base: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Merge diffusion ``pipeline`` + ``algo`` into GEN request sampling params.

    AgentLoopWorkerTQ builds AR-only params (temperature / top_p / logprobs). GEN
    must also carry ``num_inference_steps``, ``noise_level``, ``sde_window_*``, or
    Bagel FlowGRPO will not stash ``trajectory_latents`` / logprobs for training.
    """
    pipeline = getattr(rollout_config, "pipeline", None)
    if pipeline is None and isinstance(rollout_config, dict):
        pipeline = rollout_config.get("pipeline")
    algo = getattr(rollout_config, "algo", None)
    if algo is None and isinstance(rollout_config, dict):
        algo = rollout_config.get("algo")

    params: dict[str, Any] = {}
    params.update(config_to_sampling_dict(pipeline))
    params.update(config_to_sampling_dict(algo))
    if base:
        # ``base`` is the worker's per-phase sampling params; on a validation batch it
        # carries the val overlay (``val_overlay_sampling_params``). Keep this branch
        # *after* the train merge so the val kernel wins -- applying the training
        # ``pipeline``/``algo`` last stamped the train steps/noise over validation.
        # Keep non-AR keys (e.g. global_steps) from the worker; drop text-decoding knobs.
        params.update({k: v for k, v in base.items() if k not in _AR_ONLY_SAMPLING_KEYS})

    calculate = getattr(rollout_config, "calculate_log_probs", None)
    if calculate is None and isinstance(rollout_config, dict):
        calculate = rollout_config.get("calculate_log_probs")
    if calculate is None:
        raise ValueError(
            "bagel_corl GEN requires actor_rollout_ref.rollout.calculate_log_probs=True "
            "(do not default it on; missing config hides ODE rollouts)."
        )
    if not bool(calculate):
        raise ValueError(
            "bagel_corl GEN requires actor_rollout_ref.rollout.calculate_log_probs=True "
            "(refuse soft-skip of GEN diffusion_loss)."
        )
    # Resolve the decode *after* the train/val merge, so ``base``'s per-phase
    # ``logprobs`` (False on a validation batch) is what decides the val kernel.
    if resolve_gen_decode_mode() == GEN_DECODE_ODE:
        # Deterministic, log-prob-free. The image is the whole output, so the row
        # carries no trajectory and the GEN *training* lane drops it by design.
        params["num_inference_steps"] = GEN_ODE_STEPS
        params["noise_level"] = GEN_ODE_NOISE_LEVEL
        # An ODE step has ``std_dev_t == 0``; asking for a Gaussian log-prob there is
        # ``0/0``. FlowGRPO's SDE stash does not apply.
        params["logprobs"] = False
        if seed is not None:
            params["seed"] = int(seed)
        return params

    # SDE.
    wants_logprobs = (base or {}).get("logprobs") is not False
    params["logprobs"] = wants_logprobs

    noise_level = float(params.get("noise_level", 0.0) or 0.0)
    if wants_logprobs and noise_level <= 0.0:
        raise ValueError(
            "bagel_corl GEN requires rollout.algo.noise_level > 0 so the SDE window "
            f"records latents/logprobs (got noise_level={noise_level}); "
            f"set {GEN_DECODE_ENV}={GEN_DECODE_ODE} to decode deterministically instead."
        )
    if wants_logprobs and params.get("num_inference_steps") is None:
        raise ValueError("bagel_corl GEN requires rollout.pipeline.num_inference_steps")
    if wants_logprobs and not params.get("sde_window_size"):
        # No window => the adapter's fallback applies noise on every step, the
        # terminal one included, and that noise is never removed. Fail loud instead
        # of training on -- and rewarding -- grainy images.
        raise ValueError(
            "bagel_corl GEN requires rollout.algo.sde_window_size > 0 so the SDE noise "
            "is bounded to a window the deterministic tail can undo (an unset window "
            "noises every step and bakes it into the PNG); set "
            f"{GEN_DECODE_ENV}={GEN_DECODE_ODE} to decode deterministically instead."
        )
    if seed is not None:
        params["seed"] = int(seed)
    return params


def _maybe_unbatch_vector(value: Any) -> Any:
    """Unbatch (1, T) → (T,) for logprobs / timesteps; leave (T,) alone."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() == 2 and int(value.shape[0]) == 1:
            return value[0]
        return value
    if isinstance(value, list | tuple) and len(value) == 1:
        return value[0]
    return value


def _maybe_unbatch_latents(value: Any) -> Any:
    """Unbatch (1, T, ...) → (T, ...); never squeeze a lone timestep on dim0."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        # Strategy already peels B; only peel leftover batch when clearly B,T,* (≥3D with B=1).
        if value.dim() >= 3 and int(value.shape[0]) == 1:
            return value[0]
        return value
    if isinstance(value, list | tuple) and len(value) == 1:
        return value[0]
    return value


def extract_gen_traj_from_diffusion_output(output: Any) -> tuple[Any, Any, Any]:
    """Pull ``all_latents`` / timesteps / log_probs from verl_omni ``DiffusionOutput``."""
    extra = dict(getattr(output, "extra_fields", None) or {})
    latents = _maybe_unbatch_latents(extra.get("all_latents"))
    timesteps = _maybe_unbatch_vector(extra.get("all_timesteps"))
    log_probs = _maybe_unbatch_vector(getattr(output, "log_probs", None))
    return latents, timesteps, log_probs


def save_diffusion_image(diffusion_output: Any, *, seed: int, root: Optional[str] = None) -> str:
    """Persist uint8 CHW/HWC image for RM / UND observation; return absolute path."""
    from PIL import Image

    image_root = Path(root or os.environ.get("BAGEL_CORL_GEN_IMAGE_DIR") or "/tmp/bagel_corl_gen")
    image_root.mkdir(parents=True, exist_ok=True)
    path = image_root / f"gen_{uuid.uuid4().hex}_s{int(seed)}.png"

    tensor = diffusion_output
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != torch.uint8:
            arr = arr.float().clamp(0, 1).mul(255).round().to(torch.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.permute(1, 2, 0)
        pil = Image.fromarray(arr.numpy())
    else:
        pil = Image.fromarray(tensor) if not hasattr(tensor, "save") else tensor
    if getattr(pil, "mode", None) not in (None, "RGB"):
        pil = pil.convert("RGB")
    pil.save(path)
    return str(path.resolve())


def stash_gen_row_from_diffusion_output(
    output: Any,
    *,
    seed: int,
    image_root: Optional[str] = None,
    require_traj: bool = True,
) -> dict[str, Any]:
    """Build one GEN seed row."""
    if not hasattr(output, "diffusion_output"):
        raise RuntimeError(
            "Bagel Co-RL (Joint-Training) generate_image returned a non-diffusion output: dual-role GEN "
            "serving is not wired (bagel_single_stage is GEN-only / the AR replica is not "
            "diffusion-capable). Do not fall back to a Qwen sidecar."
        )

    stop_reason = getattr(output, "stop_reason", None)
    if stop_reason in ("aborted", "abort", "error"):
        raise RuntimeError(f"Bagel Co-RL (Joint-Training) GEN aborted (stop_reason={stop_reason!r}); refuse soft-skip.")

    latents, timesteps, log_probs = extract_gen_traj_from_diffusion_output(output)
    # Training still fails closed *before* writing anything.  Validation
    # (``require_traj=False``) falls through to save the image: an ODE decode
    # legitimately has no log-probs, and the image is the point of a val pass.
    if require_traj and (latents is None or timesteps is None or log_probs is None):
        raise RuntimeError(
            "Bagel Co-RL (Joint-Training) GEN traj stash incomplete from vLLM-Omni "
            f"(all_latents={'ok' if latents is not None else 'missing'}, "
            f"timesteps={'ok' if timesteps is not None else 'missing'}, "
            f"log_probs={'ok' if log_probs is not None else 'missing'}). "
            "Require calculate_log_probs + algo.noise_level>0 + SDE window; "
            "refuse soft-fallback to stashed-only / skip GEN loss."
        )

    from verl_omni.agent_loop.bagel_corl_lib import serving_rollout_provenance

    extra = dict(getattr(output, "extra_fields", None) or {})
    provenance = serving_rollout_provenance(extra)
    image_path = extra.get("image_path") or extra.get("path")
    if not image_path:
        image_path = save_diffusion_image(output.diffusion_output, seed=seed, root=image_root)

    return {
        "valid": True,
        "all_latents": latents,
        "timesteps": timesteps,
        "rollout_log_probs": log_probs,
        "image_path": image_path,
        **provenance,
    }
