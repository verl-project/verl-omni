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
"""Debug hooks used only by the Qwen-Image FlowGRPO nightly regression."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_DEBUG_STEP_KEY = "_nightly_debug_global_step"
_WRAPPED_ATTR = "__nightly_debug_wrapped__"
_EVENT_COUNTER_LOCK = threading.Lock()
_EVENT_COUNTER = 0


def _enabled() -> bool:
    return os.environ.get("DEBUG_DUMP_ENABLED", "1").lower() not in {"0", "false", "no"}


def _set_local_rank_device() -> None:
    if os.environ.get("RANK") is None:
        return
    try:
        local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.environ["RANK"]) % max(local_world_size, 1)
    except ValueError:
        return
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if len(devices) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank % len(devices)]
            local_rank = 0
    os.environ["LOCAL_RANK"] = str(local_rank)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except Exception as exc:
        print(f"WARN nightly debug failed to set local CUDA device: {exc}")


def _dump_dir() -> Path:
    return Path(os.environ.get("DEBUG_DUMP_DIR", "outputs/debug_dumps/current")).expanduser().resolve()


def _dump_steps() -> set[int]:
    raw = os.environ.get("DEBUG_DUMP_STEPS", "1,2")
    steps = set()
    for part in re.split(r"[, ]+", raw.strip()):
        if part:
            steps.add(int(part))
    return steps


def _next_event_id() -> int:
    global _EVENT_COUNTER
    with _EVENT_COUNTER_LOCK:
        _EVENT_COUNTER += 1
        return _EVENT_COUNTER


def _rank_info() -> dict[str, int]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except Exception:
        pass
    return {"rank": rank, "world_size": world_size}


def _json_default(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "mean": value.detach().float().mean().cpu().item() if value.numel() else 0.0,
            }
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _to_cpu_payload(value: Any, *, max_sequence_items: int = 128) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return value[:max_sequence_items].tolist()
        return value.copy()
    if isinstance(value, Mapping):
        return {str(k): _to_cpu_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_cpu_payload(v) for v in list(value)[:max_sequence_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _tensor_dict_subset(data: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    subset = {}
    for key in keys:
        try:
            if key in data:
                subset[key] = data[key]
        except Exception:
            pass
    return subset


def _extract_debug_step(data: Any = None, fallback: int | None = None) -> int | None:
    if data is not None:
        for getter in (
            lambda: data[_DEBUG_STEP_KEY],
            lambda: data.get(_DEBUG_STEP_KEY),
        ):
            try:
                value = getter()
            except Exception:
                continue
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if isinstance(value, np.ndarray):
                    value = value.reshape(-1)[0]
                elif hasattr(value, "reshape"):
                    value = value.reshape(-1)[0]
                if hasattr(value, "item"):
                    value = value.item()
                return int(value)
            except Exception:
                continue
    return fallback


def _event_dir(kind: str, identity: Mapping[str, Any]) -> Path:
    parts = [
        kind,
        f"step_{int(identity['global_step']):06d}",
        f"rank_{int(identity.get('rank', 0)):05d}",
    ]
    for key in ("ppo_epoch", "mini_batch_idx", "micro_batch_idx", "diffusion_step"):
        if key in identity and identity[key] is not None:
            parts.append(f"{key}_{int(identity[key]):04d}")
    event_id = _next_event_id()
    parts.append(f"event_{event_id:06d}")
    return _dump_dir().joinpath(*parts)


def _dump_event(kind: str, identity: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if not _enabled():
        return
    global_step = identity.get("global_step")
    if global_step is None or int(global_step) not in _dump_steps():
        return

    full_identity = {**identity, **_rank_info(), "kind": kind, "created_at": time.time()}
    output_dir = _event_dir(kind, full_identity)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "identity.json").open("w", encoding="utf-8") as file:
        json.dump(full_identity, file, indent=2, sort_keys=True, default=_json_default)

    try:
        import torch

        torch.save(_to_cpu_payload(payload), output_dir / "payload.pt")
    except Exception as exc:
        with (output_dir / "payload_error.json").open("w", encoding="utf-8") as file:
            json.dump({"error": repr(exc)}, file, indent=2)


def _install_tracking_hook() -> None:
    try:
        from verl.utils.tracking import Tracking
    except Exception:
        return

    original_log = Tracking.log
    if getattr(original_log, _WRAPPED_ATTR, False):
        return

    def log_with_jsonl(self, data, step, *args, **kwargs):
        metrics_path = os.environ.get("DEBUG_METRICS_JSONL")
        if metrics_path:
            path = Path(metrics_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"step": int(step), "data": data}
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        return original_log(self, data, step, *args, **kwargs)

    setattr(log_with_jsonl, _WRAPPED_ATTR, True)
    Tracking.log = log_with_jsonl


def _install_driver_hook() -> None:
    try:
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer
    except Exception:
        return

    original_update_actor = PolicyGradientRayTrainer._update_actor
    if getattr(original_update_actor, _WRAPPED_ATTR, False):
        return

    def update_actor_with_dump(self, batch, *args, **kwargs):
        global_step = int(getattr(self, "global_steps", 0))
        try:
            batch_size = len(batch)
        except Exception:
            batch_size = 1
        try:
            batch.non_tensor_batch[_DEBUG_STEP_KEY] = np.full(batch_size, global_step, dtype=np.int64)
        except Exception:
            pass

        payload_keys = (
            "responses",
            "log_probs",
            "old_log_probs",
            "advantages",
            "sample_level_scores",
            "sample_level_rewards",
            "latents",
            "all_latents",
            "all_timesteps",
        )
        payload = {
            "batch": _tensor_dict_subset(batch.batch, payload_keys),
            "non_tensor": {
                key: batch.non_tensor_batch[key]
                for key in ("uid", "data_source", _DEBUG_STEP_KEY)
                if key in batch.non_tensor_batch
            },
        }
        _dump_event("driver_forward", {"global_step": global_step}, payload)
        return original_update_actor(self, batch, *args, **kwargs)

    setattr(update_actor_with_dump, _WRAPPED_ATTR, True)
    PolicyGradientRayTrainer._update_actor = update_actor_with_dump


def _install_training_worker_hook() -> None:
    try:
        from verl_omni.workers.engine_workers import TrainingWorker
    except Exception:
        return

    original_train_mini_batch = TrainingWorker.train_mini_batch
    if getattr(original_train_mini_batch, _WRAPPED_ATTR, False):
        return

    def train_mini_batch_with_context(self, data, *args, **kwargs):
        global_step = _extract_debug_step(data)
        if global_step is not None:
            self._nightly_debug_global_step = global_step
            if hasattr(self, "engine"):
                self.engine._nightly_debug_global_step = global_step
        return original_train_mini_batch(self, data, *args, **kwargs)

    setattr(train_mini_batch_with_context, _WRAPPED_ATTR, True)
    TrainingWorker.train_mini_batch = train_mini_batch_with_context


def _lora_gradients(module: Any) -> dict[str, Any]:
    grads = {}
    try:
        named_parameters = module.named_parameters()
    except Exception:
        return grads
    for name, param in named_parameters:
        if "lora" not in name.lower() or getattr(param, "grad", None) is None:
            continue
        grad = param.grad
        try:
            if hasattr(grad, "full_tensor"):
                grad = grad.full_tensor()
        except Exception:
            pass
        grads[name] = grad
    return grads


def _install_engine_hooks() -> None:
    try:
        from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
    except Exception:
        return

    original_run_batch = DiffusersFSDPEngine._run_forward_backward_batch
    if not getattr(original_run_batch, _WRAPPED_ATTR, False):

        def run_forward_backward_batch_with_dump(self, data, loss_function, forward_only, *args, **kwargs):
            output = original_run_batch(self, data, loss_function, forward_only, *args, **kwargs)
            global_step = _extract_debug_step(data, getattr(self, "_nightly_debug_global_step", None))
            if global_step is not None:
                identity = {
                    "global_step": global_step,
                    "forward_only": bool(forward_only),
                    "timesteps_key": kwargs.get("timesteps_key"),
                }
                _dump_event("actor_forward", identity, output)
            return output

        setattr(run_forward_backward_batch_with_dump, _WRAPPED_ATTR, True)
        DiffusersFSDPEngine._run_forward_backward_batch = run_forward_backward_batch_with_dump

    original_forward_step = DiffusersFSDPEngine.forward_step
    if os.environ.get("DEBUG_DUMP_FORWARD_STEPS", "0").lower() in {"1", "true", "yes"} and not getattr(
        original_forward_step, _WRAPPED_ATTR, False
    ):

        def forward_step_with_dump(self, micro_batch, loss_function, forward_only, step, *args, **kwargs):
            loss, meta_info = original_forward_step(
                self, micro_batch, loss_function, forward_only, step, *args, **kwargs
            )
            global_step = _extract_debug_step(micro_batch, getattr(self, "_nightly_debug_global_step", None))
            if global_step is not None:
                _dump_event(
                    "actor_forward_step",
                    {"global_step": global_step, "diffusion_step": int(step), "forward_only": bool(forward_only)},
                    {"loss": loss, "meta_info": meta_info},
                )
            return loss, meta_info

        setattr(forward_step_with_dump, _WRAPPED_ATTR, True)
        DiffusersFSDPEngine.forward_step = forward_step_with_dump

    original_optimizer_step = DiffusersFSDPEngine.optimizer_step
    if getattr(original_optimizer_step, _WRAPPED_ATTR, False):
        return

    def optimizer_step_with_dump(self, *args, **kwargs):
        global_step = getattr(self, "_nightly_debug_global_step", None)
        if global_step is not None:
            grads = _lora_gradients(getattr(self, "module", None))
            if grads:
                _dump_event("lora_gradients", {"global_step": int(global_step)}, {"gradients": grads})
        grad_norm = original_optimizer_step(self, *args, **kwargs)
        if isinstance(grad_norm, float) and not math.isfinite(grad_norm):
            print(f"WARN nightly debug observed non-finite grad_norm: {grad_norm}")
        return grad_norm

    setattr(optimizer_step_with_dump, _WRAPPED_ATTR, True)
    DiffusersFSDPEngine.optimizer_step = optimizer_step_with_dump


def install_debug_hooks() -> None:
    """Install all test-side monkey patches. Safe to call multiple times."""
    if not _enabled():
        return
    _set_local_rank_device()
    _install_tracking_hook()
    _install_driver_hook()
    _install_training_worker_hook()
    _install_engine_hooks()
