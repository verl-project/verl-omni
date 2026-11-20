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
"""Opt-in observation of real diffusion trainer recomputation and SDE inputs.

Installed by an explicit Ray worker-process setup hook. No model configuration,
RNG state, loss, optimizer or precision setting is changed by this observer.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

_CONTEXT_KEY = "_diffusion_consistency_capture"
_WRAPPED = "__diffusion_consistency_capture_wrapped__"
_ENGINE_CONTEXT: ContextVar[dict | None] = ContextVar("diffusion_capture_batch", default=None)
_MISSING = object()

# Units belong to the observer, keyed by the resolved adapter rather than its architecture alias.
_TIMESTEP_UNITS = {
    ("verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter", "StableDiffusion3FlowGRPO"): "training_step",
}


def _selected(step: int) -> bool:
    raw = os.environ.get("DIFFUSION_CAPTURE_STEPS", "1")
    steps = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not steps or any(value < 0 for value in steps):
        raise ValueError("DIFFUSION_CAPTURE_STEPS must contain nonnegative integer steps")
    return step in steps


def _cpu(value):
    return value.detach().to(device="cpu").clone()


def _write(kind: str, context: dict, payload: dict, **extra: Any) -> None:
    import torch
    import torch.distributed as dist

    root = Path(os.environ["DIFFUSION_CAPTURE_DIR"]).absolute()
    if root.is_symlink():
        raise ValueError("capture root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    event = root / f"{kind}-{uuid4().hex}"
    event.mkdir()
    target = event / "payload.pt"
    torch.save(payload, target)
    record = {
        "version": 1,
        "kind": kind,
        **context,
        "rank": dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        "payload_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "capture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **extra,
    }
    # Completion metadata is published last; readers reject partial event dirs.
    with (event / "record.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _driver_pair(data, result, context: dict) -> None:
    reference = data.batch["rollout_log_probs"]
    actual = result.batch["old_log_probs"]
    raw_uids = data.non_tensor_batch.get("uid")
    uids = [str(value) for value in raw_uids] if raw_uids is not None else [str(i) for i in range(len(reference))]
    _write(
        "driver_pair",
        context,
        {
            "actor_log_probs": _cpu(actual.float()),
            "rollout_log_probs": _cpu(reference.float()),
            "timesteps": _cpu(data.batch["all_timesteps"].float()),
            "uids": uids,
        },
    )


def _schedule_inputs(engine, micro_batch, step: int) -> tuple[dict, bool]:
    from verl_omni.pipelines.model_base import DiffusionModelBase
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    scheduler = engine.scheduler
    timesteps = micro_batch["all_timesteps"][:, step]
    payload = {
        "rollout_log_probs": _cpu(micro_batch["rollout_log_probs"][:, step].float()).reshape(-1, 1),
        "timesteps": _cpu(timesteps.float()).reshape(-1, 1),
    }
    adapter = DiffusionModelBase.get_class(engine.model_config)
    contract = (
        isinstance(scheduler, FlowMatchSDEDiscreteScheduler)
        and _TIMESTEP_UNITS.get((adapter.__module__, adapter.__qualname__)) == "training_step"
    )
    if contract:
        import torch

        indices = [scheduler.index_for_timestep(t) for t in timesteps]
        sigma = scheduler.sigmas
        raw = scheduler.timesteps
        payload.update(
            sigma=_cpu(torch.stack([sigma[i] for i in indices]).float()).reshape(-1, 1),
            sigma_next=_cpu(torch.stack([sigma[i + 1] for i in indices]).float()).reshape(-1, 1),
            next_timesteps=_cpu(
                torch.stack([raw[i + 1] if i + 1 < len(raw) else raw.new_zeros(()) for i in indices]).float()
            ).reshape(-1, 1),
            sigma_indices=torch.tensor(indices, dtype=torch.int64),
            num_train_timesteps=int(scheduler.config.num_train_timesteps),
        )
    return payload, contract


def install_trainer_capture() -> None:
    """Observe selected real trainer batches through Ray worker setup.

    The current consumer is PPO/SDE recomputation through the shared Diffusers
    engine. Other algorithms are not inferred from an architecture name.
    """
    if not os.environ.get("DIFFUSION_CAPTURE_DIR"):
        return

    from verl.utils import tensordict_utils as tu

    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
    from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine, PPODiffusersFSDPEngine

    original_recompute = PolicyGradientDiffusionTrainerV1._compute_old_log_prob
    if not getattr(original_recompute, _WRAPPED, False):

        @wraps(original_recompute)
        def recompute_with_capture(self, data, *args, **kwargs):
            step = int(self.global_steps)
            if not os.environ.get("DIFFUSION_CAPTURE_DIR") or not _selected(step):
                return original_recompute(self, data, *args, **kwargs)
            if "rollout_log_probs" not in data.batch or "all_timesteps" not in data.batch:
                raise ValueError("trainer capture requires rollout log probabilities and raw diffusion timesteps")
            context = {
                "run_id": Path(os.environ["DIFFUSION_CAPTURE_DIR"]).absolute().name,
                "batch_id": uuid4().hex,
                "global_step": step,
            }
            previous = data.meta_info.get(_CONTEXT_KEY, _MISSING)
            data.meta_info[_CONTEXT_KEY] = context
            try:
                result = original_recompute(self, data, *args, **kwargs)
                _driver_pair(data, result, context)
                return result
            finally:
                if previous is _MISSING:
                    data.meta_info.pop(_CONTEXT_KEY, None)
                else:
                    data.meta_info[_CONTEXT_KEY] = previous

        setattr(recompute_with_capture, _WRAPPED, True)
        PolicyGradientDiffusionTrainerV1._compute_old_log_prob = recompute_with_capture

    original_run = DiffusersFSDPEngine._run_forward_backward_batch
    if not getattr(original_run, _WRAPPED, False):

        @wraps(original_run)
        def run_with_capture(self, data, loss_function, forward_only, *args, **kwargs):
            context = tu.get_non_tensor_data(data, _CONTEXT_KEY, default=None) if forward_only else None
            token = _ENGINE_CONTEXT.set(context)
            try:
                return original_run(self, data, loss_function, forward_only, *args, **kwargs)
            finally:
                _ENGINE_CONTEXT.reset(token)

        setattr(run_with_capture, _WRAPPED, True)
        DiffusersFSDPEngine._run_forward_backward_batch = run_with_capture

    original_step = PPODiffusersFSDPEngine.forward_step
    if not getattr(original_step, _WRAPPED, False):

        @wraps(original_step)
        def step_with_capture(self, micro_batch, loss_function, forward_only, step, *args, **kwargs):
            context = _ENGINE_CONTEXT.get()
            capture = context is not None and forward_only and self.is_mp_src_rank_with_outputs()
            snapshot, contract = _schedule_inputs(self, micro_batch, step) if capture else (None, False)
            loss, output = original_step(self, micro_batch, loss_function, forward_only, step, *args, **kwargs)
            if capture:
                snapshot["actor_log_probs"] = _cpu(output["model_output"]["log_probs"].float()).reshape(-1, 1)
                _write("engine_forward", context, snapshot, sigma_timestep_contract=contract)
            return loss, output

        setattr(step_with_capture, _WRAPPED, True)
        PPODiffusersFSDPEngine.forward_step = step_with_capture
