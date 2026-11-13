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

"""Single-forward MSProbe capture for paired diffusion diagnostics.

The supplied policy ID declares checkpoint provenance; this helper cannot attest
weight equality. Official MSProbe tensor analysis owns that comparison.
"""

import hashlib
import json
import math
import re
import stat
from pathlib import Path

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STEP = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_PHASE = {"actor": "actorlog_prob", "rollout": "rolloutgenerate"}
_INPUT_NAMES = ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(type(value) in (int, float), f"{name} must be a finite JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def _context(context):
    _require(type(context) is dict, "context must be a dict")
    run_id = context.get("run_id")
    policy_id = context.get("policy_id")
    step = context.get("global_step")
    sample_id = context.get("sample_id")
    trajectory = context.get("trajectory_index")
    _require(type(run_id) is str and bool(run_id.strip()), "run_id missing")
    _require(type(policy_id) is str and _HEX64.fullmatch(policy_id) is not None, "policy_id must be lowercase SHA-256")
    _require(
        type(step) is int or (type(step) is str and _STEP.fullmatch(step) is not None),
        "global_step must be an integer or canonical integer string",
    )
    _require(type(sample_id) is str and bool(sample_id.strip()), "sample_id missing")
    _require(type(trajectory) is int and trajectory >= 0, "trajectory_index must be a nonnegative integer")
    _require(context.get("branch") == "cond", "branch must be cond")
    _require(context.get("timestep_unit") == "raw_scheduler", "timestep_unit must be raw_scheduler")
    _require(type(context.get("sde_type")) is str and bool(context["sde_type"].strip()), "sde_type missing")
    return {
        "run_id": run_id,
        "policy_id": policy_id,
        "global_step": int(step),
        "sample_id": sample_id,
        "trajectory_index": trajectory,
        "branch": "cond",
        "timestep_unit": "raw_scheduler",
        "timestep": _finite(context.get("timestep"), "timestep"),
        "noise_level": _finite(context.get("noise_level"), "noise_level"),
        "sde_type": context["sde_type"],
    }


def _tensor_inputs(inputs):
    # Import only on the enabled path. Copies/hashes happen before debugger.start.
    import torch

    _require(type(inputs) is dict, "inputs must be a dict of actual forward kwargs")
    result = {}
    batch_size = None
    for name in _INPUT_NAMES:
        tensor = inputs.get(name)
        _require(isinstance(tensor, torch.Tensor), f"{name} must be an actual torch.Tensor")
        _require(tensor.dtype == torch.float32, f"{name} must be float32")
        _require(tensor.ndim >= 1 and tensor.shape[0] == 1, f"{name} must have actual batch dimension B=1")
        if batch_size is None:
            batch_size = tensor.shape[0]
        _require(tensor.shape[0] == batch_size, f"{name} batch dimension mismatch")
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        result[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    # The public pairing reader names this input prompt_embeds; forward uses encoder_hidden_states.
    return {
        "hidden_states": result["hidden_states"],
        "prompt_embeds": result["encoder_hidden_states"],
        "pooled_projections": result["pooled_projections"],
    }, result["timestep"]


def _regular_nonempty(path):
    try:
        metadata = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0


def _actual_dump(dump_root, capture_dir):
    candidates = []
    for path in dump_root.rglob("*.json"):
        resolved = path.resolve()
        _require(resolved.is_relative_to(capture_dir), "dump JSON escapes capture directory")
        if not _regular_nonempty(resolved):
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid MSProbe JSON: {resolved}") from exc
        if type(payload) is not dict or type(payload.get("data")) is not dict:
            continue
        modules = [
            value
            for key, value in payload["data"].items()
            if type(key) is str and key.startswith("Module.") and type(value) is dict
        ]
        if not modules:
            continue
        _require(payload.get("task") == "tensor", "MSProbe dump is not tensor task")
        data_root = payload.get("dump_data_dir")
        _require(type(data_root) is str and bool(data_root), "tensor dump_data_dir missing")
        data_root = Path(data_root).resolve()
        _require(data_root.is_relative_to(capture_dir), "tensor data escapes capture directory")
        names = []

        def collect(value, names):
            if type(value) is dict:
                if "data_name" in value:
                    names.append(value["data_name"])
                for child in value.values():
                    collect(child, names)
            elif type(value) is list:
                for child in value:
                    collect(child, names)

        for module_data in modules:
            collect(module_data, names)
        _require(bool(names), "Module.* dump has no tensor artifact entries")
        for name in names:
            _require(type(name) is str and bool(name), "invalid tensor data_name")
            artifact = (data_root / name).resolve()
            _require(
                artifact.is_relative_to(data_root) and _regular_nonempty(artifact),
                f"tensor artifact missing or outside dump: {name}",
            )
        candidates.append(resolved)
    _require(len(candidates) == 1, f"expected exactly one nonempty Module.* tensor dump, found {len(candidates)}")
    return candidates[0]


def capture_forward(
    *,
    enabled: bool,
    role: str,
    module,
    forward: callable,
    inputs: dict,
    context: dict,
    output_root: Path,
    debugger_factory=None,
):
    """Capture one actual forward; return its exact result or propagate its error."""
    if not enabled:
        return forward()

    _require(type(role) is str and role in _PHASE, "role must be actor or rollout")
    fields = _context(context)
    input_metadata, timestep_input = _tensor_inputs(inputs)
    _require(isinstance(output_root, Path), "output_root must be a Path")
    root = output_root.resolve()
    _require(root.is_dir(), "output_root must already exist as a directory")
    _require(module is not None, "module is required for L0 capture")

    # The digest is a directory key, not a provenance or security attestation.
    identity = json.dumps(
        {
            key: fields[key]
            for key in ("run_id", "policy_id", "global_step", "sample_id", "trajectory_index", "timestep")
        },
        sort_keys=True,
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    role_dir = root / role
    role_dir.mkdir(exist_ok=True)
    _require(role_dir.resolve().is_relative_to(root), "role directory escapes output_root")
    capture_dir = role_dir / key
    capture_dir.mkdir(exist_ok=False)
    capture_dir = capture_dir.resolve()
    _require(capture_dir.is_relative_to(root), "capture directory escapes output_root")
    dump_root = capture_dir / "dump"
    config_path = capture_dir / "config.json"
    cleanup_error = None
    try:
        config_path.write_text(
            json.dumps(
                {
                    "task": "tensor",
                    "dump_path": str(dump_root),
                    "rank": [],
                    "step": [0],
                    "level": "L0",
                    "async_dump": False,
                    "tensor": {"scope": [], "list": [], "data_mode": ["all"]},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if debugger_factory is None:
            from msprobe.pytorch import PrecisionDebugger

            debugger_factory = PrecisionDebugger
        debugger = debugger_factory(config_path=str(config_path))
        try:
            debugger.start(model=module)
            result = forward()
        except BaseException as original:
            try:
                debugger.stop()
            except BaseException as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                original.add_note(f"MSProbe stop also failed: {cleanup_error}")
            raise
        debugger.stop()
        debugger.step()
        dump_path = _actual_dump(dump_root, capture_dir)
        _require(
            dump_path == dump_root / "step0/rank0/dump.json",
            "MSProbe dump path differs from the public replay byte contract",
        )
        record = {
            **fields,
            "role": role,
            "phase": _PHASE[role],
            "inputs": input_metadata,
            "dump_path": str(dump_path),
            "msprobe_step": 0,
            "timestep_input": timestep_input,
        }
        (capture_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return result
    except BaseException as exc:
        try:
            (capture_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "incomplete",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "stop_error": cleanup_error,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        raise
