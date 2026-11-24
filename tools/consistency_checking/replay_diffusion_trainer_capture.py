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

"""Replay complete trainer driver pairs and bounded engine probes on CPU.

The capture records attest to payload bytes and source identity, not model
weights or execution provenance. Engine probes do not establish full coverage.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import stat
from pathlib import Path
from typing import Any

_checker_spec = importlib.util.spec_from_file_location(
    "_diffusion_consistency_checker", Path(__file__).with_name("check_diffusion_consistency.py")
)
checker = importlib.util.module_from_spec(_checker_spec)
_checker_spec.loader.exec_module(checker)

_UUID_HEX = re.compile(r"[0-9a-f]{32}\Z")
_COMMON = {
    "version",
    "kind",
    "run_id",
    "batch_id",
    "global_step",
    "rank",
    "payload_sha256",
    "capture_source_sha256",
}
_FLOAT_FIELDS = ("actor_log_probs", "rollout_log_probs", "timesteps")
_SIGMA_FIELDS = ("sigma", "sigma_next", "next_timesteps")


def _plain_float(value: Any, shape: tuple[int, int], label: str) -> None:
    import torch

    checker._require(
        type(value) is torch.Tensor
        and not value.requires_grad
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype == torch.float32
        and tuple(value.shape) == shape
        and bool(torch.isfinite(value).all()),
        f"{label} must be a finite CPU float32 tensor of shape {shape}",
    )


def _event(folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    checker._require(folder.is_dir() and not folder.is_symlink(), f"invalid event directory: {folder.name}")
    checker._require(
        {p.name for p in folder.iterdir()} == {"record.json", "payload.pt"}, f"partial event: {folder.name}"
    )
    for name in ("record.json", "payload.pt"):
        path = folder / name
        checker._require(
            not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),
            f"event file must be regular: {folder.name}/{name}",
        )
    record = checker._json((folder / "record.json").read_bytes(), f"{folder.name}/record.json")
    checker._require(type(record) is dict, f"record must be an object: {folder.name}")
    kind = record.get("kind")
    checker._require(kind in ("driver_pair", "engine_forward"), f"event kind invalid: {folder.name}")
    keys = _COMMON | ({"sigma_timestep_contract"} if kind == "engine_forward" else set())
    checker._require(set(record) == keys, f"record fields invalid: {folder.name}")
    checker._require(type(record["version"]) is int and record["version"] == 1, "record version must be 1")
    checker._require(type(record["run_id"]) is str and bool(record["run_id"]), "run_id invalid")
    checker._require(
        type(record["batch_id"]) is str and _UUID_HEX.fullmatch(record["batch_id"]) is not None,
        "batch_id must be a hex UUID",
    )
    checker._require(type(record["global_step"]) is int, "global_step must be an integer")
    checker._require(type(record["rank"]) is int and record["rank"] >= 0, "rank invalid")
    for field in ("payload_sha256", "capture_source_sha256"):
        value = record[field]
        checker._require(type(value) is str and checker._HEX64.fullmatch(value) is not None, f"{field} invalid")
    if kind == "engine_forward":
        checker._require(type(record["sigma_timestep_contract"]) is bool, "sigma_timestep_contract invalid")
    data = (folder / "payload.pt").read_bytes()
    checker._require(checker._digest(data) == record["payload_sha256"], f"payload SHA256 mismatch: {folder.name}")
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except MemoryError:
        raise
    except Exception as exc:
        raise ValueError(f"payload cannot be decoded: {folder.name}") from exc
    checker._require(type(payload) is dict, f"payload map invalid: {folder.name}")
    required = set(_FLOAT_FIELDS)
    if kind == "driver_pair":
        required |= {"uids"}
    elif record["sigma_timestep_contract"]:
        required |= {*_SIGMA_FIELDS, "num_train_timesteps", "sigma_indices"}
    checker._require(set(payload) == required, f"payload fields invalid: {folder.name}")
    actor = payload["actor_log_probs"]
    checker._require(type(actor) is torch.Tensor and actor.ndim == 2, "actor_log_probs must be a matrix")
    n, t = actor.shape
    checker._require(n > 0 and t > 0 and (kind == "driver_pair" or t == 1), "payload matrix shape invalid")
    shape = (n, t)
    for field in _FLOAT_FIELDS:
        _plain_float(payload[field], shape, field)
    if kind == "driver_pair":
        uids = payload["uids"]
        checker._require(
            type(uids) is list and len(uids) == n and all(type(uid) is str for uid in uids),
            "uids must be a string list aligned with rows",
        )
    elif record["sigma_timestep_contract"]:
        for field in _SIGMA_FIELDS:
            _plain_float(payload[field], shape, field)
        checker._require(
            type(payload["num_train_timesteps"]) is int and payload["num_train_timesteps"] > 0,
            "num_train_timesteps must be a positive integer",
        )
        indices = payload["sigma_indices"]
        checker._require(
            type(indices) is torch.Tensor
            and not indices.requires_grad
            and indices.device.type == "cpu"
            and indices.layout == torch.strided
            and indices.dtype == torch.long
            and tuple(indices.shape) == (n,)
            and bool((indices >= 0).all()),
            "sigma_indices must be a nonnegative CPU long vector",
        )
    return record, payload


def _per_timestep(actor: Any, rollout: Any, timesteps: Any) -> list[dict[str, Any]]:
    """Group absolute log-probability differences by raw timestep value."""
    difference = (actor.double() - rollout.double()).abs()
    result = []
    for step in timesteps.unique(sorted=True):
        selected = difference[timesteps == step]
        result.append(
            {
                "raw_timestep": float(step),
                "count": int(selected.numel()),
                "rollout_corr/logprob_abs_diff_mean": float(selected.mean()),
                "rollout_corr/logprob_abs_diff_max": float(selected.max()),
            }
        )
    return result


def _sigma_faults(record: dict[str, Any], payload: dict[str, Any], event_name: str) -> list[dict[str, Any]]:
    """Compare the consumed current and next sigmas with their raw timesteps."""
    import torch

    if not record["sigma_timestep_contract"]:
        return []
    scale = payload["num_train_timesteps"]
    faults = []
    for field, step_field in (("sigma", "timesteps"), ("sigma_next", "next_timesteps")):
        actual = payload[field].double()
        expected = payload[step_field].double() / scale
        close = torch.isclose(actual, expected, atol=1e-6, rtol=1e-5)
        for row in (~close[:, 0]).nonzero(as_tuple=True)[0].tolist():
            faults.append(
                {
                    "reason": "scheduler_sigma_timestep_mismatch",
                    "event": event_name,
                    "sample_local_row": row,
                    "sigma_index": int(payload["sigma_indices"][row]),
                    "raw_timestep": float(payload[step_field][row, 0]),
                    "field": field,
                    "expected": float(expected[row, 0]),
                    "actual": float(actual[row, 0]),
                }
            )
    return faults


def replay(root: Path, output: Path, atol: float = 1e-5, rtol: float = 1e-3) -> dict[str, Any]:
    """Validate a fresh capture root and write one exclusive JSON report."""
    import torch

    checker._number(atol, "atol", nonnegative=True)
    checker._number(rtol, "rtol", nonnegative=True)
    root = root.absolute()
    checker._require(".." not in root.parts and root == root.resolve(strict=True), "root must be unsymlinked")
    checker._require(root.is_dir(), "root must be a directory")
    output = output.absolute()
    checker._require(
        ".." not in output.parts and not output.resolve().is_relative_to(root), "output must be outside root"
    )
    checker._require(output.parent.is_dir() and not output.exists() and not output.is_symlink(), "output must be fresh")
    checker._require(not torch.cuda.is_initialized(), "CPU replay initialized CUDA")
    folders = sorted(root.iterdir())
    checker._require(
        bool(folders) and all(p.is_dir() and not p.is_symlink() for p in folders), "root has invalid events"
    )
    events = [(folder.name, *_event(folder)) for folder in folders]
    drivers = {}
    for name, record, payload in events:
        if record["kind"] != "driver_pair":
            continue
        key = (record["run_id"], record["batch_id"], record["global_step"])
        checker._require(key not in drivers, f"duplicate driver batch: {name}")
        drivers[key] = (name, record, payload)
    checker._require(bool(drivers), "at least one driver_pair is required")
    engine_count = len(events) - len(drivers)
    checker._require(engine_count > 0, "at least one engine_forward is required")
    for name, record, _ in events:
        if record["kind"] != "engine_forward":
            continue
        key = (record["run_id"], record["batch_id"], record["global_step"])
        checker._require(key in drivers, f"orphan engine event: {name}")
        checker._require(
            record["capture_source_sha256"] == drivers[key][1]["capture_source_sha256"],
            f"capture source mismatch: {name}",
        )
    findings = []
    sigma_contract_probe_count = 0
    any_numeric = False
    any_logic = False
    for name, record, payload in events:
        comparison = checker.compare_log_probs(payload["actor_log_probs"], payload["rollout_log_probs"], atol, rtol)
        faults = _sigma_faults(record, payload, name) if record["kind"] == "engine_forward" else []
        sigma_contract_probe_count += int(record.get("sigma_timestep_contract", False))
        any_numeric |= not comparison["allclose"]
        any_logic |= bool(faults)
        findings.append(
            {
                "event": name,
                "kind": record["kind"],
                "run_id": record["run_id"],
                "batch_id": record["batch_id"],
                "global_step": record["global_step"],
                "rank": record["rank"],
                "shape": list(payload["actor_log_probs"].shape),
                "comparison": comparison,
                "per_timestep": _per_timestep(
                    payload["actor_log_probs"], payload["rollout_log_probs"], payload["timesteps"]
                ),
                "sigma_timestep_contract": record.get("sigma_timestep_contract"),
                "sigma_faults": faults,
            }
        )
    status = "logic_mismatch" if any_logic else "numerical_mismatch" if any_numeric else "admitted"
    report = {
        "status": status,
        "scope": {
            "driver_pairs": "every saved row and timestep in every driver_pair payload",
            "engine_forward": "saved forward probes only; no full engine coverage claim",
            "sigma": "consumed current and next sigma values only when sigma_timestep_contract is true",
            "model": "no model weights or model identity attested; shared diffusion PPO log-probability fields only",
        },
        "atol": atol,
        "rtol": rtol,
        "driver_pair_count": len(drivers),
        "engine_forward_count": engine_count,
        "sigma_contract_probe_count": sigma_contract_probe_count,
        "sigma_diagnosis_available": sigma_contract_probe_count > 0,
        "findings": findings,
    }
    checker._require(not torch.cuda.is_initialized(), "CPU replay initialized CUDA")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    """Run CPU replay and encode the classified verdict as an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    report = replay(args.root, args.output, args.atol, args.rtol)
    raise SystemExit({"admitted": 0, "logic_mismatch": 2, "numerical_mismatch": 3}[report["status"]])


if __name__ == "__main__":
    main()
