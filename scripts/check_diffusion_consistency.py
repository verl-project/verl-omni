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

"""Check a pinned CPU diffusion rollout/actor artifact pair.

This reader verifies saved bytes and declared metadata; it does not run a model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STEP = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_PHASE = {"actor": "actorlog_prob", "rollout": "rolloutgenerate"}
_INPUTS = ("hidden_states", "prompt_embeds", "pooled_projections")


def _require(ok: bool, message: str) -> None:
    """Reject a violated artifact contract with its specific reason."""
    if not ok:
        raise ValueError(message)


def _digest(data: bytes) -> str:
    """Return the lowercase SHA256 digest used by pin entries."""
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes, label: str) -> Any:
    """Decode strict UTF-8 JSON without duplicate keys or nonfinite numbers."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Build an object while rejecting repeated JSON keys."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        """Reject JSON extensions for nonfinite numeric literals."""
        raise ValueError(f"{label}: nonfinite JSON number {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid UTF-8 JSON") from exc


def _number(value: Any, label: str, nonnegative: bool = False) -> float:
    """Validate a finite JSON number and its optional sign constraint."""
    _require(type(value) in (int, float), f"{label} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    _require(math.isfinite(number) and (not nonnegative or number >= 0), f"{label} must be finite and valid")
    return number


def _step(value: Any) -> int:
    """Normalize an integer or canonical signed integer step."""
    _require(
        type(value) is int or (type(value) is str and _STEP.fullmatch(value) is not None),
        "global_step must be an integer or canonical signed integer string",
    )
    return int(value)


def _relative(root: Path, name: Any) -> Path:
    """Resolve a regular artifact file without symlinks or root escape."""
    _require(type(name) is str and bool(name), "artifact path must be a nonempty relative string")
    raw = Path(name)
    _require(not raw.is_absolute() and ".." not in raw.parts and raw.as_posix() == name, "invalid relative path")
    path = root / raw
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"artifact path missing: {name}") from exc
    _require(path == resolved and resolved.is_relative_to(root), f"artifact path escapes root or uses symlink: {name}")
    _require(stat.S_ISREG(path.stat().st_mode), f"artifact is not a regular file: {name}")
    return path


def _pins(root: Path, pin_name: str, external_sha: str, manifest_name: str) -> dict[str, bytes]:
    """Verify the external pin digest and return every pinned file's bytes."""
    _require(type(external_sha) is str and _HEX64.fullmatch(external_sha) is not None, "invalid external pins SHA256")
    pin_path = _relative(root, pin_name)
    pin_bytes = pin_path.read_bytes()
    _require(_digest(pin_bytes) == external_sha, "external pins SHA256 mismatch")
    try:
        lines = pin_bytes.decode("utf-8").splitlines(keepends=True)
    except UnicodeError as exc:
        raise ValueError("pins must be UTF-8") from exc
    _require(bool(lines) and all(line.endswith("\n") for line in lines), "pins need newline-terminated entries")
    pinned: dict[str, bytes] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n\x00]+)\n", line)
        _require(match is not None, "invalid pin line; use SHA256, two spaces, relative path")
        digest, name = match.groups()
        _require(name not in pinned and name != pin_name, "duplicate or self-referential pin")
        path = _relative(root, name)
        data = path.read_bytes()
        _require(_digest(data) == digest, f"pinned byte mismatch: {name}")
        pinned[name] = data
    _require(manifest_name in pinned, "manifest is not pinned")
    return pinned


def _input(item: Any, label: str) -> dict[str, Any]:
    """Validate one declared input provenance record."""
    _require(type(item) is dict and set(item) == {"sha256", "dtype", "shape"}, f"{label} must have exact metadata")
    _require(type(item["sha256"]) is str and _HEX64.fullmatch(item["sha256"]) is not None, f"{label} SHA256 invalid")
    _require(type(item["dtype"]) is str and bool(item["dtype"].strip()), f"{label} dtype missing")
    _require(
        type(item["shape"]) is list and all(type(dim) is int and dim >= 0 for dim in item["shape"]),
        f"{label} shape invalid",
    )
    return item


def _dump(data: bytes, label: str) -> None:
    """Validate the required Module entry in a pinned dump."""
    payload = _json(data, label)
    _require(type(payload) is dict and type(payload.get("data")) is dict, f"{label} needs object data")
    _require(
        any(
            type(key) is str and key.startswith("Module.") and type(value) is dict and bool(value)
            for key, value in payload["data"].items()
        ),
        f"{label} needs a nonempty Module.* object",
    )


def _pairs(records: Any, expected: dict[str, Any], root: Path, pinned: dict[str, bytes]) -> list[dict[str, Any]]:
    """Pair actor and rollout records after validating shared provenance."""
    _require(type(records) is list and bool(records), "records must be a nonempty list")
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    seen_dumps: set[str] = set()
    for index, row in enumerate(records):
        label = f"record {index}"
        _require(type(row) is dict and "dummy" not in row and "warmup" not in row, f"{label} invalid")
        role = row.get("role")
        _require(
            type(role) is str and role in _PHASE and row.get("phase") == _PHASE[role],
            f"{label} role/phase mismatch",
        )
        _require(
            row.get("run_id") == expected["run_id"]
            and row.get("policy_id") == expected["policy_id"]
            and _step(row.get("global_step")) == expected["global_step"],
            f"{label} stale run/policy/step",
        )
        sample, trajectory = row.get("sample_id"), row.get("trajectory_index")
        _require(type(sample) is str and bool(sample.strip()), f"{label} sample_id missing")
        _require(type(trajectory) is int and trajectory >= 0, f"{label} trajectory_index invalid")
        _require(
            row.get("branch") == "cond" and row.get("timestep_unit") == "raw_scheduler",
            f"{label} branch/unit mismatch",
        )
        timestep = _number(row.get("timestep"), f"{label} timestep")
        _require(
            row.get("sde_type") == expected["sde_type"]
            and _number(row.get("noise_level"), f"{label} noise_level") == expected["noise_level"],
            f"{label} SDE/noise mismatch",
        )
        inputs = {name: _input(row.get(name), f"{label} {name}") for name in _INPUTS}
        dump_name = row.get("dump_path")
        _require(type(dump_name) is str and dump_name.endswith(".json"), f"{label} dump path invalid")
        _relative(root, dump_name)
        _require(dump_name in pinned and dump_name not in seen_dumps, f"{label} missing or reused pinned dump")
        _dump(pinned[dump_name], f"{label} dump")
        seen_dumps.add(dump_name)
        key = (sample, trajectory, "cond")
        slots = grouped.setdefault(key, {})
        _require(role not in slots, f"{label} duplicate {role} for {key}")
        slots[role] = {"timestep": timestep, "inputs": inputs, "dump_path": dump_name}
    result = []
    for key, slots in sorted(grouped.items()):
        _require(set(slots) == set(_PHASE), f"unpaired key {key}")
        actor, rollout = slots["actor"], slots["rollout"]
        _require(
            abs(actor["timestep"] - rollout["timestep"]) <= expected["timestep_atol"],
            f"raw scheduler timestep mismatch for {key}",
        )
        _require(actor["inputs"] == rollout["inputs"], f"input provenance mismatch for {key}")
        result.append({"key": list(key), "actor_dump": actor["dump_path"], "rollout_dump": rollout["dump_path"]})
    return result


def check(root: Path, manifest_name: str, pin_name: str, external_sha: str, output: Path) -> dict[str, Any]:
    """Check a pinned artifact and write an exclusive JSON verdict.

    Args:
        root: Existing input directory containing all artifacts.
        manifest_name: Root-relative manifest JSON path.
        pin_name: Root-relative SHA256 pin list path.
        external_sha: Independently recorded SHA256 of the pin list.
        output: New report JSON path outside ``root``.

    Returns:
        The report written to ``output``.

    Raises:
        ValueError: Artifact bytes or declared metadata violate the contract.
        RuntimeError: The numeric comparison is outside declared tolerances.
        MemoryError: Tensor loading exhausts available memory.
    """
    root = root.resolve(strict=True)
    _require(root.is_dir(), "root must be a directory")
    output = output.absolute()
    _require(".." not in output.parts, "output traversal forbidden")
    _require(not output.resolve().is_relative_to(root), "output must be outside input root")
    _require(output.parent.is_dir() and not output.exists() and not output.is_symlink(), "output must be fresh")
    pinned = _pins(root, pin_name, external_sha, manifest_name)
    manifest = _json(pinned[manifest_name], "manifest")
    _require(
        type(manifest) is dict and set(manifest) == {"version", "expected", "records", "comparison", "atol", "rtol"},
        "manifest keys invalid",
    )
    _require(type(manifest["version"]) is int and manifest["version"] == 1, "manifest version must be 1")
    expected = manifest["expected"]
    _require(
        type(expected) is dict
        and set(expected) == {"run_id", "policy_id", "global_step", "sde_type", "noise_level", "timestep_atol"},
        "expected keys invalid",
    )
    _require(type(expected["run_id"]) is str and bool(expected["run_id"]), "run_id missing")
    _require(
        type(expected["policy_id"]) is str and _HEX64.fullmatch(expected["policy_id"]) is not None,
        "policy_id invalid",
    )
    _require(type(expected["global_step"]) is int, "expected global_step must be integer")
    _require(type(expected["sde_type"]) is str and bool(expected["sde_type"]), "sde_type missing")
    expected["noise_level"] = _number(expected["noise_level"], "noise_level")
    expected["timestep_atol"] = _number(expected["timestep_atol"], "timestep_atol", nonnegative=True)
    atol = _number(manifest["atol"], "atol", nonnegative=True)
    rtol = _number(manifest["rtol"], "rtol", nonnegative=True)
    comparison = manifest["comparison"]
    _require(type(comparison) is dict and set(comparison) == {"path", "keys"}, "comparison keys invalid")
    comparison_path, row_keys = comparison["path"], comparison["keys"]
    _require(type(comparison_path) is str and comparison_path.endswith(".pt"), "comparison path must be .pt")
    _relative(root, comparison_path)
    _require(type(row_keys) is list and bool(row_keys), "comparison row keys missing")
    paired = _pairs(manifest["records"], expected, root, pinned)
    paired_keys = {tuple(item["key"]) for item in paired}
    keys = []
    for key in row_keys:
        _require(
            type(key) is list and len(key) == 3 and type(key[0]) is str and type(key[1]) is int and key[2] == "cond",
            "comparison row key invalid",
        )
        keys.append(tuple(key))
    _require(len(keys) == len(paired_keys) and set(keys) == paired_keys, "comparison keys are not a pair bijection")
    required = {manifest_name, comparison_path} | {
        item[role] for item in paired for role in ("actor_dump", "rollout_dump")
    }
    _require(set(pinned) == required, "pin list must cover exactly manifest, comparison and paired dumps")

    import torch

    try:
        tensors = torch.load(io.BytesIO(pinned[comparison_path]), map_location="cpu", weights_only=True)
    except MemoryError:
        raise
    except Exception as exc:
        raise ValueError("comparison tensor artifact cannot be decoded") from exc
    _require(
        type(tensors) is dict and set(tensors) == {"actor_log_probs", "rollout_log_probs"},
        "comparison tensor map invalid",
    )
    actor, rollout = tensors["actor_log_probs"], tensors["rollout_log_probs"]
    for name, value in (("actor_log_probs", actor), ("rollout_log_probs", rollout)):
        _require(
            type(value) is torch.Tensor
            and not value.requires_grad
            and value.device.type == "cpu"
            and value.layout == torch.strided
            and value.dtype == torch.float32
            and value.ndim == 2
            and value.shape[0] == len(keys)
            and value.shape[1] > 0
            and bool(torch.isfinite(value).all()),
            f"{name} must be a plain dense finite CPU float32 (N,T) tensor without gradients",
        )
    _require(actor.shape == rollout.shape, "log-probability tensor shapes differ")
    maxdiff = max(abs(float(a) - float(b)) for a, b in zip(actor.flatten(), rollout.flatten(), strict=True))
    close = bool(torch.allclose(actor.double(), rollout.double(), atol=atol, rtol=rtol))
    report = {
        "status": "admitted" if close else "numerical_mismatch",
        "declared_policy_id": expected["policy_id"],
        "pins_sha256": external_sha,
        "comparison_path": comparison_path,
        "row_keys": row_keys,
        "pair_count": len(paired),
        "max_abs_logprob_difference": maxdiff,
        "atol": atol,
        "rtol": rtol,
        "allclose": close,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not close:
        raise RuntimeError("diffusion consistency numerical mismatch; see report")
    return report


def main() -> None:
    """Parse paths and exit nonzero on any failed consistency gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pins", required=True)
    parser.add_argument("--pins-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.root, args.manifest, args.pins, args.pins_sha256, args.output)


if __name__ == "__main__":
    main()
