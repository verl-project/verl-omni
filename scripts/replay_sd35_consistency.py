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

"""Replay a pinned, existing two-prompt SD3.5 producer pair on CPU.

Replay compares all saved log probabilities and checks MSProbe dumps for the
first sample and scheduler step captured by each role.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import check_diffusion_consistency as checker

INPUTS = ("hidden_states", "prompt_embeds", "pooled_projections")
FILES = ("rollout/manifest.json", "rollout/actor_batch.pt", "actor/comparison.json", "actor/comparison.pt")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def normalize(record: object, original: Path, dump: str) -> dict:
    require(type(record) is dict, "capture record must be an object")
    nested = record.get("inputs")
    if nested is None:
        nested = {name: record.get(name) for name in INPUTS}
    require(
        type(nested) is dict and set(nested) == set(INPUTS) and all(type(value) is dict for value in nested.values()),
        "capture inputs invalid",
    )
    for name in INPUTS:
        require(name not in record or record[name] == nested[name], f"conflicting {name}")
    old = record.get("dump_path")
    require(type(old) is str, "capture dump path missing")
    old_path = Path(old)
    require(
        (old_path if old_path.is_absolute() else original / old_path) == original / dump, "capture dump path mismatch"
    )
    return {**record, "inputs": nested, **nested, "dump_path": dump}


def source_pins(root: Path, pin_file: Path, sha: str) -> dict[str, bytes]:
    """Read an external pin list with root-relative entries into verified bytes."""
    require(checker._HEX64.fullmatch(sha) is not None, "invalid independent pin digest")
    pin_file = pin_file.absolute()
    require(not pin_file.resolve().is_relative_to(root), "pin file must be outside capture root")
    require(
        pin_file.resolve(strict=True) == pin_file and pin_file.is_file(), "pin file must be regular and unsymlinked"
    )
    lines = pin_file.read_bytes()
    require(checker._digest(lines) == sha, "independent pin digest mismatch")
    source = {}
    for line in lines.decode("utf-8").splitlines(keepends=True):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n\x00]+)\n", line)
        require(match is not None, "invalid pin line")
        expected, name = match.groups()
        require(name not in source, "duplicate pin entry")
        data = checker._relative(root, name).read_bytes()
        require(checker._digest(data) == expected, f"pinned byte mismatch: {name}")
        source[name] = data
    require(len(source) == 8, "expected exactly eight producer pins")
    return source


def replay(root: Path, pin_name: str, pin_sha: str, output: Path) -> dict:
    root = root.resolve(strict=True)
    require(root.is_dir(), "capture root must be a directory")
    output = output.absolute()
    require(
        ".." not in output.parts and not output.resolve().is_relative_to(root), "output must be outside capture root"
    )
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink(), "output must be fresh")
    # Bind source bytes to the independently recorded pin-list digest before torch.load.
    source = source_pins(root, Path(pin_name), pin_sha)
    require(set(FILES).issubset(source), "producer files missing from pins")
    raw_paths = {}
    dump_paths = {}
    for role in ("rollout", "actor"):
        prefix = f"{role}/captures/{role}/"
        matches = [name for name in source if name.startswith(prefix) and name.endswith("/record.json")]
        require(len(matches) == 1 and len(Path(matches[0]).parts) == 5, f"expected one raw {role} record")
        raw_paths[role] = matches[0]
        dump = matches[0].removesuffix("record.json") + "dump/step0/rank0/dump.json"
        require(dump in source, f"missing {role} dump")
        dump_paths[role] = dump
    require(set(source) == set(FILES) | set(raw_paths.values()) | set(dump_paths.values()), "unexpected producer pins")
    rollout = checker._json(source[FILES[0]], FILES[0])
    actor = checker._json(source[FILES[2]], FILES[2])
    require(type(rollout) is dict and type(actor) is dict, "producer metadata invalid")
    require(
        rollout.get("role") == "rollout" and rollout.get("real_prompt") is True and actor.get("role") == "actor",
        "producer roles invalid",
    )
    identity = rollout.get("identity")
    require(
        type(identity) is dict and bool(identity) and identity == actor.get("identity"), "producer identity mismatch"
    )
    require(rollout.get("provenance") == actor.get("provenance"), "producer provenance mismatch")
    require(
        rollout.get("shared_run_root") == actor.get("shared_run_root")
        and rollout.get("rollout_root") == actor.get("rollout_root"),
        "producer location mismatch",
    )
    original_name = rollout.get("shared_run_root")
    require(type(original_name) is str and Path(original_name).is_absolute(), "producer original root invalid")
    original = Path(original_name)
    require(
        ".." not in original.parts and "." not in original.parts and str(original) == original_name and original.name,
        "producer original root invalid",
    )
    require(rollout.get("rollout_root") == str(original / "rollout"), "producer rollout root invalid")
    provenance = rollout.get("provenance")
    require(
        type(provenance) is dict and provenance.get("transformer_weight_hash_verified") is True,
        "producer weight attestation missing",
    )
    require(
        type(provenance.get("source_head")) is str
        and re.fullmatch(r"[0-9a-f]{40}", provenance["source_head"]) is not None,
        "producer source head invalid",
    )
    for field in ("source_manifest_sha256", "probe_sha256", "transformer_weight_sha256"):
        value = provenance.get(field)
        require(type(value) is str and checker._HEX64.fullmatch(value) is not None, f"producer {field} invalid")
    policy = provenance.get("transformer_weight_sha256")
    require(type(policy) is str and checker._HEX64.fullmatch(policy) is not None, "producer policy invalid")
    require(
        checker._digest(source[FILES[1]]) == rollout.get("actor_batch_sha256") == actor.get("rollout_batch_sha256"),
        "producer batch digest mismatch",
    )
    require(checker._digest(source[FILES[3]]) == actor.get("comparison_sha256"), "producer comparison digest mismatch")
    records = []
    for role, owner in (("rollout", rollout), ("actor", actor)):
        raw = normalize(checker._json(source[raw_paths[role]], raw_paths[role]), original, dump_paths[role])
        stored = normalize(owner.get("capture_record"), original, dump_paths[role])
        require(raw == stored, f"{role} stored record differs from raw capture")
        require(
            raw.get("role") == role and raw.get("run_id") == original.name and raw.get("policy_id") == policy,
            f"{role} capture identity invalid",
        )
        require(raw.get("trajectory_index") == 0 and raw.get("global_step") == 1, f"{role} capture index/step invalid")
        records.append(raw)
    first = records[0]
    require(
        records[1].get("sample_id") == first.get("sample_id")
        and records[1].get("timestep_input") == first.get("timestep_input"),
        "capture sample/timestep input mismatch",
    )
    sample = first.get("sample_id")
    mapping = rollout.get("request_map")
    require(
        type(mapping) is list and len(mapping) == 2 and all(type(row) is dict for row in mapping),
        "request map must contain two entries",
    )
    expected_map = [
        {
            "request_id": f"{original.name}:sd35-prompt-0",
            "sample_id": "sd35-prompt-0",
            "prompt": "A red cube on a white table.",
            "seed": 1234,
            "api_output_index": 0,
        },
        {
            "request_id": f"{original.name}:sd35-prompt-1",
            "sample_id": "sd35-prompt-1",
            "prompt": "A blue sphere on a white table.",
            "seed": 1235,
            "api_output_index": 1,
        },
    ]
    require(mapping == expected_map, "producer request map differs from fixed two-prompt recipe")
    require(sample == mapping[0]["sample_id"], "raw capture is not the first request sample")
    atol, rtol = 1e-5, 1e-3
    require(
        actor.get("allclose_atol") == atol and actor.get("allclose_rtol") == rtol,
        "producer tolerance differs from fixed recipe",
    )

    import torch

    require(not torch.cuda.is_initialized(), "CPU replay initialized CUDA")
    batch = torch.load(io.BytesIO(source[FILES[1]]), map_location="cpu", weights_only=True)
    tensors = torch.load(io.BytesIO(source[FILES[3]]), map_location="cpu", weights_only=True)
    require(
        type(batch) is dict and type(tensors) is dict and set(tensors) == {"actor_log_probs", "rollout_log_probs"},
        "producer tensor map invalid",
    )
    for value in (batch.get("old_log_probs"), *tensors.values()):
        require(
            type(value) is torch.Tensor
            and value.device.type == "cpu"
            and value.layout == torch.strided
            and value.dtype == torch.float32
            and tuple(value.shape) == (2, 2)
            and not value.requires_grad
            and bool(torch.isfinite(value).all()),
            "producer tensor must be finite CPU float32 2x2",
        )
    require(torch.equal(batch["old_log_probs"], tensors["rollout_log_probs"]), "saved rollout probabilities differ")
    actor_probs, rollout_probs = tensors["actor_log_probs"], tensors["rollout_log_probs"]
    full_close = torch.allclose(actor_probs, rollout_probs, atol=atol, rtol=rtol)
    max_diff = float((actor_probs.double() - rollout_probs.double()).abs().max())
    row = mapping[0]["api_output_index"]
    selected = {name: value[row : row + 1, :1].clone() for name, value in tensors.items()}
    manifest = {
        "version": 1,
        "expected": {
            "run_id": original.name,
            "policy_id": policy,
            "global_step": 1,
            "sde_type": first.get("sde_type"),
            "noise_level": first.get("noise_level"),
            "timestep_atol": 0.0,
        },
        "records": records,
        "comparison": {"path": "comparison.pt", "keys": [[sample, 0, "cond"]]},
        "atol": atol,
        "rtol": rtol,
    }
    output.mkdir(exist_ok=False)
    case = output / "case"
    case.mkdir()
    (case / "manifest.json").write_bytes(encoded(manifest))
    buffer = io.BytesIO()
    torch.save(selected, buffer)
    (case / "comparison.pt").write_bytes(buffer.getvalue())
    for dump in dump_paths.values():
        dest = case / dump
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source[dump])
    names = ["manifest.json", "comparison.pt", *dump_paths.values()]
    pins = "".join(f"{checker._digest((case / name).read_bytes())}  {name}\n" for name in names).encode()
    (case / "pins.sha256").write_bytes(pins)
    mismatch = None
    try:
        report = checker.check(case, "manifest.json", "pins.sha256", checker._digest(pins), output / "report.json")
    except RuntimeError as exc:
        report_path = output / "report.json"
        if str(exc) != "diffusion consistency numerical mismatch; see report" or not report_path.is_file():
            raise
        report = checker._json(report_path.read_bytes(), "report.json")
        require(
            type(report) is dict and report.get("status") == "numerical_mismatch", "checker mismatch report invalid"
        )
        mismatch = exc
    audit = {
        "status": "admitted" if full_close and mismatch is None else "numerical_mismatch",
        "source_pins_sha256": pin_sha,
        "producer_identity": identity,
        "producer_provenance": provenance,
        "sample_id": sample,
        "api_output_index": row,
        "tensor_step": 0,
        "scope": (
            "all saved log probabilities; MSProbe dumps for the first sample and step only; "
            "producer provenance is pinned metadata, not fresh source or model validation"
        ),
        "full_comparison": {
            "allclose": full_close,
            "max_abs_logprob_difference": max_diff,
            "atol": atol,
            "rtol": rtol,
        },
        "checker_report": report,
    }
    (output / "audit.json").write_bytes(encoded(audit))
    require(not torch.cuda.is_initialized(), "CPU replay initialized CUDA")
    if mismatch is not None:
        raise mismatch
    if not full_close:
        raise RuntimeError("full actor/rollout log-probability mismatch; see audit.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pins", required=True, help="source pin-list file with root-relative entries")
    parser.add_argument("--pins-sha256", required=True, help="independently recorded source pin-list digest")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory outside capture root")
    args = parser.parse_args()
    replay(args.root, args.pins, args.pins_sha256, args.output)


if __name__ == "__main__":
    main()
