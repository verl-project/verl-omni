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

"""CPU fixtures for the standalone pinned diffusion consistency reader."""

import importlib.util
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_diffusion_consistency.py"
SPEC = importlib.util.spec_from_file_location("check_diffusion_consistency", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def _fixture(tmp_path, keys=None, delta=0.0):
    """Build a pinned comparison fixture with paired rollout and actor records."""
    keys = keys or [["sample-a", 0, "cond"]]
    root = tmp_path / "input"
    root.mkdir()
    records = []
    for key in keys:
        for role, phase in (("rollout", "rolloutgenerate"), ("actor", "actorlog_prob")):
            dump = f"dumps/{key[0]}-{key[1]}-{role}.json"
            file = root / dump
            file.parent.mkdir(exist_ok=True)
            file.write_text(json.dumps({"data": {"Module.block.0": {"output": []}}}))
            metadata = {"sha256": "c" * 64, "dtype": "torch.float32", "shape": [1, 2]}
            records.append(
                {
                    "run_id": "run-one",
                    "policy_id": "a" * 64,
                    "global_step": 1,
                    "sample_id": key[0],
                    "trajectory_index": key[1],
                    "branch": "cond",
                    "timestep_unit": "raw_scheduler",
                    "timestep": 812.5,
                    "noise_level": 0.7,
                    "sde_type": "sde",
                    "role": role,
                    "phase": phase,
                    "hidden_states": metadata,
                    "prompt_embeds": metadata,
                    "pooled_projections": metadata,
                    "dump_path": dump,
                }
            )
    manifest = {
        "version": 1,
        "expected": {
            "run_id": "run-one",
            "policy_id": "a" * 64,
            "global_step": 1,
            "sde_type": "sde",
            "noise_level": 0.7,
            "timestep_atol": 0.0,
        },
        "records": records,
        "comparison": {"path": "comparison.pt", "keys": keys},
        "atol": 1e-5,
        "rtol": 1e-3,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    actor = torch.zeros((len(keys), 2), dtype=torch.float32)
    actor[0, 0] = delta
    buffer = io.BytesIO()
    torch.save({"actor_log_probs": actor, "rollout_log_probs": torch.zeros_like(actor)}, buffer)
    (root / "comparison.pt").write_bytes(buffer.getvalue())
    return root, tmp_path / "report.json"


def _repin(root):
    """Rewrite the pin list after a fixture mutation and return its digest."""
    paths = ["manifest.json", "comparison.pt"] + sorted(
        str(path.relative_to(root)) for path in (root / "dumps").glob("*.json")
    )
    (root / "pins.sha256").write_text(
        "".join(f"{checker._digest((root / name).read_bytes())}  {name}\n" for name in paths)
    )
    return checker._digest((root / "pins.sha256").read_bytes())


def _check(root, output):
    """Check a fixture using its current pinned bytes."""
    return checker.check(root, "manifest.json", "pins.sha256", _repin(root), output)


def _manifest(root):
    """Read the fixture manifest for a targeted mutation."""
    return json.loads((root / "manifest.json").read_text())


def _save_manifest(root, manifest):
    """Save a mutated fixture manifest before repinning."""
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_admits_single_and_permuted_multiple_pairs(tmp_path):
    """Accept complete pairs regardless of record ordering."""
    root, output = _fixture(tmp_path)
    assert _check(root, output)["pair_count"] == 1
    second = tmp_path / "second"
    second.mkdir()
    keys = [["sample-b", 2, "cond"], ["sample-a", 0, "cond"]]
    root, output = _fixture(second, keys)
    manifest = _manifest(root)
    manifest["records"] = list(reversed(manifest["records"]))
    _save_manifest(root, manifest)
    report = _check(root, output)
    assert report["row_keys"] == keys and report["pair_count"] == 2


def test_relocates_root_with_relative_pins(tmp_path):
    """Accept a relocated artifact rooted at the same relative paths."""
    root, output = _fixture(tmp_path)
    digest = _repin(root)
    relocated = tmp_path / "relocated"
    root.rename(relocated)
    assert checker.check(relocated, "manifest.json", "pins.sha256", digest, output)["status"] == "admitted"


@pytest.mark.parametrize("damage", ["duplicate", "nonfinite", "boolean"])
def test_rejects_strict_manifest_json(tmp_path, damage):
    """Reject duplicate keys, nonfinite numbers, and boolean steps."""
    root, output = _fixture(tmp_path)
    if damage == "duplicate":
        (root / "manifest.json").write_text('{"version":1,"version":1}')
    elif damage == "nonfinite":
        (root / "manifest.json").write_text((root / "manifest.json").read_text().replace("0.7", "NaN"))
    else:
        manifest = _manifest(root)
        manifest["expected"]["global_step"] = True
        _save_manifest(root, manifest)
    with pytest.raises(ValueError):
        _check(root, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["wrong_external", "missing", "extra", "duplicate", "bad_line", "bad_hash"])
def test_rejects_pins_and_bytes_before_tensor_load(tmp_path, damage, monkeypatch):
    """Reject broken pin contracts before attempting tensor decode."""
    root, output = _fixture(tmp_path)
    digest = _repin(root)
    if damage == "wrong_external":
        digest = "0" * 64
    else:
        lines = (root / "pins.sha256").read_text().splitlines(keepends=True)
        if damage == "missing":
            lines.pop()
        elif damage == "extra":
            lines.append(f"{'0' * 64}  extra.json\n")
        elif damage == "duplicate":
            lines[-1] = lines[0]
        elif damage == "bad_line":
            lines[0] = lines[0].replace("  ", " ", 1)
        else:
            lines[0] = "0" * 64 + lines[0][64:]
        (root / "pins.sha256").write_text("".join(lines))
        digest = checker._digest((root / "pins.sha256").read_bytes())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("tensor loaded before pin validation"))
    with pytest.raises(ValueError):
        checker.check(root, "manifest.json", "pins.sha256", digest, output)


def test_rejects_coordinated_metadata_forgery_without_external_repin(tmp_path):
    """Reject a rewritten manifest and pin list without a new external digest."""
    root, output = _fixture(tmp_path)
    digest = _repin(root)
    manifest = _manifest(root)
    for record in manifest["records"]:
        record["hidden_states"]["sha256"] = "d" * 64
    _save_manifest(root, manifest)
    lines = (root / "pins.sha256").read_text().splitlines(keepends=True)
    lines[0] = f"{checker._digest((root / 'manifest.json').read_bytes())}  manifest.json\n"
    (root / "pins.sha256").write_text("".join(lines))
    with pytest.raises(ValueError, match="external pins SHA256"):
        checker.check(root, "manifest.json", "pins.sha256", digest, output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "rollout"),
        ("policy_id", "b" * 64),
        ("noise_level", 0.8),
        ("timestep", 999.0),
        ("hidden_states", {"sha256": "d" * 64, "dtype": "torch.float32", "shape": [1, 2]}),
    ],
)
def test_rejects_mismatched_pair_metadata(tmp_path, field, value):
    """Reject actor and rollout records with inconsistent provenance."""
    root, output = _fixture(tmp_path)
    manifest = _manifest(root)
    manifest["records"][1][field] = value
    _save_manifest(root, manifest)
    with pytest.raises(ValueError):
        _check(root, output)


@pytest.mark.parametrize("damage", ["duplicate", "unpaired", "wrong_rows", "duplicate_rows"])
def test_rejects_pair_cardinality_or_row_map(tmp_path, damage):
    """Reject incomplete pairs and comparison row mapping errors."""
    root, output = _fixture(tmp_path)
    manifest = _manifest(root)
    if damage == "duplicate":
        manifest["records"].append(manifest["records"][0])
    elif damage == "unpaired":
        manifest["records"].pop()
    elif damage == "wrong_rows":
        manifest["comparison"]["keys"] = [["wrong", 0, "cond"]]
    else:
        manifest["comparison"]["keys"].append(manifest["comparison"]["keys"][0])
    _save_manifest(root, manifest)
    with pytest.raises(ValueError):
        _check(root, output)


@pytest.mark.parametrize("role", ["actor", "rollout"])
@pytest.mark.parametrize("module_name", ["Module.", "Module.block.0"])
def test_rejects_empty_pinned_module_dump(tmp_path, role, module_name, monkeypatch):
    """Reject empty module objects before decoding comparison tensors."""
    root, output = _fixture(tmp_path)
    dump = root / "dumps" / f"sample-a-0-{role}.json"
    dump.write_text(json.dumps({"data": {module_name: {}}}))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("tensor loaded before dump validation"))
    with pytest.raises(ValueError, match="nonempty Module"):
        _check(root, output)
    assert not output.exists()


def test_accepts_dump_with_a_nonempty_module(tmp_path):
    """An empty entry does not hide another valid module in the dump."""
    root, output = _fixture(tmp_path)
    dump = root / "dumps" / "sample-a-0-actor.json"
    dump.write_text(json.dumps({"data": {"Module.empty": {}, "Module.block.0": {"output": []}}}))
    assert _check(root, output)["status"] == "admitted"


@pytest.mark.parametrize("role", ["actor", "rollout"])
@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_rejects_non_strided_pinned_tensors(tmp_path, role, layout):
    """Reject sparse artifacts without entering unsupported numeric operations."""
    root, output = _fixture(tmp_path)
    tensors = torch.load(root / "comparison.pt", map_location="cpu", weights_only=True)
    value = tensors[f"{role}_log_probs"]
    tensors[f"{role}_log_probs"] = value.to_sparse() if layout == "coo" else value.to_sparse_csr()
    torch.save(tensors, root / "comparison.pt")
    with pytest.raises(ValueError, match="plain dense finite CPU float32"):
        _check(root, output)
    assert not output.exists()


def test_accepts_noncontiguous_strided_tensors(tmp_path):
    """Dense tensor views remain valid without a contiguity requirement."""
    root, output = _fixture(tmp_path)
    tensors = {
        name: torch.zeros((2, 4), dtype=torch.float32)[:1, ::2] for name in ("actor_log_probs", "rollout_log_probs")
    }
    assert all(not value.is_contiguous() for value in tensors.values())
    torch.save(tensors, root / "comparison.pt")
    assert _check(root, output)["status"] == "admitted"


@pytest.mark.parametrize(
    "damage",
    ["map", "non_tensor", "parameter", "requires_grad", "dtype", "shape", "rows", "nonfinite", "numeric", "extreme"],
)
def test_rejects_invalid_or_out_of_tolerance_tensors(tmp_path, damage):
    """Separate invalid tensor artifacts from reported numeric mismatches."""
    root, output = _fixture(tmp_path)
    tensors = torch.load(root / "comparison.pt", map_location="cpu", weights_only=True)
    if damage == "map":
        tensors["extra"] = torch.zeros((1, 2))
    elif damage == "non_tensor":
        tensors["actor_log_probs"] = [[0.0, 0.0]]
    elif damage == "parameter":
        tensors["actor_log_probs"] = torch.nn.Parameter(tensors["actor_log_probs"], requires_grad=False)
    elif damage == "requires_grad":
        tensors["actor_log_probs"].requires_grad_(True)
    elif damage == "dtype":
        tensors["actor_log_probs"] = tensors["actor_log_probs"].double()
    elif damage == "shape":
        tensors["actor_log_probs"] = torch.zeros((1, 3))
    elif damage == "rows":
        tensors["actor_log_probs"] = torch.zeros((2, 2))
    elif damage == "nonfinite":
        tensors["actor_log_probs"][0, 0] = float("nan")
    elif damage == "extreme":
        tensors["actor_log_probs"][0, 0] = torch.finfo(torch.float32).max
        tensors["rollout_log_probs"][0, 0] = -torch.finfo(torch.float32).max
    else:
        tensors["actor_log_probs"][0, 0] = 1.0
    torch.save(tensors, root / "comparison.pt")
    if damage in ("numeric", "extreme"):
        with pytest.raises(RuntimeError, match="numerical mismatch"):
            _check(root, output)
        report = json.loads(output.read_text())
        assert report["status"] == "numerical_mismatch" and report["allclose"] is False
        assert report["max_abs_logprob_difference"] > 0
    else:
        with pytest.raises(ValueError):
            _check(root, output)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("tolerance", ["below_rtol", "at_rtol", "above_rtol", "at_atol"])
def test_extreme_finite_tolerance_boundary(tmp_path, sign, tolerance):
    """Compare extreme finite inputs without overflowing the tolerance decision."""
    root, output = _fixture(tmp_path)
    limit = float(torch.finfo(torch.float32).max)
    tensors = torch.load(root / "comparison.pt", map_location="cpu", weights_only=True)
    tensors["actor_log_probs"][0, 0] = sign * limit
    tensors["rollout_log_probs"][0, 0] = -sign * limit
    torch.save(tensors, root / "comparison.pt")
    manifest = _manifest(root)
    manifest["atol"] = 2.0 * limit if tolerance == "at_atol" else 0.0
    manifest["rtol"] = {
        "below_rtol": math.nextafter(2.0, 0.0),
        "at_rtol": 2.0,
        "above_rtol": math.nextafter(2.0, math.inf),
        "at_atol": 0.0,
    }[tolerance]
    _save_manifest(root, manifest)
    if tolerance == "below_rtol":
        with pytest.raises(RuntimeError, match="numerical mismatch"):
            _check(root, output)
    else:
        assert _check(root, output)["status"] == "admitted"
    report = json.loads(output.read_text())
    assert report["allclose"] is (tolerance != "below_rtol")
    assert report["max_abs_logprob_difference"] == 2.0 * limit
    assert report["atol"] == manifest["atol"] and report["rtol"] == manifest["rtol"]


@pytest.mark.parametrize(("actor", "rollout", "accepted"), [(1.0, 2.0, True), (2.0, 1.0, False)])
def test_preserves_rollout_relative_tolerance(tmp_path, actor, rollout, accepted):
    """Keep the rollout tensor as the asymmetric relative-tolerance reference."""
    root, output = _fixture(tmp_path)
    tensors = torch.load(root / "comparison.pt", map_location="cpu", weights_only=True)
    tensors["actor_log_probs"][0, 0] = actor
    tensors["rollout_log_probs"][0, 0] = rollout
    torch.save(tensors, root / "comparison.pt")
    manifest = _manifest(root)
    manifest.update(atol=0.0, rtol=0.5)
    _save_manifest(root, manifest)
    if accepted:
        assert _check(root, output)["status"] == "admitted"
    else:
        with pytest.raises(RuntimeError, match="numerical mismatch"):
            _check(root, output)
    report = json.loads(output.read_text())
    assert report["allclose"] is accepted and report["max_abs_logprob_difference"] == 1.0


@pytest.mark.parametrize("damage", ["empty", "garbage", "truncated_zip", "unsafe_object"])
def test_rejects_undecodable_pinned_comparison(tmp_path, damage):
    """Treat validly pinned but undecodable tensor bytes as invalid artifacts."""
    root, output = _fixture(tmp_path)
    comparison = root / "comparison.pt"
    if damage == "empty":
        comparison.write_bytes(b"")
    elif damage == "garbage":
        comparison.write_bytes(b"not a torch archive")
    elif damage == "truncated_zip":
        comparison.write_bytes(comparison.read_bytes()[:32])
    else:
        buffer = io.BytesIO()
        torch.save(Path("safe-test-path"), buffer)
        comparison.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="comparison tensor artifact cannot be decoded") as raised:
        _check(root, output)
    assert raised.value.__cause__ is not None
    assert not output.exists()


def test_preserves_tensor_load_memory_error(tmp_path, monkeypatch):
    """Leave resource failures outside the invalid-artifact contract."""
    root, output = _fixture(tmp_path)

    def fail_load(*args, **kwargs):
        """Simulate resource exhaustion at the tensor decode boundary."""
        raise MemoryError("tensor load exhausted memory")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(MemoryError, match="tensor load exhausted memory"):
        _check(root, output)
    assert not output.exists()


def test_output_exclusivity_and_input_path_escape(tmp_path):
    """Require a fresh external report path and confined input paths."""
    root, output = _fixture(tmp_path)
    _check(root, output)
    with pytest.raises(ValueError, match="fresh"):
        _check(root, output)
    symlink = tmp_path / "report-link.json"
    symlink.symlink_to(root / "manifest.json")
    with pytest.raises(ValueError, match="outside input root"):
        _check(root, symlink)
    with pytest.raises(ValueError, match="outside input root"):
        _check(root, root / "report.json")
    with pytest.raises(ValueError, match="traversal"):
        _check(root, root / ".." / "traversal.json")
    manifest = _manifest(root)
    manifest["comparison"]["path"] = "../outside.pt"
    _save_manifest(root, manifest)
    with pytest.raises(ValueError, match="relative path"):
        _check(root, tmp_path / "fresh.json")


def test_rejects_symlinked_pinned_input(tmp_path):
    """Reject a pinned file that resolves through a symlink."""
    root, output = _fixture(tmp_path)
    dump = root / "dumps" / "sample-a-0-actor.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(dump.read_bytes())
    dump.unlink()
    dump.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        _check(root, output)


def test_cli_success_and_numeric_failure(tmp_path):
    """Preserve CLI exit behavior for admitted and mismatched artifacts."""
    root, output = _fixture(tmp_path)
    digest = _repin(root)
    base = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(root),
        "--manifest",
        "manifest.json",
        "--pins",
        "pins.sha256",
        "--pins-sha256",
        digest,
        "--output",
        str(output),
    ]
    assert subprocess.run(base, check=False, capture_output=True).returncode == 0
    second = tmp_path / "second"
    second.mkdir()
    root, output = _fixture(second, delta=1.0)
    digest = _repin(root)
    base[base.index("--root") + 1] = str(root)
    base[base.index("--pins-sha256") + 1] = digest
    base[base.index("--output") + 1] = str(output)
    assert subprocess.run(base, check=False, capture_output=True).returncode != 0
    assert json.loads(output.read_text())["status"] == "numerical_mismatch"
