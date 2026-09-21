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
"""Internal file, safetensors and publication utilities (not plugin interfaces)."""

import ctypes
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MANIFEST_NAME = "merge_manifest.json"
WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
INDEX_NAME = WEIGHTS_NAME + ".index.json"


def read_json(path: Path) -> dict:
    """Read a JSON object, rejecting duplicate keys instead of silently overriding them."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return value


def write_json(path: Path, value: dict) -> None:
    """Write deterministic portable metadata."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def relative_path(value: str) -> Path:
    """Reject noncanonical, absolute or traversing artifact paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts) or path.as_posix() != value:
        raise ValueError(f"Unsafe artifact path: {value}")
    return Path(value)


def sha256_file(path: Path) -> str:
    """Hash file contents without materializing complete model files."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(files: dict[str, str]) -> str:
    """Fingerprint a sorted relative-path/checksum inventory."""
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inventory(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """Compute relative-path hashes for an explicitly selected set of input files."""
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)}


def tree_files(root: Path) -> list[Path]:
    """List local assets, ignoring Hub/Git administration and rejecting directory symlinks."""
    files = []
    for path in root.rglob("*"):
        if any(part in (".git", ".cache") for part in path.relative_to(root).parts):
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError("Directory symlinks are unsupported; resolve a local pipeline snapshot first")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"Not a regular asset: {path.name}")
    return sorted(files)


@contextmanager
def publication_directory(target: Path) -> Iterator[Path]:
    """Publish owned sibling staging atomically without replacing any existing target (Linux)."""
    # rename() can replace an empty directory created by a non-cooperating writer.
    # RENAME_NOREPLACE also closes that race; the lock serializes cooperating exporters.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace publishing requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    lock = target.parent / f".{target.name}.merge.lock"
    with lock.open("x"):
        staging = None
        try:
            if os.path.lexists(target):
                raise FileExistsError(target)
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-", dir=target.parent))
            yield staging
            if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(target))
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
            lock.unlink()


def tensor_spec(tensor: torch.Tensor) -> dict:
    """Describe a tensor without embedding its values in JSON."""
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def write_weights(
    root: Path,
    weights: Iterable[tuple[str, torch.Tensor]],
    budget: int,
    *,
    weights_name: str | None = None,
) -> dict[str, dict]:
    """Write bounded safetensors shards and verify every reopened tensor against its source."""
    weights_name = weights_name or WEIGHTS_NAME
    root.mkdir(exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Weight output directory must be empty")
    shards: list[tuple[Path, list[str]]] = []
    pending: dict[str, torch.Tensor] = {}
    specs = {}
    pending_bytes = total_bytes = 0

    def flush():
        if not pending:
            return
        path = root / f"part-{len(shards) + 1:05d}.safetensors"
        save_file(pending, str(path), metadata={"format": "pt"})
        with safe_open(path, framework="pt", device="cpu") as archive:
            if set(archive.keys()) != set(pending):
                raise ValueError("Written tensor inventory differs from the planned shard")
            for key, value in pending.items():
                actual = archive.get_tensor(key)
                if tensor_spec(actual) != tensor_spec(value) or not torch.equal(actual, value):
                    raise ValueError(f"Written artifact round-trip failed for {key}")
        shards.append((path, list(pending)))
        pending.clear()

    for key, tensor in weights:
        if key in specs:
            raise ValueError(f"Duplicate output tensor: {key}")
        size = tensor.numel() * tensor.element_size()
        if pending and pending_bytes + size > budget:
            flush()
            pending_bytes = 0
        pending[key] = tensor
        specs[key] = tensor_spec(tensor)
        pending_bytes += size
        total_bytes += size
        if pending_bytes >= budget:
            flush()
            pending_bytes = 0
    flush()
    if not specs:
        raise ValueError("Cannot publish an empty transformer")
    weight_map = {}
    for number, (path, keys) in enumerate(shards, 1):
        stem = weights_name.removesuffix(".safetensors")
        name = weights_name if len(shards) == 1 else f"{stem}-{number:05d}-of-{len(shards):05d}.safetensors"
        path.rename(root / name)
        weight_map.update(dict.fromkeys(keys, name))
    if len(shards) > 1:
        write_json(
            root / f"{weights_name}.index.json",
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
        )
    return specs


def weight_files(root: Path, weights_name: str | None = None) -> dict[str, Path]:
    """Read and check an unambiguous safetensors index or single-file checkpoint."""
    weights_name = weights_name or WEIGHTS_NAME
    single, index = root / weights_name, root / (weights_name + ".index.json")
    files = set(root.glob("*.safetensors"))
    if single.is_file() and index.exists():
        raise ValueError("Both single-file weights and a shard index are present")
    if index.is_file():
        mapping = read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Invalid safetensors weight_map")
        result = {}
        for key, value in mapping.items():
            path = relative_path(value)
            if len(path.parts) != 1 or path.suffix != ".safetensors":
                raise ValueError("Weight shards must be local safetensors files")
            result[key] = root / path
        if set(result.values()) != files:
            raise ValueError("Missing or unexpected safetensors shards")
    elif single.is_file() and files == {single}:
        with safe_open(single, framework="pt", device="cpu") as archive:
            result = dict.fromkeys(archive.keys(), single)
    else:
        raise ValueError("Expected standard Diffusers safetensors weights or shard index")
    observed = {}
    for path in sorted(set(result.values())):
        with safe_open(path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                if key in observed or result.get(key) != path:
                    raise ValueError("Duplicate, missing or incorrectly indexed tensor")
                observed[key] = path
    if observed != result:
        raise ValueError("Safetensors index does not match actual tensor keys")
    return result
