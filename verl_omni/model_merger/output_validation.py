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
"""Portable validation for published Diffusers model artifacts."""

from pathlib import Path

from safetensors import safe_open

from .architectures import _PIPELINES, _TRANSFORMERS
from .utils import (
    MANIFEST_NAME,
    inventory,
    read_json,
    relative_path,
    tensor_spec,
    tree_files,
    weight_files,
)


def validate_artifact(target: str | Path) -> dict:
    """Check portable output hashes, indexes and tensor metadata without source checkpoints."""
    root = Path(target)
    manifest = read_json(root / MANIFEST_NAME)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("artifact_type")
        not in {
            "diffusers_pipeline",
            "diffusers_transformer",
            "minimax_h3_pipeline",
        }
        or manifest.get("architecture") not in _TRANSFORMERS
    ):
        raise ValueError("Unsupported merge manifest")
    component = manifest.get("trained_components")
    directory = manifest.get("tensor_directory")
    pipeline = manifest["artifact_type"] in {"diffusers_pipeline", "minimax_h3_pipeline"}
    native_h3 = manifest["artifact_type"] == "minimax_h3_pipeline"
    if component != ["transformer"]:
        raise ValueError("Invalid trained component")
    if directory != ("transformer" if pipeline else "."):
        raise ValueError("Invalid tensor directory")
    if pipeline and manifest["architecture"] not in _PIPELINES:
        raise ValueError("Unsupported pipeline artifact")
    if native_h3 != (manifest["architecture"] == "MiniMaxH3Pipeline" and pipeline):
        raise ValueError("Invalid native MiniMax H3 artifact")
    config = read_json(root / directory / "config.json")
    expected_class = "MiniMaxH3DiTModel" if native_h3 else _TRANSFORMERS[manifest["architecture"]]
    if config.get("_class_name") != expected_class:
        raise ValueError("Transformer config conflicts with manifest")
    if pipeline and read_json(root / "model_index.json").get("_class_name") != manifest["architecture"]:
        raise ValueError("Pipeline config conflicts with manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or MANIFEST_NAME in files:
        raise ValueError("Invalid output inventory")
    for name, digest in files.items():
        relative_path(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid SHA256 digest")
    actual_files = tree_files(root)
    if any(p.is_symlink() for p in actual_files) or set(files) != {
        p.relative_to(root).as_posix() for p in actual_files if p.relative_to(root).as_posix() != MANIFEST_NAME
    }:
        raise ValueError("Output file inventory mismatch or external symlink")
    if inventory(root, [root / name for name in files]) != files:
        raise ValueError("Output checksum mismatch")
    mapping = weight_files(root / directory, "model.safetensors" if native_h3 else None)
    if set(mapping) != set(manifest.get("tensors", {})):
        raise ValueError("Output tensor inventory mismatch")
    for path in set(mapping.values()):
        with safe_open(path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                if tensor_spec(archive.get_tensor(key)) != manifest["tensors"][key]:
                    raise ValueError(f"Output tensor metadata mismatch: {key}")
    return manifest
