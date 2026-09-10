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
"""Create a MiniCPM-o student checkpoint with deterministic relative-L2 noise."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TENSOR_PREFIX = "llm."


def _tensor_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def add_relative_l2_noise(tensor: torch.Tensor, ratio: float, seed: int, name: str) -> torch.Tensor:
    """Add Gaussian noise whose L2 norm is ``ratio`` times the tensor L2 norm."""
    if not tensor.is_floating_point() or tensor.numel() == 0:
        return tensor.clone()
    weight = tensor.float()
    weight_norm = torch.linalg.vector_norm(weight)
    if not torch.isfinite(weight_norm):
        raise ValueError(f"Non-finite weight norm for {name}.")
    if weight_norm == 0:
        return tensor.clone()
    generator = torch.Generator(device="cpu").manual_seed(_tensor_seed(seed, name))
    noise = torch.randn(weight.shape, dtype=torch.float32, generator=generator)
    target_norm = ratio * weight_norm
    noise.mul_(target_norm / torch.linalg.vector_norm(noise))
    output = (weight + noise).to(tensor.dtype)
    for _ in range(3):
        stored_noise_norm = torch.linalg.vector_norm(output.float() - weight)
        if stored_noise_norm == 0:
            raise ValueError(f"Noise vanished after dtype conversion for {name}.")
        noise.mul_(target_norm / stored_noise_norm)
        output = (weight + noise).to(tensor.dtype)
    return output


def _load_index(source: Path) -> tuple[Path, dict]:
    indexes = sorted(source.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError(f"Expected one safetensors index in {source}, found {len(indexes)}.")
    index = json.loads(indexes[0].read_text())
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError(f"Invalid weight_map in {indexes[0]}.")
    return indexes[0], index


def create_noised_checkpoint(source: Path, destination: Path, ratio: float, seed: int) -> dict:
    """Copy a checkpoint and perturb only floating-point ``llm.*`` tensors."""
    source = source.resolve()
    destination = destination.resolve()
    if not 0 < ratio < 1:
        raise ValueError("noise ratio must be between 0 and 1.")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if source == destination or source in destination.parents:
        raise ValueError("output model must not be inside the source model directory.")

    _, index = _load_index(source)
    shard_names = sorted(set(index["weight_map"].values()))
    missing = [name for name in shard_names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)

    def ignore_shards(path: str, names: list[str]) -> list[str]:
        if Path(path).resolve() != source:
            return []
        return [name for name in names if name in shard_names]

    perturbed = 0
    unchanged = 0
    try:
        shutil.copytree(source, temporary, symlinks=True, ignore=ignore_shards)
        for shard_name in shard_names:
            source_shard = source / shard_name
            tensors = {}
            with safe_open(source_shard, framework="pt", device="cpu") as shard:
                metadata = shard.metadata()
                for name in shard.keys():
                    tensor = shard.get_tensor(name)
                    if name.startswith(TENSOR_PREFIX) and tensor.is_floating_point() and torch.count_nonzero(tensor):
                        tensor = add_relative_l2_noise(tensor, ratio, seed, name)
                        perturbed += 1
                    else:
                        tensor = tensor.clone()
                        unchanged += 1
                    tensors[name] = tensor
            save_file(tensors, temporary / shard_name, metadata=metadata)
        manifest = {
            "format": "minicpm-o-relative-l2-noise-v1",
            "source_model": source.name,
            "tensor_prefix": TENSOR_PREFIX,
            "noise_distribution": "gaussian_direction",
            "relative_l2_ratio": ratio,
            "seed": seed,
            "perturbed_tensors": perturbed,
            "unchanged_tensors": unchanged,
        }
        (temporary / "noise_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    """Parse arguments and create the noised student checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--noise-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = create_noised_checkpoint(args.source_model, args.output_model, args.noise_ratio, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
