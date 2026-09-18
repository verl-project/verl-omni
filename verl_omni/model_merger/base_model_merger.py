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
"""Common model-merger configuration, CLI arguments, and lifecycle."""

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelMergerConfig:
    """Configuration for offline diffusion model merger operations.

    Args:
        operation: Operation type: ``merge`` or ``test``.
        backend: Checkpoint backend. This implementation currently supports only ``fsdp``.
        target_dir: Directory for a published model. Defaults to ``tmp`` for merge.
        hf_upload_path: Optional Hugging Face repository ID for the published artifact.
        private: Whether a newly created Hugging Face repository is private.
        test_hf_dir: Published artifact directory checked by the ``test`` operation.
        trust_remote_code: Permit audited Python assets from a local base model.
        local_dir: Directory containing saved rank-local model checkpoints.
        hf_model_config_path: Hugging Face actor config directory. Defaults to
            ``<local_dir>/huggingface``.
        hf_upload: Whether upload is enabled. Computed from operation and hf_upload_path.
        base_model: Compatible pretrained component or complete pipeline used for packaging.
        output_format: Publish a complete ``pipeline`` or standalone ``transformer``.
        dtype: Output tensor dtype or ``preserve``.
        max_shard_size: Maximum pending output safetensors shard size in bytes.
        trust_checkpoint: Acknowledge that rank checkpoints are trusted pickle inputs.
    """

    operation: str
    backend: str
    target_dir: str | None = "tmp"
    hf_upload_path: str | None = None
    private: bool = False
    test_hf_dir: str | None = None
    trust_remote_code: bool = False
    local_dir: str | None = None
    hf_model_config_path: str | None = None
    hf_upload: bool = field(init=False)
    base_model: str | None = None
    output_format: str = "pipeline"
    dtype: str = "preserve"
    max_shard_size: int = 2 * 1024**3
    trust_checkpoint: bool = False

    def __post_init__(self):
        if self.operation not in {"merge", "test"}:
            raise ValueError("operation must be merge or test")
        if self.backend != "fsdp":
            raise ValueError("Only the fsdp checkpoint backend is supported")
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            if not self.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False
            return
        if not self.local_dir:
            raise ValueError("local_dir must be provided for merge operation")
        if not self.base_model:
            raise ValueError("base_model must be provided for merge operation")
        if not self.target_dir:
            raise ValueError("target_dir must be provided for merge operation")
        if self.hf_model_config_path is None:
            self.hf_model_config_path = str(Path(self.local_dir) / "huggingface")
        if self.output_format not in {"pipeline", "transformer"}:
            raise ValueError("output_format must be pipeline or transformer")
        if self.dtype not in {"preserve", "float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported output dtype: {self.dtype}")
        if type(self.max_shard_size) is not int or self.max_shard_size <= 0:
            raise ValueError("max_shard_size must be a positive byte count")
        if self.trust_checkpoint is not True:
            raise ValueError("Pickled rank checkpoints require explicit trust_checkpoint=True / --trust-checkpoint")


@dataclass(frozen=True)
class MergeResult:
    """Published artifact locations; detailed verification is recorded in the manifest."""

    output_dir: Path
    manifest_path: Path


class BaseModelMerger(ABC):
    """Common merge/test and cleanup lifecycle without architecture-specific reconstruction."""

    def __init__(self, config: ModelMergerConfig):
        self.config = config

    @abstractmethod
    def merge_and_save(self) -> MergeResult | dict[str, Any]:
        """Merge and publish a model, or test a previously published artifact."""
        raise NotImplementedError

    def upload_to_huggingface(self, output_dir: Path) -> None:
        """Upload a successfully published artifact when requested."""
        if not self.config.hf_upload or not self.config.hf_upload_path:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=self.config.hf_upload_path,
            repo_type="model",
        )

    def cleanup(self) -> None:
        """Release merger resources; publication transactions own their staging paths."""
        return None


def parse_args() -> argparse.Namespace:
    """Parse verl-style shared merge and test operations."""
    parser = argparse.ArgumentParser(description="verl-omni model merger")
    commands = parser.add_subparsers(dest="operation", required=True, help="Specify merge or test operation")
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--backend", required=True, choices=["fsdp"], help="Checkpoint backend")
    base.add_argument("--local_dir", default=None, help="Path to saved model checkpoints")
    base.add_argument(
        "--hf_model_config_path",
        default=None,
        help="Actor Hugging Face config directory; defaults to <local_dir>/huggingface",
    )
    base.add_argument("--trust-remote-code", action="store_true", help="Whether to trust audited local model code")

    merge = commands.add_parser("merge", parents=[base], help="Merge model checkpoints and publish an artifact")
    merge.add_argument("--target_dir", default="tmp", help="Directory for the published model")
    merge.add_argument(
        "--base_model",
        "--base-model",
        dest="base_model",
        required=True,
        help="Compatible component or complete base pipeline",
    )
    merge.add_argument("--hf_upload_path", default=None, help="Optional Hugging Face repository ID")
    merge.add_argument("--private", action="store_true", help="Create a private Hugging Face repository")
    merge.add_argument(
        "--output_format",
        "--output-format",
        dest="output_format",
        choices=["pipeline", "transformer"],
        default="pipeline",
    )
    merge.add_argument("--dtype", choices=["preserve", "float32", "float16", "bfloat16"], default="preserve")
    merge.add_argument(
        "--max_shard_size",
        "--max-shard-size",
        dest="max_shard_size",
        type=int,
        default=2 * 1024**3,
        help="Output shard budget in bytes",
    )
    merge.add_argument("--trust-checkpoint", action="store_true", help="Acknowledge trusted pickle inputs")

    test = commands.add_parser("test", parents=[base], help="Test a published artifact")
    test.add_argument("--test_hf_dir", required=True, help="Published artifact directory to test")
    return parser.parse_args()


def generate_config_from_args(args: argparse.Namespace) -> ModelMergerConfig:
    """Build a model merger config from shared and operation-specific arguments."""
    common = {
        "operation": args.operation,
        "backend": args.backend,
        "trust_remote_code": args.trust_remote_code,
        "local_dir": args.local_dir,
        "hf_model_config_path": args.hf_model_config_path,
    }
    if args.operation == "merge":
        return ModelMergerConfig(
            **common,
            target_dir=args.target_dir,
            hf_upload_path=args.hf_upload_path,
            private=args.private,
            test_hf_dir=None,
            base_model=args.base_model,
            output_format=args.output_format,
            dtype=args.dtype,
            max_shard_size=args.max_shard_size,
            trust_checkpoint=args.trust_checkpoint,
        )
    if args.operation == "test":
        return ModelMergerConfig(
            **common,
            target_dir=None,
            hf_upload_path=None,
            private=False,
            test_hf_dir=args.test_hf_dir,
        )
    raise NotImplementedError(f"Unknown operation: {args.operation}")


def create_model_merger(config: ModelMergerConfig) -> BaseModelMerger:
    """Construct the selected backend merger."""
    if config.backend == "fsdp":
        from .fsdp_model_merger import FSDPModelMerger

        return FSDPModelMerger(config)
    raise NotImplementedError(f"Unknown backend: {config.backend}")


def run_model_merger(config: ModelMergerConfig) -> MergeResult | dict[str, Any]:
    """Run one merger operation through the common cleanup lifecycle."""
    merger = create_model_merger(config)
    try:
        return merger.merge_and_save()
    finally:
        merger.cleanup()


def merge_model(config: ModelMergerConfig) -> MergeResult:
    """Publish a model through the common merger lifecycle."""
    if config.operation != "merge":
        raise ValueError("merge_model requires operation='merge'")
    result = run_model_merger(config)
    if not isinstance(result, MergeResult):  # pragma: no cover - guarded by the operation.
        raise TypeError("merge operation did not return MergeResult")
    return result
