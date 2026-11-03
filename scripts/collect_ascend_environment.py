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
"""Collect package and source versions without importing accelerator runtimes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def collect_packages() -> dict:
    """Use the default metadata resolver and retain conflicting dist-info versions."""
    versions = defaultdict(list)
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name].append(distribution.version)
    return {
        "selected": {name: importlib.metadata.version(name) for name in sorted(versions)},
        "duplicate_metadata": {name: values for name, values in sorted(versions.items()) if len(values) > 1},
        "method": "importlib.metadata.version; runtime build identifiers can differ from distribution metadata",
    }


def collect_repository(directory: Path) -> dict:
    """Record the checked-out commit and patch hash without exporting diff contents or remotes."""

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(directory), *arguments], stderr=subprocess.PIPE)

    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain")),
        "tracked_patch_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "patch_scope": "Tracked files only; dirty also reports untracked files, which are not hashed.",
    }


def collect_version_file(path: Path) -> dict:
    """Hash a version file and select version/build fields instead of installer user or path fields."""
    data = path.read_bytes()
    fields = {}
    for line in data.decode(errors="replace").splitlines():
        match = re.match(r"\s*([\w.-]*(?:version|build)[\w.-]*)\s*[:=]\s*(.*?)\s*$", line, flags=re.IGNORECASE)
        sensitive = ("token", "secret", "password", "credential", "user", "path", "url")
        if match and not any(word in match.group(1).lower() for word in sensitive) and "://" not in match.group(2):
            fields[match.group(1)] = match.group(2)
    return {"sha256": hashlib.sha256(data).hexdigest(), "fields": fields}


def parse_named_path(value: str) -> tuple[str, Path]:
    """Parse an explicit LABEL=PATH without interpreting it as shell code."""
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    return label, Path(path)


def main() -> None:
    """Write a manifest; callers choose which source and CANN/driver files to include."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", action="append", type=parse_named_path, default=[], metavar="LABEL=PATH")
    parser.add_argument("--version-file", action="append", type=parse_named_path, default=[], metavar="LABEL=PATH")
    arguments = parser.parse_args()
    for values in [arguments.repo, arguments.version_file]:
        if len({name for name, _ in values}) != len(values):
            parser.error("Labels must be unique within --repo or --version-file")
    environment_keys = [
        "OMP_NUM_THREADS",
        "HCCL_OP_EXPANSION_MODE",
        "SOC_VERSION",
        "VLLM_ASCEND_ENABLE_NZ",
        "TORCH_COMPILE_DISABLE",
        "TORCHINDUCTOR_DISABLE",
    ]
    manifest = {
        "captured_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": collect_packages(),
        "repositories": {name: collect_repository(path) for name, path in arguments.repo},
        "version_files": {name: collect_version_file(path) for name, path in arguments.version_file},
        "environment": {key: os.environ.get(key) for key in environment_keys},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
