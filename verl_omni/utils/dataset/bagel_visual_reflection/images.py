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
"""Image resolution and content hashing for visual-reflection data."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .contracts import ImageRef, RejectionReason, VisualReflectionDataError, validate_image_ref


class ImageResolver(Protocol):
    """Resolve one supported source image value into an audited ``ImageRef``."""

    def __call__(
        self,
        value: Any,
        *,
        field: str,
        index: int,
        source_record_id: str,
    ) -> ImageRef:
        """Resolve and hash one source image value."""
        ...


class LocalImageResolver:
    """Resolve paths or undecoded ``datasets.Image`` mappings from local assets."""

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self._path_cache: dict[Path, ImageRef] = {}

    def __call__(
        self,
        value: Any,
        *,
        field: str,
        index: int,
        source_record_id: str,
    ) -> ImageRef:
        """Resolve a path, raw image mapping, or already canonical image reference."""
        if isinstance(value, Mapping):
            if set(value) == {"uri", "sha256"}:
                return validate_image_asset(value, asset_root=self.base_dir)
            return self._resolve_image_mapping(
                value,
                field=field,
                index=index,
                source_record_id=source_record_id,
            )
        if isinstance(value, str | os.PathLike):
            return self._resolve_path(
                Path(value),
                field=field,
                index=index,
                source_record_id=source_record_id,
            )
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field}[{index}] must be a path or undecoded image mapping",
            field=f"{field}[{index}]",
            source_record_id=source_record_id,
        )

    def _resolve_image_mapping(
        self,
        value: Mapping[str, Any],
        *,
        field: str,
        index: int,
        source_record_id: str,
    ) -> ImageRef:
        unknown = set(value) - {"path", "bytes"}
        if unknown or not ({"path", "bytes"} & set(value)):
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"{field}[{index}] has unsupported image mapping keys: {sorted(value)}",
                field=f"{field}[{index}]",
                source_record_id=source_record_id,
            )
        raw_bytes = value.get("bytes")
        raw_path = value.get("path")
        if raw_bytes is not None:
            if not isinstance(raw_bytes, bytes | bytearray | memoryview):
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_FIELD_TYPE,
                    f"{field}[{index}].bytes must contain raw bytes",
                    field=f"{field}[{index}].bytes",
                    source_record_id=source_record_id,
                )
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise VisualReflectionDataError(
                    RejectionReason.MISSING_IMAGE,
                    f"{field}[{index}] raw bytes require a stable path URI",
                    field=f"{field}[{index}].path",
                    source_record_id=source_record_id,
                )
            resolved, uri = _resolve_under_root(Path(raw_path), self.base_dir, field=f"{field}[{index}].path")
            if not resolved.is_file():
                raise VisualReflectionDataError(
                    RejectionReason.MISSING_IMAGE,
                    f"{field}[{index}] bytes are not materialized at {resolved}",
                    field=f"{field}[{index}].path",
                    source_record_id=source_record_id,
                )
            declared_bytes = bytes(raw_bytes)
            actual_bytes = resolved.read_bytes()
            if actual_bytes != declared_bytes:
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_IMAGE_HASH,
                    f"{field}[{index}] bytes do not match the materialized path",
                    field=f"{field}[{index}].bytes",
                    source_record_id=source_record_id,
                )
            return {"uri": uri, "sha256": hashlib.sha256(actual_bytes).hexdigest()}
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise VisualReflectionDataError(
                RejectionReason.MISSING_IMAGE,
                f"{field}[{index}] contains neither image bytes nor a path",
                field=f"{field}[{index}]",
                source_record_id=source_record_id,
            )
        return self._resolve_path(
            Path(raw_path),
            field=field,
            index=index,
            source_record_id=source_record_id,
        )

    def _resolve_path(
        self,
        raw_path: Path,
        *,
        field: str,
        index: int,
        source_record_id: str,
    ) -> ImageRef:
        if not str(raw_path).strip():
            raise VisualReflectionDataError(
                RejectionReason.MISSING_IMAGE,
                f"{field}[{index}] has an empty image path",
                field=f"{field}[{index}]",
                source_record_id=source_record_id,
            )
        resolved, uri = _resolve_under_root(raw_path, self.base_dir, field=f"{field}[{index}]")
        cached = self._path_cache.get(resolved)
        if cached is not None:
            return dict(cached)
        if not resolved.is_file():
            raise VisualReflectionDataError(
                RejectionReason.MISSING_IMAGE,
                f"{field}[{index}] image does not exist: {resolved}",
                field=f"{field}[{index}]",
                source_record_id=source_record_id,
            )
        digest = _sha256_file(resolved)
        image_ref: ImageRef = {"uri": uri, "sha256": digest}
        self._path_cache[resolved] = image_ref
        return dict(image_ref)


def validate_image_asset(image: Mapping[str, Any], *, asset_root: str | os.PathLike[str]) -> ImageRef:
    """Re-hash a local canonical image reference and verify its declared digest."""
    image_ref = validate_image_ref(image)
    root = Path(asset_root).expanduser().resolve()
    resolved, uri = _resolve_under_root(Path(image_ref["uri"]), root, field="image.uri")
    if not resolved.is_file():
        raise VisualReflectionDataError(
            RejectionReason.MISSING_IMAGE,
            f"canonical image does not exist: {resolved}",
            field="image.uri",
        )
    actual = _sha256_file(resolved)
    if actual != image_ref["sha256"]:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_IMAGE_HASH,
            f"declared image digest {image_ref['sha256']} does not match {actual}",
            field="image.sha256",
        )
    return {"uri": uri, "sha256": actual}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _resolve_under_root(path: Path, root: Path, *, field: str) -> tuple[Path, str]:
    expanded = path.expanduser()
    resolved = expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise VisualReflectionDataError(
            RejectionReason.MISSING_IMAGE,
            f"{field} escapes the configured asset root: {path}",
            field=field,
        ) from exc
    return resolved, relative.as_posix()
