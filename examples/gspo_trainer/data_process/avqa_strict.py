# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exclude AVQA training rows sharing image or audio bytes with validation.

The official AVQA-R1 split reuses problem IDs and some media across splits.
Keep the original parquet files unchanged; this writes a separate strict train.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _media_paths(row) -> tuple[Path, Path]:
    """Read the existing AVQA converter's one-image, one-audio parquet shape."""
    images, audios = row["images"], row["audios"]
    if len(images) != 1 or len(audios) != 1:
        raise ValueError("Expected exactly one image and one audio per AVQA row")
    image, audio = Path(images[0]["image"]), Path(audios[0])
    if not image.is_file() or not audio.is_file():
        raise FileNotFoundError(f"Missing AVQA media: {image}, {audio}")
    return image, audio


def _digest(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        cache[path] = digest.hexdigest()
    return cache[path]


def make_strict_train(train_file: Path, validation_file: Path, output_file: Path) -> dict:
    """Write a strict train parquet while preserving both source splits.

    Rows whose image or audio bytes occur in validation are excluded. The
    returned report records counts and which media triggered each exclusion.
    Missing media raise ``FileNotFoundError``; overlapping input/output paths
    raise ``ValueError`` before any parquet is written.
    """
    train_file = Path(train_file).expanduser().resolve()
    validation_file = Path(validation_file).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    if train_file == validation_file:
        raise ValueError("Training and validation must be different files")
    if output_file in {train_file, validation_file}:
        raise ValueError("Strict output must not overwrite an original split")

    train = pd.read_parquet(train_file)
    validation = pd.read_parquet(validation_file)
    cache: dict[Path, str] = {}
    validation_image_hashes: set[str] = set()
    validation_audio_hashes: set[str] = set()
    for _, row in validation.iterrows():
        image, audio = _media_paths(row)
        validation_image_hashes.add(_digest(image, cache))
        validation_audio_hashes.add(_digest(audio, cache))

    kept = []
    excluded = []
    for index, row in train.iterrows():
        image, audio = _media_paths(row)
        shared_image = _digest(image, cache) in validation_image_hashes
        shared_audio = _digest(audio, cache) in validation_audio_hashes
        if shared_image or shared_audio:
            excluded.append({"row": int(index), "shared_image": shared_image, "shared_audio": shared_audio})
        else:
            kept.append(index)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".avqa-strict-", suffix=".parquet", dir=output_file.parent)
    os.close(fd)
    try:
        train.loc[kept].reset_index(drop=True).to_parquet(
            temporary, engine="pyarrow", index=False, use_dictionary=False
        )
        os.replace(temporary, output_file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "source_train": len(train),
        "validation_unchanged": len(validation),
        "strict_train": len(kept),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "output": str(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--validation_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--audit_file", type=Path, required=True)
    args = parser.parse_args()
    audit_file = args.audit_file.expanduser().resolve()
    protected = {path.expanduser().resolve() for path in (args.train_file, args.validation_file, args.output_file)}
    if audit_file in protected:
        raise ValueError("Audit output must not overwrite a parquet split")
    audit = make_strict_train(args.train_file, args.validation_file, args.output_file)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "excluded"}))


if __name__ == "__main__":
    main()
