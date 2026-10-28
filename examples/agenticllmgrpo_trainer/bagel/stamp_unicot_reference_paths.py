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
"""Stamp UniCoT parquet rows with extra_info.reference_image_path (REBUILD_UNICOT=1)."""

from __future__ import annotations

import argparse
import os

import pandas as pd


def stamp_reference_image_path(frame: pd.DataFrame) -> pd.DataFrame:
    extra = []
    for _, row in frame.iterrows():
        info = dict(row["extra_info"]) if "extra_info" in row and isinstance(row["extra_info"], dict) else {}
        ref = info.get("reference_image_path")
        if not ref:
            images = row.get("images") or row.get("input_image") or []
            if isinstance(images, str):
                images = [images]
            if images:
                info["reference_image_path"] = images[0]
        extra.append(info)
    frame = frame.copy()
    frame["extra_info"] = extra
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("REBUILD_UNICOT") != "1":
        raise SystemExit("Refusing to rewrite parquet unless REBUILD_UNICOT=1")
    frame = pd.read_parquet(args.input)
    stamp_reference_image_path(frame).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
