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
"""Schema checks for the toy agentic parquet generator."""

from __future__ import annotations

import pandas as pd

from tests.special_e2e.create_dummy_agentic_data import DATA_SOURCE, USER_PROMPTS, build_rows, main

REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def test_build_rows_schema_and_prompt_seed():
    rows = build_rows("train", 3)
    assert len(rows) == 3
    assert {k for row in rows for k in row} >= REQUIRED_COLUMNS

    first = rows[0]
    assert first["data_source"] == DATA_SOURCE
    assert isinstance(first["prompt"], list)
    assert first["prompt"][0]["role"] == "system"
    # Hermes few-shot precedes the live user request.
    assert first["prompt"][1]["role"] == "user"
    assert any(
        isinstance(m, dict) and m.get("role") == "assistant" and "<tool_call>" in str(m.get("content", ""))
        for m in first["prompt"]
    )
    assert first["prompt"][-1]["role"] == "user"
    assert first["prompt"][-1]["content"] == USER_PROMPTS[0]
    assert first["extra_info"]["raw_prompt"] == USER_PROMPTS[0]
    assert first["extra_info"]["toy_agentic"] is True
    # Seeds only — nested offline turns are produced online by the agent loop.
    assert "turns" not in first


def test_main_writes_train_and_val_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_dummy_agentic_data.py",
            "--local_save_dir",
            str(tmp_path),
            "--train_size",
            "4",
            "--val_size",
            "2",
        ],
    )
    main()

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    assert train_path.is_file()
    assert val_path.is_file()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    assert len(train_df) == 4
    assert len(val_df) == 2
    assert REQUIRED_COLUMNS.issubset(train_df.columns)
    assert train_df.iloc[0]["prompt"][-1]["content"] in USER_PROMPTS
