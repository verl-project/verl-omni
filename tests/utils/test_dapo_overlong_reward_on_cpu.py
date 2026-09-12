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

"""Overlong penalty must change the reward on a truncated response (#446 Phase 2)."""

import numpy as np
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager

MAX_RESP_LEN = 16
OVERLONG_BUFFER_LEN = 4
OVERLONG_PENALTY_FACTOR = 1.0


def _build_local_tokenizer(tmp_path) -> AutoTokenizer:
    # A tiny in-memory tokenizer, so this test needs no network access or
    # downloaded artifacts (the DAPO manager only calls tokenizer.decode()).
    vocab = {"[UNK]": 0, **{str(i): i + 1 for i in range(100)}}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast_tokenizer.save_pretrained(tmp_path)
    return AutoTokenizer.from_pretrained(tmp_path)


def _compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0


def _build_manager(overlong_enable: bool, tokenizer) -> DAPORewardManager:
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "overlong_buffer_cfg": {
                        "enable": overlong_enable,
                        "len": OVERLONG_BUFFER_LEN,
                        "penalty_factor": OVERLONG_PENALTY_FACTOR,
                        "log": True,
                    },
                    "max_resp_len": MAX_RESP_LEN,
                }
            }
        }
    )
    return DAPORewardManager(config, tokenizer, _compute_score)


def _make_truncated_response() -> DataProto:
    # valid_len == MAX_RESP_LEN, i.e. exceed_len == OVERLONG_BUFFER_LEN -> full penalty.
    response_ids = torch.randint(0, 100, (1, MAX_RESP_LEN))
    attention_mask = torch.ones(1, MAX_RESP_LEN, dtype=torch.long)
    non_tensors = {
        "data_source": np.array(["dummy"], dtype=object),
        "reward_model": np.array([{"ground_truth": "x"}], dtype=object),
        "extra_info": np.array([{}], dtype=object),
    }
    return DataProto.from_dict(
        tensors={"responses": response_ids, "attention_mask": attention_mask},
        non_tensors=non_tensors,
    )


def test_overlong_penalty_changes_reward_on_truncated_response(tmp_path):
    batch = _make_truncated_response()
    tokenizer = _build_local_tokenizer(tmp_path)

    disabled = _build_manager(overlong_enable=False, tokenizer=tokenizer)
    enabled = _build_manager(overlong_enable=True, tokenizer=tokenizer)

    result_disabled = disabled.loop.run_until_complete(disabled.run_single(batch))
    result_enabled = enabled.loop.run_until_complete(enabled.run_single(batch))

    assert result_disabled["reward_score"] == 1.0
    assert "overlong" not in result_disabled["reward_extra_info"]

    assert result_enabled["reward_score"] < result_disabled["reward_score"]
    assert result_enabled["reward_extra_info"]["overlong"]
    assert result_enabled["reward_extra_info"]["overlong_reward"] < 0
