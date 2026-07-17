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
"""CPU tests for the stage-output-type reader that detects a codec (TTS) rollout."""

import yaml

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import _read_stage_output_type


def _write(tmp_path, stages):
    p = tmp_path / "stages.yaml"
    p.write_text(yaml.safe_dump({"stage_args": stages}))
    return {"vllm_omni": {"stage_configs_path": str(p)}}


def test_codec_stage(tmp_path):
    ek = _write(tmp_path, [{"engine_args": {"engine_output_type": "codec"}, "final_output": True}])
    assert _read_stage_output_type(ek) == "codec"


def test_final_output_type_wins(tmp_path):
    ek = _write(tmp_path, [{"engine_args": {"engine_output_type": "codec"}, "final_output_type": "codec"}])
    assert _read_stage_output_type(ek) == "codec"


def test_text_stage_is_not_codec(tmp_path):
    ek = _write(tmp_path, [{"engine_args": {"engine_output_type": "text"}}])
    assert _read_stage_output_type(ek) == "text"


def test_terminal_stage_selected(tmp_path):
    ek = _write(
        tmp_path,
        [
            {"engine_args": {"engine_output_type": "codec"}},
            {"engine_args": {"engine_output_type": "audio"}, "final_output": True},
        ],
    )
    assert _read_stage_output_type(ek) == "audio"


def test_hyphenated_key(tmp_path):
    p = tmp_path / "stages.yaml"
    p.write_text(yaml.safe_dump({"stage_args": [{"engine_args": {"engine_output_type": "codec"}}]}))
    assert _read_stage_output_type({"vllm_omni": {"stage-configs-path": str(p)}}) == "codec"


def test_missing_config_is_none():
    assert _read_stage_output_type({}) is None
    assert _read_stage_output_type({"vllm_omni": {}}) is None
    assert _read_stage_output_type(None) is None
