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

"""CPU tests for the LTX-2.3 rollout output contract."""

import importlib.util
from pathlib import Path


def _load_ltx2_common_module():
    module_path = Path(__file__).parents[2] / "verl_omni/pipelines/ltx2_flow_grpo/common.py"
    spec = importlib.util.spec_from_file_location("ltx2_flow_grpo_common", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generic_image_output_is_normalized_to_video_tensor():
    common = _load_ltx2_common_module()

    assert common.normalize_ltx_output_type("image") == "pt"
    assert common.normalize_ltx_output_type("pt") == "pt"
    assert common.normalize_ltx_output_type("latent") == "latent"
    assert common.normalize_ltx_output_type(None) is None
