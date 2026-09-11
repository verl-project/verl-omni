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

import hashlib

import pytest

from verl_omni.utils.fs import diffusion_model_provenance


class TestDiffusionModelProvenance:
    @pytest.mark.parametrize("source", ["snapshot", "download", "local"])
    def test_revision_is_recorded_only_when_available(self, tmp_path, source):
        revision = "a" * 40
        root = tmp_path / "snapshots" / revision if source == "snapshot" else tmp_path
        (root / "transformer").mkdir(parents=True)
        config = b'{"in_channels": 64}'
        (root / "transformer/config.json").write_bytes(config)
        if source == "download":
            metadata = root / ".cache/huggingface/download/model_index.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(revision + "\netag\n0\n")
        value = diffusion_model_provenance(str(root))
        assert value["base_model_revision"] == (None if source == "local" else revision)
        assert value["base_transformer_config_sha256"] == hashlib.sha256(config).hexdigest()
