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
"""CPU tests for omni model and rollout registries."""

import pytest
from omegaconf import OmegaConf

from verl_omni.pipelines.model_base import OmniModelBase, OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_omni import Qwen3OmniRolloutAdapter, Qwen3OmniThinkerAdapter


class TestOmniModelBaseRegistry:
    def test_builtin_qwen3_omni_thinker_adapter_registered(self):
        cfg = OmegaConf.create(
            {
                "architecture": "Qwen3OmniMoeForConditionalGeneration",
                "model_stage": "thinker",
                "external_lib": None,
            }
        )

        assert OmniModelBase.get_class(cfg) is Qwen3OmniThinkerAdapter

    def test_get_class_unknown_architecture_raises(self):
        cfg = OmegaConf.create(
            {
                "architecture": "__DoesNotExist__",
                "model_stage": "thinker",
                "external_lib": None,
            }
        )

        with pytest.raises(NotImplementedError, match="No omni model registered"):
            OmniModelBase.get_class(cfg)


class TestQwen3OmniThinkerAdapter:
    def test_thinker_loading_metadata(self):
        kwargs = Qwen3OmniThinkerAdapter.get_model_loading_kwargs({})

        assert kwargs["hf_config_name"] == "thinker_config"
        assert kwargs["no_split_modules"] == ["Qwen3OmniMoeThinkerTextDecoderLayer"]
        assert kwargs["tie_word_embeddings_override"] is False

    def test_thinker_strips_downstream_stages(self):
        assert Qwen3OmniThinkerAdapter.get_strip_modules({}) == ["talker", "code2wav", "code_predictor"]


class TestOmniRolloutPipelineBaseRegistry:
    def test_builtin_qwen3_omni_rollout_adapter_registered(self):
        assert OmniRolloutPipelineBase.get_class("qwen3_omni_moe") is Qwen3OmniRolloutAdapter

    def test_qwen3_omni_thinker_stage_config_uses_all_gpus(self):
        stage_config = Qwen3OmniRolloutAdapter.get_stage_config(pipeline_mode="thinker_only", num_gpus=4)

        assert stage_config[0]["runtime"]["devices"] == "0,1,2,3"
        assert stage_config[0]["engine_args"]["model_stage"] == "thinker"
        assert stage_config[0]["engine_args"]["tensor_parallel_size"] == 4
        assert stage_config[0]["final_output_type"] == "text"

    def test_qwen3_omni_talker_pipeline_is_deferred(self):
        with pytest.raises(NotImplementedError, match="Only thinker_only is supported"):
            Qwen3OmniRolloutAdapter.get_stage_config(pipeline_mode="thinker_talker", num_gpus=4)
