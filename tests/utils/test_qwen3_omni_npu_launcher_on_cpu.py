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

from pathlib import Path


def test_avqa_npu_launcher_wires_v1_multimodal_training():
    launcher_dir = Path(__file__).parents[2] / "examples/gspo_trainer/qwen3_omni"
    avqa_launcher = (launcher_dir / "run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh").read_text(encoding="utf-8")

    required_settings = (
        "python3 -m verl_omni.trainer.main_omni",
        "data.custom_cls.name=QwenOmniRLHFDataset",
        "++data.mm_processor_kwargs.sampling_rate=16000",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe",
        "reward.reward_manager.source=register",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py",
    )
    assert all(setting in avqa_launcher for setting in required_settings)
    assert "models.transformers" not in avqa_launcher
