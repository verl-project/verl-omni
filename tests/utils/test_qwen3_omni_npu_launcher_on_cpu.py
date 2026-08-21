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

import os
import subprocess
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


def test_omnivideo_qi_launcher_wires_v1_gspo_and_qi_reward():
    launcher_dir = Path(__file__).parents[2] / "examples/gspo_trainer/qwen3_omni"
    launcher = (launcher_dir / "run_qwen3_omni_thinker_gspo_npu_omnivideo_qi_v1.sh").read_text(encoding="utf-8")

    required_settings = (
        "python3 -m verl_omni.trainer.main_omni",
        "+data.use_audio_in_video=true",
        "++data.mm_processor_kwargs.use_audio_in_video=true",
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
        "actor_rollout_ref.actor.clip_ratio_low=3e-4",
        "actor_rollout_ref.actor.clip_ratio_high=4e-4",
        "actor_rollout_ref.actor.kl_loss_coef=0.03",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/omnivideo_qi.py",
        "reward.reward_model.enable=${REWARD_MODEL_ENABLE}",
        "reward.reward_model.enable_resource_pool=false",
        'reward.reward_model.model_path="${OMNIVIDEO_QI_JUDGE_MODEL}"',
        "reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP}",
        "reward.reward_model.rollout.free_cache_engine=true",
        "REWARD_NUM_WORKERS:-$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP))",
        "NUM_GPUS_ACTOR_ROLLOUT_REWARD must be divisible by REWARD_TP",
        "Qwen/Qwen3-VL-8B-Instruct",
    )
    assert all(setting in launcher for setting in required_settings)
    assert "OMNIVIDEO_QI_JUDGE_URL:?" not in launcher


def test_omnivideo_qi_launcher_sources_ascend_env_without_nounset(tmp_path):
    launcher = (
        Path(__file__).parents[2]
        / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_omnivideo_qi_v1.sh"
    )
    ascend_home = tmp_path / "Ascend/cann"
    atb_home = tmp_path / "Ascend/nnal/atb"
    bin_dir = tmp_path / "bin"
    ascend_home.mkdir(parents=True)
    atb_home.mkdir(parents=True)
    bin_dir.mkdir()
    (ascend_home / "set_env.sh").write_text(":\n", encoding="utf-8")
    (atb_home / "set_env.sh").write_text('test -z "$ZSH_VERSION"\n', encoding="utf-8")
    fake_python = bin_dir / "python3"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("ZSH_VERSION", None)
    env.update(
        {
            "ASCEND_HOME_PATH": str(ascend_home),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
