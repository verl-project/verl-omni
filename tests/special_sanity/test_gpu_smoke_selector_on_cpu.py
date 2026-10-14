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

import importlib.util
import sys
from pathlib import Path


def _load_selector():
    path = Path(__file__).parents[1] / "gpu_smoke/select_gpu_smoke_groups.py"
    spec = importlib.util.spec_from_file_location("gpu_smoke_selector_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen3_tts_smoke_files_select_only_omni_e2e_group():
    selector = _load_selector()

    selected = selector.select_group_names(
        [
            ".github/qwen_tts_pin.txt",
            "examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh",
            "tests/special_e2e/build_qwen3_tts_tiny_random.py",
            "tests/special_e2e/run_qwen3_tts_grpo_smoke.sh",
            "tests/special_e2e/create_dummy_qwen3_tts_grpo_data.py",
            "tests/special_e2e/qwen3_tts_dummy_reward.py",
        ]
    )

    assert selected == ["ci-e2e-omni"]


def test_minicpmo_smoke_files_select_only_omni_e2e_group():
    selector = _load_selector()

    selected = selector.select_group_names(
        [
            "tests/special_e2e/build_minicpmo_tiny_random.py",
            "tests/special_e2e/create_dummy_minicpmo_avqa_data.py",
            "tests/special_e2e/run_gspo_minicpmo_thinker_lora_v1_smoke.sh",
        ]
    )

    assert selected == ["ci-e2e-omni"]


def test_minicpm_pipeline_changes_select_omni_e2e_group():
    selector = _load_selector()

    for path in (
        "verl_omni/pipelines/minicpm/thinker_training_adapter.py",
        "verl_omni/pipelines/minicpm/omni_rollout_adapter.py",
        "verl_omni/pipelines/minicpm/media_inputs.py",
        "verl_omni/pipelines/minicpm/prompt_parity.py",
        "verl_omni/pipelines/minicpm/reward_decode.py",
    ):
        assert "ci-e2e-omni" in selector.select_group_names([path]), path


def test_shared_omni_veomni_engine_and_adapters_select_omni_smoke():
    selector = _load_selector()
    for path in (
        "verl_omni/workers/engine/veomni/omni_impl.py",
        "verl_omni/pipelines/model_base.py",
        "verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py",
        "verl_omni/pipelines/qwen3_omni/veomni.py",
        "tests/workers/test_omni_veomni_engine_on_cpu.py",
        "tests/pipelines/test_qwen3_omni_veomni_on_cpu.py",
    ):
        assert "ci-e2e-omni" in selector.select_group_names([path]), path


def test_omni_e2e_runner_forwards_trainer_args_to_every_run_test():
    # A run_test that drops the trainer args silently loses the configured
    # rendezvous port range, so every launched command must forward them.
    script = Path(__file__).parents[1] / "gpu_smoke/run_gpu_smoke_omni_e2e.sh"
    blocks = script.read_text().split("run_test ")[1:]
    assert blocks, "no run_test blocks found in the omni e2e runner"
    for block in blocks:
        commands = [line for line in block.splitlines() if not line.lstrip().startswith("#")]
        assert any('"${omni_trainer_args[@]}"' in line for line in commands), block.splitlines()[0]
