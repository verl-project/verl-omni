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

"""Launcher contract test for the MiniCPM-o 4.5 simplex OPD wrapper.

Pins the wiring ``run_simplex_opd_lora.sh`` adds on top of the GSPO AVQA base
recipe — the distillation group, the DATASET case table, and the teacher
resource validation — not the tunable hyperparameters.
"""

import re
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "examples/opd_trainer/minicpm_o/run_simplex_opd_lora.sh"
_REPO_ROOT = _SCRIPT.resolve().parents[3]


def _active_text() -> str:
    """The script with comment lines dropped."""
    return "\n".join(line for line in _SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#"))


def _branch(case: str) -> str:
    """The body of one ``case`` branch, between ``<case>)`` and the next ``;;``."""
    match = re.search(rf"^\s+{re.escape(case)}\)\n(.*?)\n\s+;;", _active_text(), re.DOTALL | re.MULTILINE)
    assert match is not None, f"missing DATASET={case} branch"
    return match.group(1)


def _branch_value(case: str, key: str) -> str:
    pattern = re.escape(f"{key}=${{{key}:-") + '"([^"]+)"'
    matches = re.findall(pattern, _branch(case))
    assert len(matches) == 1, f"expected exactly one {key} default in DATASET={case}, got {matches}"
    return matches[0]


def test_wrapper_requires_student_and_teacher_checkpoints():
    text = _active_text()
    assert "STUDENT_MODEL:?Set STUDENT_MODEL" in text
    assert "TEACHER_MODEL:?Set TEACHER_MODEL" in text


def test_wrapper_adds_the_distillation_group_to_the_gspo_base():
    text = _active_text()
    assert "examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh" in text
    assert "actor_rollout_ref.model.use_remove_padding=false" in text
    assert "distillation.enabled=true" in text
    assert "distillation.teacher_models.teacher_model.inference.name=vllm_omni" in text
    assert 'distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}"' in text
    assert "distillation.distillation_loss.loss_mode=kl" in text
    assert "distillation.distillation_loss.use_policy_gradient=true" in text


def test_teacher_gpu_budget_must_be_a_positive_multiple_of_tensor_parallel():
    assert re.search(r"TEACHER_GPUS <= 0 \|\| TEACHER_TP <= 0", _active_text())
    assert re.search(r"TEACHER_GPUS % TEACHER_TP != 0", _active_text())


def test_each_dataset_branch_selects_its_own_data_reward_and_experiment():
    branches = {
        "avqa": {
            "experiment": "minicpm_o_4_5_thinker_lora_avqa_opd",
            "train": None,
            "reward": None,
        },
        "mmk12": {
            "experiment": "minicpm_o_4_5_thinker_lora_mmk12_opd",
            "train": "$HOME/data/mmk12/train.parquet",
            "reward": "verl_omni/utils/reward_score/mmk12_reward.py",
        },
        "gsm8k": {
            "experiment": "minicpm_o_4_5_thinker_lora_gsm8k_opd",
            "train": "$HOME/data/gsm8k/train.parquet",
            "reward": "verl_omni/utils/reward_score/gsm8k_reward.py",
        },
    }
    for case, expected in branches.items():
        branch = _branch(case)
        assert f"EXPERIMENT_NAME={expected['experiment']}" in branch, case
        if expected["train"] is not None:
            assert _branch_value(case, "TRAIN_FILE") == expected["train"], case
            assert _branch_value(case, "VAL_FILE") == expected["train"].replace("train.parquet", "test.parquet"), case
        if expected["reward"] is not None:
            assert _branch_value(case, "REWARD_FUNCTION_PATH").endswith(expected["reward"]), case
            assert "reward.custom_reward_function.name=compute_score" in branch, case


def test_branch_reward_paths_resolve_in_the_repo():
    for case in ("mmk12", "gsm8k"):
        reward = _branch_value(case, "REWARD_FUNCTION_PATH")
        assert (_REPO_ROOT / reward.removeprefix("${REPO_ROOT}/")).is_file(), case


def test_unknown_dataset_is_rejected():
    assert re.search(r"DATASET must be avqa, mmk12, or gsm8k", _active_text())
