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
"""CPU contracts for the Hindi SFT-LoRA GRPO recipe and its input assets."""

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from safetensors.torch import save_file

ROOT = Path(__file__).parents[2]
LAUNCHER = ROOT / "examples/grpo_trainer/qwen3_tts/run_qwen3_tts_hindi_grpo.sh"


def _load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepare():
    return _load_script(
        "qwen3_tts_hindi_data_test",
        "examples/grpo_trainer/qwen3_tts/data_process/prepare_indicvoices_hindi_grpo.py",
    )


@pytest.fixture(scope="module")
def merger():
    return _load_script(
        "qwen3_tts_hindi_sft_merge_test",
        "examples/grpo_trainer/qwen3_tts/data_process/merge_hindi_sft_lora.py",
    )


def _sample(index, *, scenario="Read", duration=2.0, snr=25.0, normalized=None):
    return {
        "text": f"raw {index}",
        "normalized": normalized if normalized is not None else f"normalized {index}",
        "scenario": scenario,
        "duration": duration,
        "snr": snr,
    }


def _mlx_adapter(merger):
    tensors = {}
    for module in merger.expected_modules():
        tensors[f"{module}.lora_a"] = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        tensors[f"{module}.lora_b"] = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    return tensors


def _capture_launcher(tmp_path, *extra_overrides):
    capture = tmp_path / "args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${CAPTURE_PATH}"\n')
    fake_python.chmod(0o755)
    model_path = tmp_path / "qwen3-tts-hindi-sft"
    model_path.mkdir()
    train_file = tmp_path / "train.parquet"
    validation_file = tmp_path / "validation.parquet"
    speaker_path = tmp_path / "speaker.json"
    train_file.touch()
    validation_file.touch()
    speaker_path.write_text("[0.0, 1.0]")
    env = os.environ | {
        "PYTHON_BIN": str(fake_python),
        "CAPTURE_PATH": str(capture),
        "MODEL_PATH": str(model_path),
        "TRAIN_FILE": str(train_file),
        "VAL_FILE": str(validation_file),
        "SPK_EMBED_PATH": str(speaker_path),
        "SCORER_URL": "http://127.0.0.1:18080/score",
        "OUTPUT_DIR": str(tmp_path / "output"),
    }
    subprocess.run(["bash", str(LAUNCHER), *extra_overrides], check=True, env=env, capture_output=True, text=True)
    return capture.read_text().splitlines()


def _override_map(arguments):
    result = {}
    for argument in arguments[2:]:
        if "=" in argument:
            key, value = argument.lstrip("+").split("=", 1)
            result[key] = value
    return result


def test_first_raw_read_window_is_filtered_before_validation_collection(prepare):
    samples = [
        _sample(0, scenario="Extempore"),
        _sample(1, duration=0.9),
        _sample(2, snr=19.9),
        _sample(3, normalized="preferred text"),
        _sample(4, normalized=""),
    ]

    train, validation, audit = prepare.collect_read_prompt_splits(samples, raw_read_limit=3, validation_count=1)

    assert [row["source_index"] for row in train] == [3]
    assert train[0]["text"] == "preferred text"
    assert [row["source_index"] for row in validation] == [4]
    assert validation[0]["text"] == "raw 4"
    assert audit["rejected_train"] == {"duration": 1, "snr": 1}


def test_non_string_scenario_fails_closed(prepare):
    with pytest.raises(TypeError, match="scenario must be a string"):
        prepare.collect_read_prompt_splits([_sample(0, scenario=1)], raw_read_limit=1, validation_count=1)


def test_standard_verl_rows_are_built_from_pinned_scenario_strings(prepare):
    samples = [_sample(index) for index in range(8)]
    train, validation, _ = prepare.collect_read_prompt_splits(samples, raw_read_limit=5, validation_count=3)
    splits = prepare.build_splits(
        train,
        validation,
        expected_train_count=5,
        expected_validation_count=3,
    )

    assert len(splits["train"]) == 5
    assert splits["validation"][0]["extra_info"]["split"] == "validation"
    assert splits["validation"][0]["reward_model"] == {
        "style": "model",
        "ground_truth": "normalized 5",
    }


@pytest.mark.parametrize("snr", [None, float("nan")])
def test_missing_or_non_finite_snr_is_rejected(prepare, snr):
    assert prepare._record(_sample(0, snr=snr), 0) == (None, "snr")


def test_split_fails_closed_on_text_leakage(prepare):
    train = [prepare._record(_sample(0, normalized="duplicate"), 0)[0]]
    validation = [prepare._record(_sample(1, normalized="duplicate"), 1)[0]]

    with pytest.raises(ValueError, match="overlap"):
        prepare.build_splits(train, validation, expected_train_count=1, expected_validation_count=1)


def test_split_fails_closed_on_normalized_to_raw_text_leakage(prepare):
    train = [prepare._record(_sample(0, normalized="raw 1"), 0)[0]]
    validation = [prepare._record(_sample(1, normalized="different"), 1)[0]]

    with pytest.raises(ValueError, match="overlap"):
        prepare.build_splits(train, validation, expected_train_count=1, expected_validation_count=1)


def test_published_training_count_is_fail_closed(prepare):
    with pytest.raises(ValueError, match="Expected 863 training prompts"):
        prepare.build_splits(
            [prepare._record(_sample(0), 0)[0]],
            [prepare._record(_sample(1), 1)[0]],
            expected_validation_count=1,
        )


def test_content_hash_is_order_sensitive_and_reproducible(prepare):
    rows = [{"id": 1}, {"id": 2}]

    assert prepare._content_sha256(rows) == prepare._content_sha256(list(rows))
    assert prepare._content_sha256(rows) != prepare._content_sha256(list(reversed(rows)))


def test_mlx_adapter_mapping_is_complete_and_peft_compatible(merger, tmp_path):
    source = tmp_path / "adapters.safetensors"
    save_file(_mlx_adapter(merger), source)

    converted = merger.load_mlx_adapter(source, expected_sha256=None)

    assert len(converted) == 462
    assert all(key.startswith("base_model.model.talker.") for key in converted)
    assert all(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in converted)


def test_merge_preserves_base_checkpoint_dtype(merger, tmp_path):
    save_file({"weight": torch.zeros(2, dtype=torch.bfloat16)}, tmp_path / "model.safetensors")

    assert merger.checkpoint_dtype(tmp_path) is torch.bfloat16


def test_merge_fails_closed_on_incomplete_topology(merger, tmp_path):
    tensors = _mlx_adapter(merger)
    tensors.pop(next(key for key in tensors if key.endswith(".lora_b")))
    source = tmp_path / "broken.safetensors"
    save_file(tensors, source)

    with pytest.raises(ValueError, match="topology mismatch"):
        merger.load_mlx_adapter(source, expected_sha256=None)


def test_merge_checks_published_source_hash(merger, tmp_path):
    source = tmp_path / "adapters.safetensors"
    save_file(_mlx_adapter(merger), source)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        merger.load_mlx_adapter(source, expected_sha256="0" * 64)


def test_merge_checks_every_pinned_base_asset(merger, tmp_path):
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"base")
    expected = {"model.safetensors": hashlib.sha256(b"base").hexdigest()}

    assert merger.validate_pinned_base(tmp_path, expected) == expected

    model_path.write_bytes(b"different")
    with pytest.raises(ValueError, match="Base asset SHA-256 mismatch"):
        merger.validate_pinned_base(tmp_path, expected)


def test_recipe_wraps_the_merged_generic_qwen3_tts_path(tmp_path):
    arguments = _capture_launcher(tmp_path)
    overrides = _override_map(arguments)
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="omni_trainer", overrides=arguments[2:])

    assert arguments[:2] == ["-m", "verl_omni.trainer.main_omni"]
    assert config.actor_rollout_ref.model.model_stage == "talker"
    assert overrides["data.train_batch_size"] == "4"
    assert overrides["data.train_max_samples"] == "863"
    assert overrides["data.val_max_samples"] == "100"
    assert overrides["data.validation_shuffle"] == "false"
    assert overrides["actor_rollout_ref.model.lora_rank"] == "8"
    assert overrides["actor_rollout_ref.model.lora_alpha"] == "16"
    assert overrides["actor_rollout_ref.model.lora_dtype"] == "float32"
    assert overrides["actor_rollout_ref.model.lora.merge"] == "true"
    assert overrides["actor_rollout_ref.actor.fsdp_config.model_dtype"] == "float32"
    assert overrides["actor_rollout_ref.actor.fsdp_config.dtype"] == "bfloat16"
    assert overrides["actor_rollout_ref.actor.optim.lr"] == "5e-6"
    assert overrides["actor_rollout_ref.actor.kl_loss_coef"] == "0.08"
    assert overrides["actor_rollout_ref.actor.kl_loss_type"] == "k3"
    assert overrides["actor_rollout_ref.rollout.n"] == "4"
    assert overrides["actor_rollout_ref.rollout.logprobs_mode"] == "processed_logprobs"
    assert overrides["algorithm.adv_estimator"] == "grpo"
    assert overrides["reward.custom_reward_function.path"].endswith("audio_http_scorer_client")
    assert overrides["reward.reward_manager.name"] == "AudioRewardManager"


def test_recipe_keeps_complete_step_20_validation_and_allows_smoke_overrides(tmp_path):
    arguments = _capture_launcher(tmp_path, "trainer.total_training_steps=3", "trainer.test_freq=-1")
    overrides = _override_map(arguments)
    launcher_text = LAUNCHER.read_text()
    generic_text = (ROOT / "examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh").read_text()

    assert 'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"' in launcher_text
    assert 'TEST_FREQ="${TEST_FREQ:-20}"' in launcher_text
    assert 'SAVE_FREQ="${SAVE_FREQ:-20}"' in launcher_text
    assert "trainer.log_val_generations=100" in generic_text
    assert overrides["trainer.total_training_steps"] == "3"
    assert overrides["trainer.test_freq"] == "-1"
    assert overrides["trainer.max_actor_ckpt_to_keep"] == "21"
    assert "qwen3_tts_single_turn" not in launcher_text
    assert "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO" not in launcher_text
    assert "qwen3_tts_compat" not in launcher_text
