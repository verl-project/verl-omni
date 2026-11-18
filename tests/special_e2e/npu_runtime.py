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
"""Shared Hydra runtime overrides for NPU smoke launchers."""

import json
import os
import re
from collections.abc import Mapping


def npu_precompile_threads(environ: Mapping[str, str] | None = None) -> str:
    """Bound tiny-smoke compiler pools while preserving explicit thread counts."""
    env = os.environ if environ is None else environ
    value = env.get("TORCHNPU_PRECOMPILE_THREADS") or "2"
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("TORCHNPU_PRECOMPILE_THREADS must be a positive integer.")
    return value


def npu_runtime_overrides(environ: Mapping[str, str] | None = None) -> list[str]:
    """Return launcher defaults; caller-specific Hydra arguments may follow them."""
    env = os.environ if environ is None else environ
    overrides = [
        "trainer.device=npu",
        "data.dataloader_num_workers=0",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=2",
    ]
    for name, value in {
        "ASCEND_VISIBLE_DEVICES": env["ASCEND_VISIBLE_DEVICES"],
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": env.get(
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1"
        ),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ASCEND_ENABLE_NZ": "0",
        "TORCHNPU_PRECOMPILE_THREADS": npu_precompile_threads(env),
    }.items():
        overrides.append(f"++ray_kwargs.ray_init.runtime_env.env_vars.{name}={json.dumps(value)}")
    # Rollout and reward servers both sleep/wake; Ascend rejects NZ weights on wake_up.
    overrides.extend(
        [
            "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0",
            "++reward.reward_model.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0",
        ]
    )
    if ports := env.get("RAY_MASTER_PORT_RANGE"):
        overrides.append(f"trainer.ray_master_port_range=[{ports}]")
    if env.get("NPU_SMOKE_SKIP_GLOBAL_CLEANUP") == "1":
        overrides.append("+ray_kwargs.ray_init.address=local")
    return overrides


if __name__ == "__main__":
    print("\n".join(npu_runtime_overrides()))
