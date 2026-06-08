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
"""
L2 GPU test: train–rollout consistency after LoRA weight sync (OCR recipe).

Mirrors production hybrid-engine flow:
  init → sync LoRA to vLLM → consistency check
  → (perturb LoRA → sync → consistency check) × N
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from verl_omni.testing.train_rollout_consistency import TrainRolloutConsistencyMetrics

from .consistency_check import run_hybrid_consistency_check
from .hybrid_consistency_session import HybridConsistencySession, perturb_hybrid_actor_lora, run_hybrid_session
from .train_rollout_consistency_helpers import QwenImageConsistencyProfile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.fixture(scope="module")
def ocr_profile(qwen_image_consistency_profile: QwenImageConsistencyProfile):
    if qwen_image_consistency_profile.recipe != "ocr":
        pytest.skip("weight-sync test requires OCR recipe")
    return qwen_image_consistency_profile


def _format_metrics_row(step: int, metrics: TrainRolloutConsistencyMetrics) -> str:
    return (
        f"step {step:2d}: logprob_abs_max={metrics.logprob_abs_max:.6f}  "
        f"logprob_abs_mean={metrics.logprob_abs_mean:.6f}  "
        f"logprob_rmse={metrics.logprob_rmse:.6f}  "
        f"ratio_mean={metrics.ratio_mean:.6f}  "
        f"ratio_std={metrics.ratio_std:.6f}  "
        f"pg_clipfrac={metrics.pg_clipfrac:.6f}  "
        f"ppo_kl={metrics.ppo_kl:.6f}"
    )


def _print_weight_sync_summary(history: list[tuple[int, TrainRolloutConsistencyMetrics]]) -> None:
    print("\n[weight-sync consistency summary]")
    for step, metrics in history:
        print(_format_metrics_row(step, metrics))
    if len(history) >= 2:
        baseline = history[0][1].logprob_abs_max
        final = history[-1][1].logprob_abs_max
        ratio = final / baseline if baseline > 0 else float("inf")
        print(
            f"  logprob_abs_max: step {history[0][0]}={baseline:.6f} → step {history[-1][0]}={final:.6f} ({ratio:.2f}×)"
        )


async def _run_weight_sync_consistency(
    profile: QwenImageConsistencyProfile,
    tmp_path: Path,
    *,
    num_steps: int,
) -> list[tuple[int, TrainRolloutConsistencyMetrics]]:
    history: list[tuple[int, TrainRolloutConsistencyMetrics]] = []
    session = await HybridConsistencySession.create(profile)
    try:
        await session.sync_weights(global_steps=0)
        metrics = await run_hybrid_consistency_check(
            profile,
            tmp_path,
            label_suffix=" step-0",
            sync_before_check=False,
            metadata_extra={"phase": "baseline", "weight_sync_step": 0},
            session=session,
        )
        history.append((0, metrics))

        for step in range(1, num_steps + 1):
            perturb_hybrid_actor_lora(session.worker_group, scale=1e-3, seed=12345 + step)
            await session.sync_weights(global_steps=step)
            metrics = await run_hybrid_consistency_check(
                profile,
                tmp_path,
                label_suffix=f" step-{step}",
                sync_before_check=False,
                metadata_extra={"phase": "post_perturb_sync", "weight_sync_step": step},
                session=session,
            )
            history.append((step, metrics))
    finally:
        await session.close()

    _print_weight_sync_summary(history)
    return history


def test_qwen_image_lora_weight_sync_consistency(
    ray_runtime,
    tmp_path,
    ocr_profile,
    request: pytest.FixtureRequest,
) -> None:
    """LoRA actor weights stay consistent with vLLM rollout after layered_summon sync."""
    num_steps = request.config.getoption("--weight-sync-steps")
    if num_steps < 1:
        pytest.skip("--weight-sync-steps must be >= 1")

    history = run_hybrid_session(_run_weight_sync_consistency(ocr_profile, tmp_path, num_steps=num_steps))
    assert history, "expected at least one consistency check"
