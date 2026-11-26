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
"""Reward-calibration entry gate for duplex GRPO (RFC verl-omni #905 §3.6).

Rolls the *unmodified* starting checkpoint over a mined event set (via the
same session driver the trainer uses), scores every episode with the rule
reward, and audits each axis's normalized mean against the mid-band gate: an
axis that floors or saturates the base model means the event construction or
the reward band is misaligned with the checkpoint's actual behavior and must
be fixed before training. Exits non-zero on a failed gate.

Usage (GPU host with the capture-capable vLLM-Omni installed)::

    python scripts/duplex_calibration_audit.py \
        --model /path/to/MiniCPM-o-4_5 --episodes /path/to/episodes.parquet \
        --output calibration.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from verl_omni.utils.dataset.duplex.calibration import CALIBRATION_MAX_MEAN_RATIO
from verl_omni.utils.dataset.duplex.deploy_config import (
    stage_init_timeout_from_env,
    write_deploy_config_for,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="MiniCPM-o 4.5 checkpoint directory")
    parser.add_argument("--episodes", required=True, help="mined duplex episode parquet")
    parser.add_argument("--output", default="duplex_calibration.json", help="audit report path")
    parser.add_argument("--num-episodes", type=int, default=48, help="episodes to roll per run")
    parser.add_argument(
        "--max-mean-ratio",
        type=float,
        default=CALIBRATION_MAX_MEAN_RATIO,
        help="ceiling on each axis's normalized mean reward (the floor is CALIBRATION_MIN_MEAN_RATIO)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=stage_init_timeout_from_env(),
        help="per-stage init timeout in seconds (env VERL_OMNI_DUPLEX_STAGE_INIT_TIMEOUT)",
    )
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="path for the generated thinker-only duplex deploy config "
        "(default: next to --output with a .deploy.yaml suffix)",
    )
    return parser.parse_args(argv)


def _duplex_deploy_config_path(args: argparse.Namespace) -> str:
    """Generate the thinker-only duplex deploy config the engine client needs."""
    return write_deploy_config_for(
        output_path=args.output, deploy_config=args.deploy_config, tensor_parallel_size=args.tensor_parallel_size
    )


def _build_engine(args: argparse.Namespace):
    """A ``DuplexOmni`` with the duplex KV window's cache settings."""
    from verl_omni.utils.dataset.duplex.deploy_config import build_duplex_gate_engine

    return build_duplex_gate_engine(
        model=args.model,
        deploy_config=_duplex_deploy_config_path(args),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        stage_init_timeout=args.stage_init_timeout,
    )


def roll_rewards(args: argparse.Namespace) -> dict[str, list[float]]:
    """Drive the base policy over the event set and collect per-axis rewards.

    Uses the same engine path as the rollout strategy: a ``DuplexOmni`` built
    from engine args with the thinker-only duplex deploy config, sessions
    with capture on, the same chunk cadence. The DuplexStrategy driver is
    reused directly so calibration measures exactly what training would
    measure.
    """

    from transformers import AutoProcessor

    from verl_omni.utils.dataset.duplex.duplex_rl_dataset import DuplexEpisodeDataset
    from verl_omni.utils.reward_score.duplex_turn_taking import compute_score_duplex_turn_taking

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    engine = _build_engine(args)
    dataset = DuplexEpisodeDataset([args.episodes])
    rewards: dict[str, list[float]] = defaultdict(list)
    try:
        for index in range(min(args.num_episodes, len(dataset))):
            sample = dataset[index]
            episode = sample["duplex_episode"]
            run_spec = {
                "episode_id": episode["episode_id"],
                "axis": episode["axis"],
                "seed": sample["duplex_seed"] + index,
                "chunks": episode["chunks"],
                "event": episode["event"],
                "commit": bool(episode.get("commit")),
                "tail_horizon_s": episode.get("tail_horizon_s", 8.0),
            }
            if episode["axis"] == "interruption":
                run_spec["inject"] = {
                    "delay_s": episode.get("inject_delay_s", 0.6),
                    "tail_silence_chunks": episode.get("tail_silence_chunks", 2),
                }
            session = None
            try:
                from verl_omni.workers.rollout.vllm_rollout.vllm_omni_duplex_strategy import DuplexStrategy

                driver = DuplexStrategy.__new__(DuplexStrategy)
                driver.server = type("S", (), {"engine": engine})()
                result = asyncio.run(driver.run_duplex_episode(run_spec))
                score = compute_score_duplex_turn_taking(
                    "calibration", "", "", extra_info=_replay_info(result, processor)
                )
                rewards[episode["axis"]].append(float(score["score"]))
            finally:
                if session is not None and not session.closed:
                    engine.close_session(session.session_id)
    finally:
        engine.shutdown()
    return dict(rewards)


def _replay_info(result: dict, processor) -> dict:
    """Assemble the reward's extra_info from one episode result.

    Args:
        result: One run_duplex_episode result.
        processor: The checkpoint's processor, resolving the framing
            vocabulary for the replay build.

    Returns:
        The replay payload exactly as the training agent loop stamps it (the
        scorer reads the trip-wire tokens at each layout's action_position):
        the same interruption re-anchoring, so the yield reward measures the
        live injection the driver actually made.
    """
    from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY, build_replay, resolve_duplex_vocabulary
    from verl_omni.utils.dataset.duplex.trajectory import DuplexEpisodeEvent, DuplexTrajectory, reanchor_interruption

    trajectory = DuplexTrajectory.assemble(
        session_id=result["session_id"],
        seed=0,
        event=DuplexEpisodeEvent.from_mapping(result.get("event") or {}),
        record_set_wires=result["record_sets"],
        client_audio_chunks=result.get("client_chunks") or [],
    )
    trajectory = reanchor_interruption(
        trajectory,
        speech_chunk_start_index=result.get("speech_chunk_start_index"),
        speech_chunk_end_index=result.get("speech_chunk_end_index"),
        seed=0,
    )
    # The session opened without explicit instructions (the driver defaults
    # in the policy's own system text), so the replay context matches.
    replay = build_replay(trajectory, resolve_duplex_vocabulary(processor), instructions=None)
    return {DUPLEX_REPLAY_KEY: replay.to_extra_fields(include_pcm=False)}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rewards = roll_rewards(args)
    from verl_omni.utils.dataset.duplex.calibration import audit_calibration

    report = audit_calibration(rewards, max_mean_ratio=args.max_mean_ratio)
    Path(args.output).write_text(json.dumps(dict(report), indent=2, default=str), encoding="utf-8")
    print(json.dumps(dict(report), indent=2, default=str))
    if not report.passed:
        print("calibration gate FAILED — fix the event construction or reward band before training", file=sys.stderr)
        return 1
    print("calibration gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
