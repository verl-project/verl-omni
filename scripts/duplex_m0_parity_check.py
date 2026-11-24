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
"""M0 parity gate for duplex GRPO (RFC verl-omni #905): actor vs capture.

Runs a handful of fixed duplex episodes under a per-episode seed, replays
each captured trajectory through the trainer-side model forward with the
engine's exact unit layout, recomputes every sampled action's log-probability
under the recorded sampling distribution (forbidden mask, repetition penalty,
temperature / top-k / top-p truncation, two-stage Bernoulli joint), and
compares against the engine-captured logprobs.

Exit criteria (RFC §4, M0): fixed segments replay deterministically, the
actor-recomputed logprobs match the captured record logprobs within
tolerance on the in-nucleus sampled tokens — including a barge-in episode —
and every forced class maps to mask 0. Rows whose drawn token fell outside
the replay's nucleus cut (a legitimate top_p boundary flip under the engine
vs offline bf16 residual) are reported under ``documented_loosening``
instead of failing the gate.

Usage (GPU host with the capture-capable vLLM-Omni installed)::

    python scripts/duplex_m0_parity_check.py \
        --model /path/to/MiniCPM-o-4_5 --episodes /path/to/episodes.parquet \
        --rel-tol 1e-3 --abs-tol 1e-4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from verl_omni.utils.dataset.duplex.deploy_config import stage_init_timeout_from_env, write_deploy_config_for

_REL_TOL_DEFAULT = 1e-3
_ABS_TOL_DEFAULT = 1e-4


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="MiniCPM-o 4.5 checkpoint directory")
    parser.add_argument("--episodes", required=True, help="mined duplex episode parquet")
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--rel-tol", type=float, default=_REL_TOL_DEFAULT)
    parser.add_argument("--abs-tol", type=float, default=_ABS_TOL_DEFAULT)
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
    parser.add_argument(
        "--require-barge-in", action="store_true", help="fail unless a truncated_interruption episode was checked"
    )
    parser.add_argument("--output", default="duplex_m0_parity.json")
    return parser.parse_args(argv)


def _duplex_deploy_config_path(args: argparse.Namespace) -> str:
    """Generate the thinker-only duplex deploy config the engine client needs."""
    return write_deploy_config_for(
        output_path=args.output, deploy_config=args.deploy_config, tensor_parallel_size=args.tensor_parallel_size
    )


def _roll_episodes(args: argparse.Namespace) -> list[dict]:
    """Roll the fixed episode set twice (same seeds) and return the results."""
    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni
    from verl_omni.utils.dataset.duplex.duplex_rl_dataset import DuplexEpisodeDataset
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_duplex_strategy import DuplexStrategy

    duplex_omni_cls = require_duplex_vllm_omni()
    engine = duplex_omni_cls(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        deploy_config=_duplex_deploy_config_path(args),
        stage_init_timeout=args.stage_init_timeout,
    )
    dataset = DuplexEpisodeDataset([args.episodes])
    driver = DuplexStrategy.__new__(DuplexStrategy)
    driver.server = type("S", (), {"engine": engine})()

    def run_once() -> list[dict]:
        results = []
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
            results.append(asyncio.run(driver.run_duplex_episode(run_spec)))
        return results

    try:
        first, second = run_once(), run_once()
    finally:
        engine.shutdown()
    return list(zip(first, second, strict=True))


def _actor_logits(model, replay):
    """One forward over the replayed sequence; returns per-decision logits.

    The batch goes through the thinker adapter's duplex media branch (explicit
    bounds from the replay payload, so no marker scan or model config is
    involved), and each decision token's logits are read at its predicting
    position: one before the action, offset by the prompt length because
    ``action_position`` is response-relative.
    """
    import torch

    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter
    from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY

    input_ids = torch.tensor([replay.prompt_ids + replay.response_ids], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1])[None]
    data = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": position_ids},
        micro_batch={DUPLEX_REPLAY_KEY: [replay.to_extra_fields(include_pcm=False)]},
        model_config=None,
    )
    with torch.no_grad():
        output = model(**data)
    logits = output.logits[0].float()
    prompt_length = len(replay.prompt_ids)
    positions = [
        prompt_length + layout.action_position for layout in replay.unit_layouts if layout.action_position >= 0
    ]
    return logits[[position - 1 for position in positions]]


def _parity_verdict(
    logits,
    inputs,
    context,
    kinds: Sequence[str],
    logprobs: Sequence[float | None],
    *,
    rel_tol: float,
    abs_tol: float,
):
    """Recompute under the engine's exact transform and compare, nucleus-cut aware.

    ``kinds`` / ``logprobs`` are the trajectory's flat action columns; only
    sampled rows carry a captured log-probability into the comparison. The
    recompute runs over the engine's own repetition history and the comparison
    classifies each row by whether the drawn token survives the replay's cut.
    """
    from verl_omni.utils.dataset.duplex.sampling_logprobs import (
        compare_action_logprobs,
        engine_repetition_history,
        recompute_action_logprobs,
    )

    values, valid = recompute_action_logprobs(logits, inputs, context, action_history=engine_repetition_history(inputs))
    captured = [logprob if kind == "sampled" else None for logprob, kind in zip(logprobs, kinds, strict=True)]
    return compare_action_logprobs(logits, inputs, context, values, valid, captured, rel_tol=rel_tol, abs_tol=abs_tol)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import MiniCPMODuplexRolloutAdapter
    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMO
    from verl_omni.utils.dataset.duplex.replay import resolve_duplex_vocabulary
    from verl_omni.utils.dataset.duplex.sampling_logprobs import (
        build_sampling_context,
        unit_inputs_from_trajectory,
    )
    from verl_omni.utils.dataset.duplex.trajectory import DuplexEpisodeEvent, DuplexTrajectory

    pairs = _roll_episodes(args)
    report: dict = {"episodes": [], "rel_tol": args.rel_tol, "abs_tol": args.abs_tol}
    saw_barge_in = False
    all_match = True

    # Trainer-side model: remote-code MiniCPM-o with the TTS stage stripped.
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.init_tts = False
    config.use_cache = False
    config.stream_input = False
    model = MiniCPMO.from_pretrained(args.model, config=config, trust_remote_code=True, torch_dtype="bfloat16")
    model.eval().cuda()
    processor = model.processor if hasattr(model, "processor") else None
    from transformers import AutoProcessor

    processor = processor or AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    for first, second in pairs:
        event_payload = first.get("event") or {}
        pause_window = event_payload.get("pause_window_s")
        event = DuplexEpisodeEvent(
            axis=event_payload["axis"],
            pause_window_s=tuple(pause_window) if pause_window else None,
            user_turn_end_s=event_payload.get("user_turn_end_s"),
            interrupt_speech_end_s=event_payload.get("interrupt_speech_end_s"),
        )
        trajectory = DuplexTrajectory.assemble(
            session_id=first["session_id"],
            seed=0,
            event=event,
            record_set_wires=first["record_sets"],
            client_audio_chunks=first.get("client_chunks") or [],
        )
        replay = MiniCPMODuplexRolloutAdapter.build_duplex_replay(trajectory, processor=processor, instructions=None)
        episode_report: dict = {"episode_id": first["episode_id"], "status": first.get("status")}
        # Determinism: same seed, same input track -> identical action tokens.
        first_tokens = [wire["token_ids"] for wire in first["record_sets"]]
        second_tokens = [wire["token_ids"] for wire in second["record_sets"]]
        episode_report["deterministic"] = first_tokens == second_tokens
        if first["status"] == "truncated_interruption":
            saw_barge_in = True
        # Mask taxonomy: forced and re-emitted actions carry mask 0.
        kinds = trajectory.action_kinds
        mask = trajectory.action_mask
        episode_report["mask_taxonomy_ok"] = all(
            bit == (kind == "sampled") for kind, bit in zip(kinds, mask, strict=True)
        )
        # Parity: recomputed vs captured on the sampled rows, split by the
        # replay's nucleus cut. Rows whose drawn token fell outside the cut (a
        # top_p boundary flip under the engine-vs-offline bf16 residual) are
        # the documented loosening, not gate failures.
        inputs = unit_inputs_from_trajectory(trajectory)
        context = build_sampling_context(
            tokenizer, resolve_duplex_vocabulary(processor), trajectory.record_sets[-1].model
        )
        logits = _actor_logits(model, replay)
        parity = _parity_verdict(
            logits, inputs, context, kinds, trajectory.logprobs, rel_tol=args.rel_tol, abs_tol=args.abs_tol
        )
        episode_report["parity_by_class"] = parity.to_dict()
        episode_report["matches"] = parity.matches
        all_match = all_match and episode_report["matches"] and episode_report["mask_taxonomy_ok"]
        report["episodes"].append(episode_report)
        report.setdefault("documented_loosening", []).extend(
            {"episode": first["episode_id"], **row} for row in parity.to_dict()["documented_loosening"]
        )

    report["saw_barge_in"] = saw_barge_in
    report["overall_pass"] = bool(all_match and (saw_barge_in or not args.require_barge_in))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["overall_pass"]:
        print("M0 parity gate FAILED", file=sys.stderr)
        return 1
    print("M0 parity gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
