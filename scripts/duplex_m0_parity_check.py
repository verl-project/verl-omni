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
engine's exact unit layout, and compares every sampled action's captured
log-probability against the two-stage policy's pre-truncation joint
evaluated on the actor logits: ``log P(chunk_eos)`` on the raw softmax for
a boundary hit, ``log(1 - P) + log_softmax(mask(z) / T)[a]`` for a content
token, with the forbidden set ``tts_pad + bad_token_ids + chunk_eos``
masked out of stage 2. The engine captures the same joint, so the
comparison is direct — no truncation replay, no exemptions.

Exit criteria (RFC §4, M0): fixed segments replay deterministically, the
per-row |delta| stays within ``--max-abs-nats`` and its mean within
``--mean-abs-nats`` on every sampled row — including a barge-in episode —
and every forced class maps to mask 0. Each episode also reports
``audio_embedding_parity``, the streaming-vs-offline audio embedding gap per
unit, as a diagnostic that does not gate.

Usage (GPU host with the capture-capable vLLM-Omni installed)::

    python scripts/duplex_m0_parity_check.py \
        --model /path/to/MiniCPM-o-4_5 --episodes /path/to/episodes.parquet \
        --max-abs-nats 0.05 --mean-abs-nats 0.01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from verl_omni.utils.dataset.duplex.deploy_config import stage_init_timeout_from_env, write_deploy_config_for
from verl_omni.utils.dataset.duplex.policy_measure import resolve_forbidden_token_ids, two_stage_logprob
from verl_omni.utils.dataset.duplex.records import ACTION_FORCED, ACTION_REEMIT, ACTION_SAMPLED, RecordSetError

_MAX_ABS_NATS_DEFAULT = 0.05
_MEAN_ABS_NATS_DEFAULT = 0.01


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="MiniCPM-o 4.5 checkpoint directory")
    parser.add_argument("--episodes", required=True, help="mined duplex episode parquet")
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--max-abs-nats", type=float, default=_MAX_ABS_NATS_DEFAULT)
    parser.add_argument("--mean-abs-nats", type=float, default=_MEAN_ABS_NATS_DEFAULT)
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


def _actor_model_data(replay) -> dict:
    """The actor model call for one replayed episode, as ``prepare_model_inputs`` builds it.

    The batch goes through the thinker adapter's duplex media branch (explicit
    bounds from the replay payload, so no marker scan or model config is
    involved).
    """
    import torch

    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter
    from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY

    input_ids = torch.tensor([replay.prompt_ids + replay.response_ids], dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1])[None]
    return MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": position_ids},
        micro_batch={DUPLEX_REPLAY_KEY: [replay.to_extra_fields(include_pcm=False)]},
        model_config=None,
    )


def _actor_logits(model, model_inputs: dict, replay):
    """One forward over the replayed sequence; returns per-decision logits.

    Args:
        model: The trainer-side MiniCPM-o.
        model_inputs: The episode's prepared call (``_actor_model_data``), built
            once per episode and shared with the embedding-parity diagnostic.
        replay: The replayed episode, for the decision positions.

    Returns:
        ``[num_decisions, V]`` logits: each decision token's logits are read at
        its predicting position — one before the action, offset by the prompt
        length because ``action_position`` is response-relative.
    """
    import torch

    with torch.no_grad():
        output = model(**model_inputs)
    logits = output.logits[0].float()
    prompt_length = len(replay.prompt_ids)
    positions = [
        prompt_length + layout.action_position for layout in replay.unit_layouts if layout.action_position >= 0
    ]
    return logits[[position - 1 for position in positions]]


def _audio_embedding_parity(model, model_inputs: dict, *, has_reference: bool) -> dict:
    """Per streaming unit, ``max|emb_streaming − emb_offline|`` over the same mel clip.

    The patched ``get_audio_embedding`` streams the unit clips of a ``data``
    dict carrying ``audio_clip_kinds`` and encodes every clip offline once the
    key is dropped, so both readings come from the same clips. Unit 0 encodes
    without history on either side and should sit near bf16 noise; units ≥ 1
    carry the engine's cross-unit KV history the offline encoder lacks, so
    their gap is the encoder mismatch a residual log-probability delta traces
    to. Diagnostic only: the gate does not read it.

    Args:
        model: The trainer-side MiniCPM-o with the audio patches installed.
        model_inputs: ``_actor_model_data`` for the episode.
        has_reference: Whether clip 0 is the episode's offline reference block.

    Returns:
        ``unit0_max_abs``, ``units_ge1_max_abs`` (``None`` without such a unit)
        and ``per_unit`` — one value per streaming unit in feed order.
    """
    import torch

    from verl_omni.pipelines.minicpm.thinker_training_adapter import split_minicpm_forward_kwargs

    model_data, _ = split_minicpm_forward_kwargs(model_inputs)
    offline_data = {key: value for key, value in model_data.items() if key != "audio_clip_kinds"}
    with torch.no_grad():
        streaming = model.get_audio_embedding(model_data)[0]
        offline = model.get_audio_embedding(offline_data)[0]
    units = slice(1, None) if has_reference else slice(None)
    per_unit = [
        float((streamed.float() - encoded.float()).abs().max())
        for streamed, encoded in zip(streaming[units], offline[units], strict=True)
    ]
    return {
        "unit0_max_abs": per_unit[0] if per_unit else None,
        "units_ge1_max_abs": max(per_unit[1:]) if len(per_unit) > 1 else None,
        "per_unit": per_unit,
    }


def _parity_rows(
    logits, actions, *, chunk_eos_id: int, forbidden_index, max_abs_nats: float, mean_abs_nats: float
) -> dict:
    """Compare actor logits against the captured pre-truncation joint logprobs.

    Args:
        logits: ``[num_actions, V]`` actor logits at the actions' predicting
            positions (``_actor_logits``), in action order.
        actions: Per action, ``(token_id, action_kind, temperature,
            captured_logprob)`` in action order; ``captured_logprob`` is
            ``None`` where no draw happened (forced / reemit rows).

    Returns:
        The episode's parity fields: per-row ``delta`` (with the row's
        ``index`` / ``token_id`` / ``captured`` / ``recomputed`` values),
        ``max_abs_delta``, ``mean_abs_delta``, ``num_compared``,
        ``num_skipped_forced_or_reemit``.

    Raises:
        RecordSetError: On a column misalignment, an unknown action kind, or a
            non-finite comparison value.
    """
    if logits.shape[0] != len(actions):
        raise RecordSetError(f"parity comparison got {logits.shape[0]} logit rows for {len(actions)} actions")
    rows: list[dict] = []
    deltas: list[float] = []
    skipped = 0
    for index, (logits_row, (token_id, kind, temperature, captured)) in enumerate(zip(logits, actions, strict=True)):
        if kind in (ACTION_FORCED, ACTION_REEMIT):
            skipped += 1
            continue
        if kind != ACTION_SAMPLED:
            raise RecordSetError(f"unknown duplex action kind {kind!r} at action {index}")
        recomputed = two_stage_logprob(
            logits_row, token_id, chunk_eos_id=chunk_eos_id, forbidden_index=forbidden_index, temperature=temperature
        )
        if captured is None or not math.isfinite(recomputed) or not math.isfinite(captured):
            raise RecordSetError(
                f"duplex action {index} (token {token_id}) compares non-finite: "
                f"recomputed={recomputed!r}, captured={captured!r}"
            )
        delta = recomputed - captured
        deltas.append(abs(delta))
        rows.append(
            {"index": index, "token_id": token_id, "captured": captured, "recomputed": recomputed, "delta": delta}
        )
    return {
        "num_compared": len(deltas),
        "num_skipped_forced_or_reemit": skipped,
        "max_abs_delta": max(deltas, default=0.0),
        "mean_abs_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import MiniCPMODuplexRolloutAdapter
    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMO
    from verl_omni.utils.dataset.duplex.trajectory import DuplexEpisodeEvent, DuplexTrajectory

    pairs = _roll_episodes(args)
    report: dict = {"episodes": [], "max_abs_nats": args.max_abs_nats, "mean_abs_nats": args.mean_abs_nats}
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
    # The engine composes the same set (MiniCPMO45DuplexPolicy.native_forbidden_token_ids);
    # chunk_eos is its last element by construction.
    forbidden_ids = resolve_forbidden_token_ids(tokenizer)
    chunk_eos_id = forbidden_ids[-1]
    import torch

    forbidden_index = torch.tensor(forbidden_ids, dtype=torch.long)

    for first, second in pairs:
        event_payload = first.get("event") or {}
        event = DuplexEpisodeEvent.from_mapping(event_payload)
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
        # Parity: the captured pre-truncation joint vs the actor logits under
        # the same measure, on every sampled row.
        actions: list[tuple[int, str, float, float | None]] = []
        for record_set in trajectory.record_sets:
            for record in record_set.records():
                temperature = float(record.sampling.get("temperature", 0.7) or 0.0)
                for action in record.actions:
                    actions.append((action.token_id, action.action_kind, temperature, action.logprob))
        model_inputs = _actor_model_data(replay)
        logits = _actor_logits(model, model_inputs, replay)
        parity = _parity_rows(
            logits,
            actions,
            chunk_eos_id=chunk_eos_id,
            forbidden_index=forbidden_index,
            max_abs_nats=args.max_abs_nats,
            mean_abs_nats=args.mean_abs_nats,
        )
        episode_report["parity"] = parity
        episode_report["audio_embedding_parity"] = _audio_embedding_parity(
            model, model_inputs, has_reference=replay.ref_audio is not None
        )
        # The gate needs every sampled row compared: zero compared rows is no
        # evidence, not a pass.
        gate = (
            parity["num_compared"] > 0
            and parity["max_abs_delta"] <= args.max_abs_nats
            and parity["mean_abs_delta"] <= args.mean_abs_nats
        )
        all_match = all_match and gate and episode_report["mask_taxonomy_ok"]
        report["episodes"].append(episode_report)

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
