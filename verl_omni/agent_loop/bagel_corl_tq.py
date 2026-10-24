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
"""Dual-lane TransferQueue packing for Bagel Co-RL (Joint-Training) (UND episode key vs GEN seed keys).

In-episode counters (RFC):
  J = UND policy turns; K = generate_image calls; J >= K.
  S = seeds per call (FlowGRPO group size).
  N = sibling episodes (rollout.n) — not packed here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import ray
import torch
import transfer_queue as tq
from verl.experimental.agent_loop import AgentLoopManager, AgentLoopOutput
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _settle_session_tasks
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from verl_omni.agent_loop.bagel_corl_lib import (
    EpisodeRollout,
    GenSample,
    _as_gen_sample,
    episode_und_reward,
    strip_pixels_for_actor,
    und_episode_rm_scores,
)
from verl_omni.agent_loop.bagel_corl_rm import bind_bagel_rm_handles
from verl_omni.agent_loop.diffusion_agent_loop import _config_to_sampling_dict
from verl_omni.tools.trajectory.hydra_env import bind_agentic_image_gen
from verl_omni.tools.trajectory.paths import bind_run_artifacts
from verl_omni.utils.agentic_val_viz import VAL_VIZ_PROMPT_KEY, resolve_agentic_val_viz_provider

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Ray forbids subclassing an already-@ray.remote actor class. Inherit the
# underlying AgentLoopWorkerTQ implementation, then re-wrap with @ray.remote.
_AgentLoopWorkerTQBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class


def _optional_val_field(val_kwargs: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from ``val_kwargs`` when present, else ``default`` -- never raising.

    The worker holds ``val_kwargs`` as a **struct** OmegaConf node containing only the
    keys the recipe actually set.  In struct mode a missing key raises
    ``ConfigAttributeError`` rather than yielding ``None``, so a plain attribute read is
    unsafe.

    The dataclass form (``DiffusionSamplingConfig``) always carries the field, so both
    shapes must work and a missing key must degrade to "not set".
    """
    if val_kwargs is None:
        return default
    try:
        return getattr(val_kwargs, name)
    except Exception:  # noqa: BLE001 - a config probe must never fail a rollout
        pass
    try:
        return val_kwargs[name]
    except Exception:  # noqa: BLE001
        return default


def val_overlay_sampling_params(sampling_params: dict, rollout_config: Any) -> dict:
    """Return ``sampling_params`` with the validation diffusion kernel overlaid.

    ``AgentLoopWorkerTQ.generate_sequences`` builds ``sampling_params`` from the
    *training* rollout config and, when validating, overrides only the three AR knobs
    (``top_p`` / ``top_k`` / ``temperature``).  ``val_kwargs.pipeline`` and
    ``val_kwargs.algo`` are never applied, so the recipe's ``VAL_NOISE_LEVEL`` /
    ``VAL_GEN_STEPS`` / val ``cfg_*`` never reach the diffusion request:
    ``build_gen_sampling_params`` then fills those keys from the training
    ``pipeline`` / ``algo``, and validation denoises with the *training* kernel --
    exactly as stochastic as training and non-reproducible between runs.

    ``logprobs=False`` is required, not cosmetic: ``build_gen_sampling_params`` raises
    when a request wants log-probs *and* carries ``noise_level <= 0``, because a
    deterministic decode has no SDE log-probs to stash.  Setting it is also what makes
    it true that ``base["logprobs"]`` is the per-phase answer
    ``bagel_corl_gen_serve`` documents -- the base worker does not set it.  The
    ``val/`` artifact tag in ``BagelMultiturnAgentLoop.run`` keys on this same flag.

    Mirrors ``diffusion_agent_loop_tq``, which performs this same overlay on the stock
    diffusion path (``composite_agent_loop`` and ``diffusion_agent_loop`` likewise);
    the bagel loop was the one path missing it.
    """
    params = dict(sampling_params)
    val_kwargs = getattr(rollout_config, "val_kwargs", None)
    params.update(_config_to_sampling_dict(_optional_val_field(val_kwargs, "pipeline")))
    params.update(_config_to_sampling_dict(_optional_val_field(val_kwargs, "algo")))
    # ``seed`` is set only when the config actually carries one. Absent, the agent loop
    # keeps deriving its per-episode seed from ``global_steps``/sample index -- which is
    # still reproducible for a given step, unlike inventing value that no recipe asked for.
    seed = _optional_val_field(val_kwargs, "seed")
    if seed is not None:
        params["seed"] = seed
    params["logprobs"] = False
    return params


__all__ = [
    "BagelCorlAgentLoopManagerTQ",
    "BagelCorlAgentLoopWorkerTQ",
    "DualLanePack",
    "classify_episode_pattern",
    "episode_from_agent_extra",
    "is_val_viz_prompt",
    "val_overlay_sampling_params",
    "gate_episode_jk",
    "gen_tq_key",
    "pack_dual_lane_episode",
    "put_dual_lane_rows",
    "split_und_gen_from_und_records",
    "split_und_gen_metas",
    "und_row_reward",
    "und_tq_key",
]


def is_val_viz_prompt(prompt: Mapping[str, Any]) -> bool:
    """True when ``prompt`` is a holdout validation-visualization row.

    The holdout is dispatched through this same worker (it needs the episode to run,
    because the agent loop is what dumps the reference PNGs), but it is not a training
    or validation *row*: see ``BagelCorlAgentLoopWorkerTQ._run_holdout_prompt``. The
    marker column is what distinguishes the two, and it travels with the prompt from
    ``AgenticValVizProvider.build_tensordict_batch``, so no worker-side state is needed.

    Absent/blank/``0``/``False`` all read as "not a holdout"; a value this recipe never
    writes (e.g. ``"False"`` from a YAML round-trip) must not switch the holdout on.
    """
    raw = prompt.get(VAL_VIZ_PROMPT_KEY)
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in {"", "0", "false", "none", "no"}


def und_tq_key(*, dataset_task_uid: str, session_id: int | str, episode_index: int | str) -> str:
    return f"{dataset_task_uid}_{session_id}_{episode_index}"


def gen_tq_key(und_key: str, *, gen_call_id: str, seed_index: int) -> str:
    return f"{und_key}::gen::{gen_call_id}::{seed_index}"


def gate_episode_jk(j: int, k: int) -> None:
    if j < 0 or k < 0:
        raise ValueError(f"J and K must be non-negative, got J={j} K={k}")
    if j < k:
        raise ValueError(f"in-episode invariant J >= K violated: J={j} K={k}")


def classify_episode_pattern(j: int, k: int) -> str:
    """Return paired | mixed | gen_off. Raises if J < K."""
    gate_episode_jk(j, k)
    if k == 0:
        return "gen_off"
    if j == k:
        return "paired"
    return "mixed"


@dataclass
class DualLanePack:
    """One episode → one UND record + zero-or-more GEN seed records."""

    und_key: str
    und_record: dict[str, Any]
    gen_records: list[dict[str, Any]] = field(default_factory=list)
    child_gen_keys: list[str] = field(default_factory=list)
    j: int = 0
    k: int = 0
    pattern: str = "gen_off"
    metrics: dict[str, float] = field(default_factory=dict)


def _count_gen_calls(samples: list[GenSample]) -> int:
    groups: set[str] = set()
    for sample in samples:
        if sample.valid:
            groups.add(str(sample.gen_group_uid))
    return len(groups)


def _sample_has_gen_traj(sample: GenSample) -> bool:
    """Complete FlowGRPO seed requires latents + timesteps + rollout logprobs."""
    return (
        sample.valid
        and sample.all_latents is not None
        and sample.timesteps is not None
        and sample.rollout_log_probs is not None
    )


def _complete_groups(samples: list[GenSample], *, expected_s: int) -> dict[str, list[GenSample]]:
    by_group: dict[str, list[GenSample]] = {}
    for sample in samples:
        if not _sample_has_gen_traj(sample):
            continue
        by_group.setdefault(str(sample.gen_group_uid), []).append(sample)
    complete: dict[str, list[GenSample]] = {}
    for gid, rows in by_group.items():
        if len(rows) == expected_s:
            complete[gid] = sorted(rows, key=lambda s: int(s.seed_index))
    return complete


def pack_dual_lane_episode(
    episode: EpisodeRollout,
    *,
    und_key: str,
    expected_s: int,
    dataset_task_uid: str | None = None,
) -> DualLanePack:
    """Pack UND + GEN TQ records. Pattern 3 (K=0) yields zero GEN keys."""
    if expected_s < 1:
        raise ValueError("expected_s must be >= 1")

    j = int(episode.turns)
    if getattr(episode, "num_gen_calls", None) is not None:
        k = int(episode.num_gen_calls)
    else:
        k = _count_gen_calls(episode.gen_samples)
    pattern = classify_episode_pattern(j, k)

    complete = _complete_groups(episode.gen_samples, expected_s=expected_s)
    dropped = max(0, k - len(complete)) if k > 0 else 0
    k_complete = len(complete)

    child_keys: list[str] = []
    gen_records: list[dict[str, Any]] = []
    for gen_call_id, rows in complete.items():
        for sample in rows:
            key = gen_tq_key(und_key, gen_call_id=gen_call_id, seed_index=int(sample.seed_index))
            child_keys.append(key)
            row = strip_pixels_for_actor(
                {
                    "bagel_role": "gen",
                    "episode_uid": episode.episode_uid,
                    "und_group_uid": episode.und_group_uid,
                    "gen_group_uid": sample.gen_group_uid,
                    "gen_sample_uid": sample.gen_sample_uid,
                    "seed_index": sample.seed_index,
                    "parent_und_key": und_key,
                    "uid": sample.gen_group_uid,
                    "prompt_token_ids": list(sample.prompt_token_ids),
                    "all_latents": sample.all_latents,
                    "timesteps": sample.timesteps,
                    "rollout_log_probs": sample.rollout_log_probs,
                    "rm_score": sample.rm_score,
                    "rm_score_source": sample.rm_score_source,
                    "call_role": sample.call_role,
                    "gen_call_index": sample.gen_call_index,
                    "good_enough": sample.good_enough,
                    "image_path": sample.image_path,
                    # RFC §4.4.2a: the reuse identity, per GEN row.
                    "cond_uid": sample.cond_uid,
                    "cond_len": int(sample.cond_len or 0),
                }
            )
            gen_records.append(
                {
                    "key": key,
                    "fields": row,
                    # ``is_auxiliary`` keeps the replay buffer from sampling this seed row into a
                    # training batch: it shares the UND episode's uid but has no token sequence for
                    # the UND pass. The GEN lane reaches the trainer through the UND row's
                    # ``child_gen_keys`` instead.
                    "tag": {"bagel_role": "gen", "is_auxiliary": True, "status": "success"},
                }
            )

    task_uid = dataset_task_uid or episode.und_group_uid
    # Distinct conditionings this episode encoded. R2's win is "S seeds share one
    # conditioning", so the count of *distinct* uids is what the per-call ratio is
    # measured against — a bare per-row ``cond_uid`` cannot distinguish that from
    # S independent conditionings.
    cond_uids = sorted({str(s.cond_uid) for s in episode.gen_samples if s.valid and s.cond_uid})
    und_fields = {
        "bagel_role": "und",
        "episode_uid": episode.episode_uid,
        "und_group_uid": episode.und_group_uid,
        "dataset_task_uid": task_uid,
        "uid": episode.und_group_uid,
        "parent_und_key": und_key,
        "prompt_ids": list(episode.prompt_ids),
        "response_ids": list(episode.response_ids),
        "response_mask": list(episode.response_mask),
        "rollout_log_probs": list(episode.rollout_log_probs),
        "child_gen_keys": list(child_keys),
        "episode_J": j,
        "episode_K": k,
        "episode_pattern": pattern,
        "used_image_credit": bool(episode.used_image_credit),
        "token_level_scores": float(episode.und_reward),
        "num_gen_calls": k,
        "policy_version": int(episode.policy_version),
        "gen_cond_uids": cond_uids,
        "gen_cond_len": int(max((s.cond_len for s in episode.gen_samples if s.valid), default=0)),
    }
    metrics = {
        "episode/J": float(j),
        "episode/K": float(k),
        "episode/K_complete": float(k_complete),
        # K=0 incidence: the episode never issued a single ``generate_image`` call, so the GEN
        # lane had nothing to score and UND fell back to the pattern-3 non-image scalar. Averaged
        # over the sibling batch by ``aggregate_episode_metrics``, this is the step's K=0 *rate*
        # -- distinct from ``gen/skipped_no_groups``, which keys on ``k_complete`` and therefore
        # also counts an episode whose K>=1 groups were all dropped.
        "episode/K_zero": 1.0 if k == 0 else 0.0,
        "gen/num_rows": float(len(gen_records)),
        "gen/dropped_incomplete_groups": float(dropped),
        # ``used_image_credit`` is the authoritative signal (set when a generate_image
        # call returned a usable image and the observation was fed back to UND).
        # Keying off ``k_complete == 0`` under-counted: an episode with K>=1 whose
        # every group was dropped also gave UND no image credit but reported 0 here,
        # while the flatten path (bagel_corl_lib.flatten_multiturn_rollouts) counted it.
        "und/no_image_credit": 0.0 if episode.used_image_credit else 1.0,
        "gen/skipped_no_groups": 1.0 if k_complete == 0 else 0.0,
        "episode/pattern_paired": 1.0 if pattern == "paired" else 0.0,
        "episode/pattern_mixed": 1.0 if pattern == "mixed" else 0.0,
        "episode/pattern_gen_off": 1.0 if pattern == "gen_off" else 0.0,
        # RFC §4.4.4 R2, measured from the engine's cache counters (absent when
        # the engine exposed none — never a fabricated value).
        **dict(episode.r2_metrics or {}),
    }
    return DualLanePack(
        und_key=und_key,
        und_record={"key": und_key, "fields": und_fields, "tag": {"bagel_role": "und", "status": "success"}},
        gen_records=gen_records,
        child_gen_keys=child_keys,
        j=j,
        k=k,
        pattern=pattern,
        metrics=metrics,
    )


def und_row_reward(episode: EpisodeRollout, response_mask: Any) -> torch.Tensor:
    """Build the ``rm_scores`` column for a UND row from the episode's RFC reward.

    Two details have to line up, which is why this is one helper rather than two lines at the
    call site:

    * the column must exist at all -- ``AgentLoopOutput.as_dict`` only publishes ``rm_scores``
      when ``reward_score`` is set (``verl/experimental/agent_loop/agent_loop.py:143``), and the
      v1 UND advantage reads it unconditionally (``trainer_base.py:1595``). The UND ingest
      therefore writes this column itself instead of relying on the optional ``as_dict`` path;
    * the score itself must land on the last *trainable* token (see
      ``bagel_corl_lib.und_episode_rm_scores`` for why ``as_dict``'s ``[-1]`` is wrong here).

    With the served RM disabled (``ENABLE_RM=0`` in the recipe, until Hermes + composite step are
    green) the reward is the RFC ``und/no_image_credit`` scalar: the final rated
    ``generate_image`` call plus the rewrite gain when the RM ran, else the non-image UND
    scalar -- exactly the value ``flatten_multiturn_rollouts`` writes to ``token_level_scores``.
    """
    reward, _ = episode_und_reward(episode)
    return torch.tensor(und_episode_rm_scores(response_mask, reward), dtype=torch.float32)


def split_und_gen_from_und_records(
    und_records: list[Mapping[str, Any]],
    gen_by_key: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """P1 gather: from UND records (+ optional GEN key map) produce und_batch / gen_batch field lists."""
    und_batch: list[dict[str, Any]] = []
    gen_batch: list[dict[str, Any]] = []
    gen_by_key = gen_by_key or {}
    for rec in und_records:
        fields = dict(rec.get("fields") or rec)
        und_batch.append(fields)
        for key in list(fields.get("child_gen_keys") or []):
            gen_rec = gen_by_key.get(key)
            if gen_rec is None:
                continue
            gen_fields = dict(gen_rec.get("fields") or gen_rec)
            gen_batch.append(gen_fields)
    return und_batch, gen_batch


# Plan name alias.
split_und_gen_metas = split_und_gen_from_und_records


def episode_from_agent_extra(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    extra: Mapping[str, Any],
    und_group_uid: str,
    reward_score: float | None = None,
    rollout_log_probs: list[float] | None = None,
) -> EpisodeRollout:
    """Rebuild ``EpisodeRollout`` from ``AgentLoopOutput`` fields / extra_fields.

    ``rollout_log_probs`` rides on ``AgentLoopOutput.response_logprobs`` (it is not part of
    ``extra_fields``), so the caller has to hand it over explicitly; falling back to
    ``extra["response_logprobs"]`` keeps the ``extra``-only rebuild path faithful too.
    """
    raw_samples = extra.get("gen_samples") or []
    samples: list[GenSample] = []
    for item in raw_samples:
        sample = _as_gen_sample(item)
        if sample is not None:
            samples.append(sample)
    turns = int(extra.get("turns") or extra.get("num_turns") or 1)
    num_gen = extra.get("num_gen_calls")
    if num_gen is None:
        num_gen = _count_gen_calls(samples)
    return EpisodeRollout(
        und_group_uid=str(extra.get("und_group_uid") or und_group_uid),
        episode_uid=str(extra.get("episode_uid") or f"{und_group_uid}-ep"),
        policy_version=int(extra.get("policy_version") or 0),
        prompt_ids=list(prompt_ids),
        response_ids=list(response_ids),
        response_mask=list(response_mask),
        rollout_log_probs=list(
            rollout_log_probs if rollout_log_probs is not None else (extra.get("response_logprobs") or [])
        ),
        turns=turns,
        gen_samples=samples,
        used_image_credit=bool(extra.get("used_image_credit", False)),
        autonomous_done=bool(extra.get("autonomous_done", False)),
        forced_reflection=bool(extra.get("forced_reflection", False)),
        corl_reward_version=str(extra.get("corl_reward_version", "v1")),
        bagel_corl_reward=dict(extra.get("bagel_corl_reward") or {}),
        terminal_reward=float(reward_score) if reward_score is not None else None,
        und_reward=float(reward_score if reward_score is not None else extra.get("und_reward") or 0.0),
        num_gen_calls=int(num_gen),
        r2_metrics=dict(extra.get("bagel_corl_r2") or {}),
    )


@ray.remote
class BagelCorlAgentLoopWorkerTQ(_AgentLoopWorkerTQBase):
    """TransferQueue worker that writes dual-lane UND + GEN keys for Bagel Co-RL (Joint-Training)."""

    def __init__(
        self,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
    ):
        """Accept the handles the pinned ``TaskRunnerV1`` forwards via the manager
        (``main_ppo`` passes ``trainer.get_reward_handles()`` unconditionally).

        The signature mirrors ``AgentLoopManager._init_agent_loop_workers``'
        ``.remote(config, llm_client, teacher_client, reward_loop_worker_handles)``
        call positionally. It must NOT be ``(*args, reward_loop_worker_handles=None)``:
        the base call then receives the handle list both positionally and by keyword
        and raises ``TypeError: got multiple values for argument
        'reward_loop_worker_handles'`` before any episode runs.

        ``[0]`` is the GEN pool (the composite contract's first slot, which it
        generically calls "dit"; Bagel is a MoT model with no DiT) and is bound for the
        in-loop RM adapter; the full list stays on ``self`` for the inherited
        episode-reward ``_compute_score`` when the base supports it.
        """
        # Bind before anything reads agentic knobs: ``BagelMultiturnAgentLoop.run``
        # calls ``agentic_get("good_enough_threshold")``, which raises while unbound.
        bind_agentic_image_gen(config)
        # Bind the per-run artifact dir (``trainer.experiment_name`` → run name) so the
        # episode trajectory dumps land in ``<e2e_root>/<experiment_name>/`` -- for this
        # recipe ``outputs/e2e/bagel_corl_pr1/`` -- instead of the ``agentic_run`` default.
        # Without it ``resolve_run_dir()`` falls back to that placeholder name because this
        # recipe never set ``agentic_image_gen.e2e_root``.
        bind_run_artifacts(config)
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        handles = list(reward_loop_worker_handles) if reward_loop_worker_handles else None
        if handles is not None and not hasattr(self, "reward_loop_worker_handles"):
            self.reward_loop_worker_handles = handles
        bind_bagel_rm_handles(handles)

    def _expected_s(self) -> int:
        """Seeds per ``generate_image`` call (RFC §5 knob; no silent default).

        The recipe sets ``agent.gen_samples_per_call``; defaulting to 4 here would
        let the dual-lane GEN seed gate accept a group size nobody configured.
        """
        agent = self.config.actor_rollout_ref.rollout.agent
        raw = agent.get("gen_samples_per_call")
        if raw is None:
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.rollout.agent.gen_samples_per_call; "
                "the dual-lane GEN seed gate cannot infer S"
            )
        return int(raw)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Overlay the validation diffusion kernel before the episode runs.

        ``AgentLoopWorkerTQ.generate_sequences`` builds ``sampling_params`` from the
        *training* rollout config and, when validating, overrides only the three AR knobs
        (``top_p`` / ``top_k`` / ``temperature``). ``val_kwargs.pipeline`` and
        ``val_kwargs.algo`` are never applied, so the recipe's ``VAL_NOISE_LEVEL`` /
        ``VAL_GEN_STEPS`` / val ``cfg_*`` never reach the diffusion request:
        ``build_gen_sampling_params`` then fills those keys from the training
        ``pipeline`` / ``algo``, and validation denoises with the *training* kernel --
        exactly as stochastic as training and non-reproducible between runs.

        ``logprobs=False`` is required, not cosmetic: ``build_gen_sampling_params``
        raises when a request wants log-probs *and* carries ``noise_level <= 0``,
        because a deterministic decode has no SDE log-probs to stash. Setting it is
        also what makes it true that ``base["logprobs"]`` is the per-phase answer
        that ``bagel_corl_gen_serve`` documents -- the base worker does not set it.

        Mirrors ``diffusion_agent_loop_tq``, which performs this same overlay on the
        stock diffusion path (``composite_agent_loop`` and ``diffusion_agent_loop``
        likewise); the bagel loop was the one path missing it.

        A holdout validation-visualization prompt (``AGENTIC_VAL_VIZ=1``) takes the
        overlay *and* skips TransferQueue entirely: see ``_run_holdout_prompt``.
        """
        if trajectory.get("validate"):
            config = self.config.actor_rollout_ref.rollout
            sampling_params = val_overlay_sampling_params(sampling_params, config)
        if not is_val_viz_prompt(prompt):
            await super()._run_prompt(prompt, sampling_params, trajectory=trajectory, trace=trace)
            return
        await self._run_holdout_prompt(prompt, sampling_params, trajectory=trajectory, trace=trace)

    async def _run_holdout_prompt(
        self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False
    ) -> None:
        """Run one holdout-viz prompt to completion without touching TransferQueue.

        The base ``_run_prompt`` brackets every prompt with ``tq.async_kv_put`` status
        markers in ``"train"``/``"val"``, and ``_agent_loop_postprocess`` then writes the
        episode's rows into the same partition. Neither is the holdout's transport:
        ``_validate`` samples ``"val"`` with ``batch_size=len(batch)``, so four extra rows
        there either displace real validation rows or leave the buffer short, and nothing
        ever waits on a holdout's status marker. The holdout's consumers are the PNGs and
        the trajectory the episode dumps itself (``dump_episode_images`` /
        ``dump_episode_trace``, called from ``BagelMultiturnAgentLoop.run``), so this path
        exists to run the sessions and write **no** TransferQueue row at all.

        Because this worker *does* still run the base's fire-and-forget dispatch, the
        manager waits on :meth:`wait_for_background` before it lets the step advance.

        ``n`` mirrors the base: a holdout is a validation batch, so it rolls out
        ``rollout.val_kwargs.n`` siblings.
        """
        config = self.config.actor_rollout_ref.rollout
        n = prompt.pop("__rollout_n__", config.val_kwargs.n)
        prompt.pop("__do_sample__", None)
        tasks = [
            asyncio.create_task(
                self._run_agent_loop(sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt)
            )
            for i in range(n)
        ]
        for error in await _settle_session_tasks(tasks):
            # Same contract as the base: one bad session must not abort its siblings, and
            # a holdout failure must never take the validation pass down with it.
            logger.error(
                "holdout viz prompt failed uid=%s",
                prompt.get("uid"),
                exc_info=(type(error), error, error.__traceback__),
            )

    async def wait_for_background(self) -> None:
        """Block until every episode this worker has accepted has settled.

        ``generate_sequences`` is fire-and-forget by design, so a holdout dispatched
        through it is still decoding when the caller returns. The manager calls this once
        per holdout so the step's reference PNGs exist *before* the trainer advances (and
        re-syncs or sleeps the replicas out from under an in-flight episode).

        The sibling validation batch the trainer is already blocked on is part of the
        same task set and is simply waited on here too; no task can be added after that
        batch because the trainer dispatches the next one only after ``sample`` returns.
        """
        while self.background_tasks:
            await asyncio.gather(*list(self.background_tasks), return_exceptions=True)

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put one UND episode key plus zero-or-more GEN seed keys into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        if is_val_viz_prompt(kwargs):
            # Not a row: the episode's artifacts are already on disk (see
            # ``_run_holdout_prompt``). Scoring it would spend a reward-model round trip on
            # a number nothing logs, so the holdout stops here.
            logger.info(
                "bagel_corl holdout viz episode done uid=%s session=%s (artifacts dumped, no TQ row)",
                uid,
                session_id,
            )
            return
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            raise RuntimeError(f"Empty bagel Co-RL (Joint-Training) agent output for prompt {uid}_{session_id}")

        # The RM manager forwards extra_info only. Overwrite any dataset-supplied
        # lookalike with the actual tool envelope before the live colocated call.
        kwargs = dict(kwargs)
        reward_metadata = dict(kwargs.get("extra_info") or {})
        for key in ("bagel_corl_reward", "corl_reward_version", "autonomous_done", "forced_reflection"):
            reward_metadata.pop(key, None)
        episode_extra = outputs[-1].extra_fields or {}
        reward_metadata["corl_reward_version"] = episode_extra.get("corl_reward_version", "v1")
        if episode_extra.get("corl_reward_version") == "v2":
            envelope = dict(episode_extra["bagel_corl_reward"])
            reward_metadata["bagel_corl_reward"] = envelope
            reward_metadata.update(
                {
                    k: v
                    for k, v in envelope.items()
                    if k.startswith("image_reward_") or k in ("num_gen_calls", "autonomous_done", "forced_reflection")
                }
            )
        kwargs["extra_info"] = reward_metadata
        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for out in outputs[:-1]:
                out.reward_score = final_output.reward_score
                out.extra_fields["reward_extra_info"] = final_output.extra_fields.get("reward_extra_info")

        # Bagel Co-RL (Joint-Training): one serial episode per session → single UND index 0.
        if len(outputs) != 1:
            raise RuntimeError(
                f"Bagel Co-RL (Joint-Training) expected one AgentLoopOutput per session, got {len(outputs)}"
            )
        episode_output = outputs[0]
        und_key = und_tq_key(dataset_task_uid=uid, session_id=session_id, episode_index=0)
        episode = episode_from_agent_extra(
            prompt_ids=list(episode_output.prompt_ids),
            response_ids=list(episode_output.response_ids),
            response_mask=list(episode_output.response_mask),
            extra=episode_output.extra_fields or {},
            und_group_uid=str(uid),
            reward_score=episode_output.reward_score,
            rollout_log_probs=list(episode_output.response_logprobs or []),
        )
        pack = pack_dual_lane_episode(
            episode,
            und_key=und_key,
            expected_s=self._expected_s(),
            dataset_task_uid=str(uid),
        )
        logger.info(
            "bagel_corl dual-lane pack und=%s pattern=%s J=%s K=%s gen_keys=%s metrics=%s",
            und_key,
            pack.pattern,
            pack.j,
            pack.k,
            len(pack.child_gen_keys),
            pack.metrics,
        )

        # UND token row: reuse AgentLoopOutput.as_dict for TQ training tensors.
        und_field = episode_output.as_dict()
        from verl_omni.trainer.omni.bagel_corl_batch import episode_numeric_id

        und_field["bagel_episode_id"] = torch.tensor(episode_numeric_id(episode.episode_uid), dtype=torch.long)

        und_field.update({k: v for k, v in kwargs.items() if k not in und_field})
        und_field.pop("multi_modal_data", None)
        und_field["loss_mask"] = und_field["response_mask"]
        prompts = und_field["prompts"]
        responses = und_field["responses"]
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        multi_modal_inputs = self._compute_multi_modal_inputs(episode_output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        ).squeeze(0)
        und_field["input_ids"] = input_ids
        und_field["position_ids"] = position_ids
        und_field["multi_modal_inputs"] = multi_modal_inputs
        # Dual-lane identity on UND row (no nested gen_samples as train source of truth).
        und_field["uid"] = pack.und_record["fields"]["uid"]
        und_extra = dict(und_field.get("extra_fields") or {})
        und_extra.pop("gen_samples", None)
        und_extra.update(
            {
                "bagel_role": "und",
                "child_gen_keys": list(pack.child_gen_keys),
                "episode_J": pack.j,
                "episode_K": pack.k,
                "episode_pattern": pack.pattern,
                "und_group_uid": pack.und_record["fields"]["und_group_uid"],
                "episode_uid": pack.und_record["fields"]["episode_uid"],
                "bagel_corl_metrics": dict(pack.metrics),
                "min_global_steps": und_extra.get("min_global_steps", kwargs.get("global_steps")),
                "max_global_steps": und_extra.get("max_global_steps", kwargs.get("global_steps")),
            }
        )
        und_field["extra_fields"] = und_extra
        und_field["extra_info"] = dict(reward_metadata)
        if episode.corl_reward_version == "v2":
            und_field["bagel_corl_reward"] = dict(episode.bagel_corl_reward)
            und_field.update(
                {
                    k: v
                    for k, v in episode.bagel_corl_reward.items()
                    if k.startswith("image_reward_") or k in ("num_gen_calls", "autonomous_done", "forced_reflection")
                }
            )
        und_field["child_gen_keys"] = list(pack.child_gen_keys)
        und_field["bagel_role"] = "und"
        # The v1 UND advantage reads ``rm_scores`` off the TQ (``trainer_base.py:1595`` copies it
        # into ``token_level_scores``). It is the RFC ``und/no_image_credit`` episode reward; see
        # ``und_row_reward`` for why the ingest writes the column itself and where it lands.
        und_field["rm_scores"] = und_row_reward(episode, und_field["response_mask"])

        keys = [und_key]
        fields = [und_field]
        tags = [
            {
                "status": "success",
                "bagel_role": "und",
                "prompt_len": int(prompts.size(0)),
                "response_len": int(responses.size(0)),
                "seq_len": int(prompts.size(0) + responses.size(0)),
                "global_steps": kwargs["global_steps"],
                "min_global_steps": und_extra.get("min_global_steps"),
                "max_global_steps": und_extra.get("max_global_steps"),
                "episode_J": pack.j,
                "episode_K": pack.k,
                "episode_pattern": pack.pattern,
            }
        ]

        for gen_rec in pack.gen_records:
            gen_fields = dict(gen_rec["fields"])
            gen_fields["extra_fields"] = {
                "bagel_role": "gen",
                "parent_und_key": und_key,
                "min_global_steps": kwargs.get("global_steps"),
                "max_global_steps": kwargs.get("global_steps"),
            }
            # RM score as a 1-length tensor for TQ consumers that expect rm_scores.
            if gen_fields.get("rm_score") is not None:
                gen_fields["rm_scores"] = torch.tensor([float(gen_fields["rm_score"])], dtype=torch.float32)
            keys.append(gen_rec["key"])
            fields.append(gen_fields)
            tags.append(
                {
                    "status": "success",
                    "bagel_role": "gen",
                    # See ``DualLanePack.gen_records``: auxiliary rows are stored for the GEN lane
                    # (fetched by ``child_gen_keys``) and must never enter a sampled batch.
                    "is_auxiliary": True,
                    "parent_und_key": und_key,
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": kwargs.get("global_steps"),
                    "max_global_steps": kwargs.get("global_steps"),
                    "prompt_len": 0,
                    "response_len": 0,
                    "seq_len": 0,
                }
            )

        await put_dual_lane_rows(
            tq,
            keys=keys,
            field_dicts=fields,
            tags=tags,
            partition_id="train" if not validate else "val",
        )


async def put_dual_lane_rows(
    tq: Any,
    *,
    keys: list[str],
    field_dicts: list[dict[str, Any]],
    tags: list[dict[str, Any]],
    partition_id: str,
    und_row_count: int = 1,
) -> int:
    """Write one episode's dual-lane rows, one ``kv_batch_put`` **per lane**. Returns the put count.

    The UND row and the GEN rows have disjoint schemas — UND carries ``prompts``/``responses``/
    ``input_ids``/``position_ids``/``multi_modal_inputs``, GEN carries ``prompt_token_ids``/
    ``all_latents``/``timesteps``/``seed_index``/``cond_uid`` — and
    ``list_of_dict_to_tensordict`` builds its column dict from the **first** row's keys
    (``verl/utils/tensordict_utils.py:929``):

        keys = list_of_dicts[0].keys()
        dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}

    so packing both lanes into one call dies with ``KeyError: 'prompts'`` on the GEN rows. That
    is not an edge case: it fires on the first episode that actually calls ``generate_image``
    (K >= 1) — precisely the episodes this trainer exists to learn from. Pattern-3 (K = 0)
    episodes hid it, because then the list is a single UND row and the schema is trivially
    uniform.

    ``kv_batch_put`` is per key set (``transfer_queue/interface.py:768``), so splitting by lane
    is a contract-preserving change: both lanes land in the same partition with the same tags,
    and consumers already fetch them by their own keys (UND by ``batch.keys``, GEN by
    ``child_gen_keys``).
    """
    if not (len(keys) == len(field_dicts) == len(tags)):
        raise ValueError(
            f"dual-lane put needs aligned keys/fields/tags, got {len(keys)}/{len(field_dicts)}/{len(tags)}"
        )
    if und_row_count < 0 or und_row_count > len(keys):
        raise ValueError(f"und_row_count={und_row_count} is out of range for {len(keys)} rows")

    puts = 0
    for start, stop in ((0, und_row_count), (und_row_count, len(keys))):
        if start >= stop:
            continue
        await tq.async_kv_batch_put(
            keys=list(keys[start:stop]),
            fields=list_of_dict_to_tensordict(list(field_dicts[start:stop])),
            tags=list(tags[start:stop]),
            partition_id=partition_id,
        )
        puts += 1
    return puts


class BagelCorlAgentLoopManagerTQ(AgentLoopManager):
    """AgentLoopManager wired to ``BagelCorlAgentLoopWorkerTQ`` (dual-lane ingest)."""

    def __init__(self, *args, **kwargs):
        # Bind before AgentLoopManager.__init__ creates the Ray workers: the
        # bagel_multiturn_agent loop reads agentic knobs (good_enough_threshold,
        # max_generate_image_passes) via agentic_get, which raises while unbound.
        # Mirrors OmniAgentLoopManager (verl_omni/agent_loop/omni_agent_loop.py).
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is not None:
            bind_agentic_image_gen(config)
        self.agent_loop_workers_class = BagelCorlAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        # ``AGENTIC_VAL_VIZ=1`` turns on the fixed holdout; ``None`` (the default) makes
        # every method below a no-op. Read once here, in the driver process, so the
        # workers never depend on the env of whoever spawned the Ray runtime.
        self._val_viz_provider = resolve_agentic_val_viz_provider()
        self._val_viz_done_steps: set[int] = set()
        # WARNING, not INFO: the TaskRunner has no handler below WARNING for this logger, so
        # an INFO line here left a run with no holdout indistinguishable from one with it.
        logger.warning(
            "bagel_corl val holdout viz: %s (AGENTIC_VAL_VIZ=%r)",
            "on" if self._val_viz_provider is not None else "off",
            os.getenv("AGENTIC_VAL_VIZ"),
        )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts):
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
        self._maybe_run_val_viz(prompts)

    def _maybe_run_val_viz(self, prompts) -> None:
        """Run the fixed holdout once per validation step when ``AGENTIC_VAL_VIZ=1``.

        ``val-core`` is a scalar mean over the whole val parquet, so a lane can collapse
        into a degenerate-but-scoring mode and barely move it. Four *fixed* prompts --
        samples 9001/9002 (cafe) and 9003/9004 (CN poster), each rendered by the
        deterministic ODE validation kernel -- are the artifact that makes that visible,
        which is what the Mode (2a) RPCO recipe dumps as
        ``rollout_images/step_XXXXXX/sample_9001..9004``. Here they land under
        ``rollout_images/val/step_XXXXXX/sample_900N.01/`` because the bagel lane's
        episode dump prefixes every validation artifact with ``val/``.

        Two details this method exists for:

        * ``_validate`` calls this once per val *batch* (``data.val_max_samples`` /
          ``val_batch_size``), so a per-pass guard keeps one holdout per step instead of
          one per batch.
        * the holdout must survive the call: the workers dispatch episodes
          fire-and-forget, and the trainer may sleep or re-sync the replicas as soon as
          this returns, so the wait is what keeps a reference image from being torn down
          mid-denoise.

        Nothing here may raise: this is an experiment-tracking fixture and the validation
        pass it rides on is the training signal.
        """
        provider = self._val_viz_provider
        if provider is None or not _is_validate_batch(prompts):
            return
        step = _batch_scalar(prompts, "global_steps")
        try:
            step_i = int(step) if step is not None else -1
        except (TypeError, ValueError):
            step_i = -1
        if step_i in self._val_viz_done_steps:
            return
        try:
            batch = provider.build_tensordict_batch(step)
            chunkes = batch.chunk(len(self.agent_loop_workers))
            ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
                ]
            )
            ray.get([worker.wait_for_background.remote() for worker in self.agent_loop_workers])
            self._val_viz_done_steps.add(step_i)
            logger.warning("bagel_corl val holdout viz complete for step=%s", step_i)
        except Exception as exc:  # noqa: BLE001 - a fixture must never fail the val pass
            logger.warning("Failed to run the val holdout viz at step=%s: %s", step_i, exc)


def _batch_scalar(prompts: Any, key: str) -> Any:
    """Read a non-tensor V1 batch scalar (``NonTensorData``) without raising.

    ``_validate`` stamps ``global_steps`` / ``validate`` with ``assign_non_tensor_data``,
    which lands as ``NonTensorData`` holding the raw value (``.data``). A plain tensor or
    a missing key both degrade to ``None`` -- this is called on the hot path of the
    training loop, so it must not be the thing that raises.
    """
    try:
        if key not in prompts:
            return None
        value = prompts[key]
    except Exception:  # noqa: BLE001
        return None
    return getattr(value, "data", value)


def _is_validate_batch(prompts: Any) -> bool:
    """True when ``prompts`` is the validation batch ``_validate`` dispatches."""
    return bool(_batch_scalar(prompts, "validate"))
