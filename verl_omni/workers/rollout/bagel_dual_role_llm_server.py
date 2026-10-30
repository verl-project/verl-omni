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
"""Dual-role LLM client: UND AR + GEN diffusion for Bagel Co-RL (Joint-Training).

One vLLM-Omni replica is AR xor Diffusion (strategy chosen at server init).
Bagel Co-RL (Joint-Training) therefore keeps two ``LLMServerManager`` pools and routes
``generate()`` by sampling-params shape.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from omegaconf import DictConfig
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import (
    _AR_ONLY_SAMPLING_KEYS,
    _force_info_logging,
    fill_und_ar_sampling_params,
)

# ``BAGEL_CORL_DEBUG=1`` adds one entry + one exit line per routed ``generate``.
# The child client (``LLMServerClient.generate``) can sit inside its aborted-request
# retry loop for minutes without returning, and the agent loop only logs *after* a
# decode returns -- an entry with no matching exit names the call that never
# finished, which is the difference between "the engine is slow" and "the client
# never got an answer at all".
_BAGEL_DUAL_ROLE_DEBUG = os.getenv("BAGEL_CORL_DEBUG") == "1"


def _env_float(name: str, default: float) -> float:
    """Read a float knob, treating an empty/unparseable value as ``default``."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return default


# --------------------------------------------------------------------------- Decode watchdog: the only client-side
# valve for a stalled engine. ---------------------------------------------------------------------------
# ``LLMServerClient.generate`` awaits ``server.generate.remote(...)`` with no timeout, and the AR strategy drains the
# engine generator with ``_collect_last_output`` -- which only returns once the orchestrator delivers a terminal
# output for the request.
_DECODE_TIMEOUT_S = _env_float("BAGEL_CORL_DECODE_TIMEOUT_S", 600.0)
_DECODE_HEARTBEAT_S = _env_float("BAGEL_CORL_DECODE_HEARTBEAT_S", 30.0)
_DECODE_MAX_RECOVERIES = int(_env_float("BAGEL_CORL_DECODE_MAX_RECOVERIES", 1.0))

logger = logging.getLogger(__name__)

# Diffusion / FlowGRPO request markers (inverse of AR-only decode knobs).
_GEN_SAMPLING_KEYS = frozenset(
    {
        "num_inference_steps",
        "noise_level",
        "sde_window_size",
        "sde_window_range",
        "sde_type",
        "height",
        "width",
        "cfg_text_scale",
        "cfg_img_scale",
    }
)


def is_bagel_gen_sampling_params(sampling_params: dict[str, Any] | None) -> bool:
    """True when the request is a GEN / FlowGRPO denoise (not UND AR decode)."""
    if not sampling_params:
        return False
    keys = set(sampling_params.keys())
    if keys & _GEN_SAMPLING_KEYS:
        return True
    # Explicit role tag from BagelMultiturnAgentLoop / gen serve helpers.
    role = sampling_params.get("bagel_role") or sampling_params.get("role")
    return str(role).lower() in {"gen", "diffusion", "generate_image"}


def summarize_prompt_embed_cache_counters(results: Any) -> dict[str, Any] | None:
    """Sum per-stage ``PromptEmbedCache.stats()`` payloads into one counter set.

    Each runner process owns its own cache, so a multi-stage deploy reports one
    dict per stage and the meaningful number is the sum. Returns ``None`` when no
    stage reported a cache at all — "no cache installed" and "a cache that has
    served nothing yet" must not both read as zero hits (RFC §4.4.4 R2 gate).
    """
    if results is None:
        return None
    if isinstance(results, dict):
        results = [results]
    totals = {"hits": 0, "misses": 0, "bypassed": 0, "size": 0}
    seen = False
    for item in results:
        if not isinstance(item, dict):
            continue
        if "hits" not in item and "misses" not in item:
            # ``{"supported": False, "error": …}`` from the stage pool, or an
            # engine that has no diffusion runner behind the probe.
            continue
        seen = True
        for key in totals:
            try:
                totals[key] += int(item.get(key) or 0)
            except (TypeError, ValueError):
                logger.warning("prompt-embed cache stats: non-integer %s=%r ignored", key, item.get(key))
    if not seen:
        return None
    totals["installed"] = True
    return totals


_STALL_ERROR_MARKERS = (
    "AR decode stalled",
    "engine yielded no terminal output",
    "decode exceeded",
)


def _is_stall_error(exc: BaseException) -> bool:
    """True when ``exc`` is a stall marker raised by our own decode valve."""
    return any(marker in str(exc) for marker in _STALL_ERROR_MARKERS)


async def _await_decode(task: asyncio.Task[Any], *, route: str, request_id: Any, timeout_s: float) -> Any:
    """Await one decode call, logging heartbeats and enforcing ``timeout_s``.

    Returns the child client's result, re-raising its exception unchanged. Raises
    :class:`asyncio.TimeoutError` when the call outlives ``timeout_s`` (after
    cancelling it, which also runs the child's ``finally: _release_server``).
    """
    t0 = time.perf_counter()
    while True:
        done, _ = await asyncio.wait({task}, timeout=_DECODE_HEARTBEAT_S if _DECODE_HEARTBEAT_S > 0 else None)
        if done:
            return task.result()
        elapsed = time.perf_counter() - t0
        if timeout_s > 0 and elapsed >= timeout_s:
            task.cancel()
            raise TimeoutError(f"decode exceeded {timeout_s:.0f}s")
        logger.warning(
            "bagel_dual_role_waiting route=%s request_id=%s elapsed_s=%.0f: engine has not returned yet "
            "(timeout at %.0fs)",
            route,
            str(request_id)[:48],
            elapsed,
            timeout_s,
        )


class BagelDualRoleLLMServerClient(LLMServerClient):
    """Route ``generate`` to UND AR or GEN diffusion clients.

    Agent loops keep calling ``server_manager.generate``; this client is what
    ``get_llm_client()`` returns for ``bagel_corl_sync``.
    """

    def __init__(
        self,
        config: DictConfig,
        *,
        und_client: LLMServerClient,
        gen_client: LLMServerClient,
    ):
        # No load-balancer of our own; each child owns its pool.
        super().__init__(config=config, load_balancer_handle=None)
        self.und_client = und_client
        self.gen_client = gen_client

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        client = self.gen_client if is_bagel_gen_sampling_params(sampling_params) else self.und_client
        route = "gen" if client is self.gen_client else "und"
        if _BAGEL_DUAL_ROLE_DEBUG:
            # ``vllm`` is imported lazily, on the first real decode, and its logger setup
            # reconfigures the *root* logger with ``force=True``. That drops the INFO/WARNING
            # records this client and the agent loop emit for the rest of the process, which is
            # how a healthy rollout came to look wedged (nothing logged after turn 1 while the
            # engines kept serving). Re-assert INFO logging on entry, i.e. *after* the import
            # that would otherwise silence us.
            _force_info_logging()
        # Strip cross-role knobs so AR/Diffusion strategies do not reject the request.
        params = dict(sampling_params or {})
        if client is self.und_client:
            for key in list(params):
                if key in _GEN_SAMPLING_KEYS or key in {"bagel_role", "role"}:
                    params.pop(key, None)
            # ...and then pin the AR knobs this lane *needs* to the configured recipe. Only
            # stripping left whatever the caller sent, and the generic AR worker always sends
            # ``repetition_penalty=1.0`` -- the as-trained collapse the checkpoint never
            # recovers from (see ``UNTRAINED_AR_SAMPLING``).
            params = fill_und_ar_sampling_params(params, self.config)
        else:
            for key in list(params):
                if key in _AR_ONLY_SAMPLING_KEYS:
                    params.pop(key, None)
        if _BAGEL_DUAL_ROLE_DEBUG:
            logger.info(
                "bagel_dual_role_generate route=%s request_id=%s prompt_tokens=%d budget=%s params=%s",
                route,
                str(request_id)[:48],
                len(prompt_ids),
                params.get("max_tokens", params.get("max_new_tokens")),
                # The sampling params the lane's strategy will actually hand the engine: the only way to tell a
                # tiny/absent ``max_tokens`` or a stray ``stop`` from a model that simply emits EOS.
                {
                    key: params[key]
                    for key in (
                        "temperature",
                        "top_p",
                        "top_k",
                        "repetition_penalty",
                        "max_tokens",
                        "max_new_tokens",
                        "stop",
                        "logprobs",
                    )
                    if key in params
                },
            )
        _t0 = time.perf_counter()
        recoveries = 0
        while True:
            task = asyncio.ensure_future(
                client.generate(
                    request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=params,
                    image_data=image_data,
                    video_data=video_data,
                    audio_data=audio_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    **kwargs,
                )
            )
            try:
                output = await _await_decode(task, route=route, request_id=request_id, timeout_s=_DECODE_TIMEOUT_S)
                break
            except BaseException as exc:  # noqa: BLE001 — only stall shapes are recovered below
                if isinstance(exc, asyncio.CancelledError):
                    raise
                recoverable = isinstance(exc, asyncio.TimeoutError) or _is_stall_error(exc)
                if not recoverable:
                    raise
                recoveries += 1
                logger.error(
                    "bagel_dual_role_stalled route=%s request_id=%s elapsed_s=%.0f params=%s: %s. "
                    "Aborting the lane's in-flight requests and resuming its engine.",
                    route,
                    str(request_id)[:48],
                    time.perf_counter() - _t0,
                    sorted(params),
                    exc,
                )
                await self._abort_and_resume(client, route=route, request_id=request_id)
                if recoveries > _DECODE_MAX_RECOVERIES:
                    raise RuntimeError(
                        f"Bagel Co-RL (Joint-Training) {route} decode stalled {recoveries} time(s) "
                        f"(actor drain timeout / client watchdog {_DECODE_TIMEOUT_S:.0f}s); the {route} engine is "
                        "not draining queued requests. See the bagel_dual_role_stalled / bagel_dual_role_abort "
                        "lines above for the engine's reported in-flight count (0 => the call never reached the "
                        "engine, i.e. client/load-balancer)."
                    ) from exc
                logger.warning(
                    "bagel_dual_role_retry route=%s request_id=%s attempt=%d after engine abort+resume",
                    route,
                    str(request_id)[:48],
                    recoveries,
                )
        if _BAGEL_DUAL_ROLE_DEBUG:
            logger.info(
                "bagel_dual_role_done route=%s request_id=%s out_tokens=%d stop_reason=%s elapsed_s=%.1f",
                route,
                str(request_id)[:48],
                len(getattr(output, "token_ids", None) or []),
                getattr(output, "stop_reason", None),
                time.perf_counter() - _t0,
            )
        return output

    async def _abort_and_resume(self, client: LLMServerClient, *, route: str, request_id: Any) -> None:
        """Abort everything the lane's engine still holds, then un-pause it."""
        server_id = None
        try:
            server_id, server = await client._acquire_server(request_id)
            state = await server.abort_all_requests.remote(reset_prefix_cache=False)
            logger.error(
                "bagel_dual_role_abort route=%s request_id=%s engine_reported=%s",
                route,
                str(request_id)[:48],
                state,
            )
            await server.resume_generation.remote()
        except Exception as exc:  # noqa: BLE001 — recovery must never mask the stall
            logger.error(
                "bagel_dual_role_abort_failed route=%s request_id=%s: %s: %s",
                route,
                str(request_id)[:48],
                type(exc).__name__,
                exc,
            )
        finally:
            if server_id is not None:
                client._release_server(server_id)

    async def prompt_embed_cache_stats(self, routing_key: str) -> dict[str, Any] | None:
        """R2 conditioning-cache counters for the GEN replica ``routing_key`` pins to."""
        try:
            # Protected helper on the shared ``LLMServerClient`` base: the pool's
            # sticky acquisition is exactly the routing ``generate`` performed, so
            # this is the only way to name the replica that served a given call.
            server_id, server = await self.gen_client._acquire_server(routing_key)
        except Exception as exc:  # noqa: BLE001 — metrics only
            logger.warning("prompt-embed cache stats: no GEN replica for key %r: %s", routing_key, exc)
            return None
        try:
            results = await server.prompt_embed_cache_stats.remote()
        except Exception as exc:  # noqa: BLE001 — metrics only
            logger.warning("prompt-embed cache stats probe failed on the GEN engine: %s", exc)
            return None
        finally:
            self.gen_client._release_server(server_id)
        return summarize_prompt_embed_cache_counters(results)
