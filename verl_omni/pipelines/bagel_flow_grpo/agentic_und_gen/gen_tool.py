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

"""The ``generate_image`` loop tool: one GEN turn of S FlowGRPO seeds per call."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from verl_omni.agent_loop.utils import derive_rollout_seed
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.samples import GenSample, gen_sample_uid
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import serving_rollout_provenance
from verl_omni.tools.trajectory.episode import current_episode


class GenerateImageCapError(RuntimeError):
    """Raised when a second GEN call is attempted under ``max_generate_passes=1``."""


def compact_image_observation(path: str) -> str:
    """UND-facing observation: path only; S seed trajectories stay on the GEN batch."""
    return f"path={path}"


def build_generate_call_meta(*, prompt: str, user_prompt: str) -> dict:
    """Build BAGEL call metadata from the current episode's generation history."""
    state = current_episode()
    record = state.latest_generation if state else None
    previous = record.prompt if record else ""
    rewrite = bool(previous)
    return {
        "call_role": "rewrite" if rewrite else "initial",
        "controlled_by_reflection": rewrite,
        "reflection": "",
        "prev_tool_prompt": previous,
        "source_image_for_reflection": (state.resolve_image() or "") if rewrite else "",
        "rewritten_prompt": prompt if rewrite else "",
        "image_generated_from_reflected_prompt": rewrite,
        "tool_prompt_equals_rewritten_prompt": rewrite,
        "content_source": "reflection" if rewrite else "initial",
        "llm_reflection": "",
        "llm_prompt": prompt,
        "model_decode": "",
        "user_prompt": user_prompt or "",
        "tool_prompt": prompt,
    }


def resolve_gen_prompt(prompt: str, *, fixed_gen_prompt: str, active_user_prompt: str) -> str:
    """The diffusion prompt for a ``generate_image`` call, honouring a recipe-fixed generator prompt."""
    if not fixed_gen_prompt:
        return prompt
    prompt = active_user_prompt if fixed_gen_prompt == "__user_request__" else fixed_gen_prompt
    if not prompt:
        raise ValueError("fixed generator prompt requires a nonempty original request")
    return prompt


class BagelGenerateImageTool:
    """One GEN turn per call (RFC ``K`` += 1); refuses calls past ``max_generate_passes``."""

    def __init__(
        self,
        *,
        gen_samples_per_call: int,
        max_generate_passes: int = 1,
        generate_fn: Callable[..., Any] | None = None,
        seed_base: int = 0,
    ):
        if gen_samples_per_call < 1:
            raise ValueError("gen_samples_per_call must be >= 1")
        if max_generate_passes < 1:
            raise ValueError("max_generate_passes must be >= 1")
        self.s = int(gen_samples_per_call)
        self.max_generate_passes = int(max_generate_passes)
        self._passes = 0
        self._generate_fn = generate_fn
        self._seed_base = int(seed_base)
        self._call_index = 0

    @property
    def passes(self) -> int:
        """``generate_image`` calls issued so far in this episode."""
        return self._passes

    def remaining_passes(self) -> int:
        """Calls still allowed under ``max_generate_passes``."""
        return max(0, self.max_generate_passes - self._passes)

    def _default_seeds(self, call_index: int) -> list[int]:
        """Fresh FlowGRPO seeds for one ``generate_image`` turn."""
        return [derive_rollout_seed(self._seed_base, call_index * self.s + seed_index) for seed_index in range(self.s)]

    async def __call__(
        self,
        *,
        prompt: str,
        prompt_token_ids: list[int],
        gen_call_id: str,
        seeds: list[int] | None = None,
    ) -> list[GenSample]:
        """Render S seeds for ``prompt`` and wrap the serving rows as ``GenSample`` records."""
        if self._passes >= self.max_generate_passes:
            raise GenerateImageCapError(
                f"max_generate_passes={self.max_generate_passes} refuses a second generate_image call"
            )
        self._passes += 1
        call_index = self._call_index
        self._call_index += 1
        if seeds is None:
            seeds = self._default_seeds(call_index)
        if len(seeds) != self.s:
            raise ValueError(f"expected S={self.s} FlowGRPO seeds for one GEN turn, got {len(seeds)}")
        if self._generate_fn is None:
            raw_rows: list[dict[str, Any]] = [{"valid": True} for _ in seeds]
        else:
            maybe = self._generate_fn(
                prompt=prompt, prompt_token_ids=prompt_token_ids, seeds=seeds, gen_call_id=gen_call_id
            )
            raw_rows = await maybe if asyncio.iscoroutine(maybe) else maybe
        samples: list[GenSample] = []
        for seed_index, (_seed, row) in enumerate(zip(seeds, raw_rows, strict=True)):
            cond_uid = row.get("cond_uid")
            samples.append(
                GenSample(
                    gen_sample_uid=gen_sample_uid(gen_call_id, seed_index),
                    gen_group_uid=gen_call_id,
                    seed_index=seed_index,
                    valid=bool(row.get("valid", True)),
                    prompt_token_ids=list(prompt_token_ids),
                    all_latents=row.get("all_latents"),
                    timesteps=row.get("timesteps"),
                    rollout_log_probs=row.get("rollout_log_probs"),
                    image_path=row.get("image_path"),
                    call_role="initial",
                    cond_uid=str(cond_uid) if cond_uid else None,
                    cond_len=int(row.get("cond_len") or 0),
                    **serving_rollout_provenance(row),
                )
            )
        return samples
