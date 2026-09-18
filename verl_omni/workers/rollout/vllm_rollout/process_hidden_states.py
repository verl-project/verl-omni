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
"""Runtime monkey-patches that teach vLLM-Omni's AR path to return the
thinker's last-layer, post-final-norm hidden states (the exact tensor
``model.compute_logits`` consumes) for full-vocabulary OPD.

The per-request payloads ride the existing ``multimodal_output`` channel under
the key ``"teacher_hidden_states"``, so no new wire fields are needed:

    runner   : hidden per request -> OmniModelRunnerOutput.multimodal_outputs
    scheduler: -> OmniEngineCoreOutput.multimodal_output
    processor: -> OmniRequestState.mm_accumulated
    client   : -> CompletionOutput.multimodal_output["teacher_hidden_states"]

Chunked prefill is not supported (a request with unscheduled prompt tokens
raises). With the omni prefix cache enabled, requests resuming from a cached
prefix get their full-prompt hidden states rebuilt from the cache.
"""

import logging

import torch

logger = logging.getLogger(__file__)

# Key under which the hidden states ride the multimodal_output channel.
TEACHER_HIDDEN_STATES_KEY = "teacher_hidden_states"
# Flag key carried on the request's SamplingParams.extra_args.
RETURN_FLAG_KEY = "return_hidden_states"

_applied = False


def request_wants_hidden_states(runner, req_id: str) -> bool:
    """True when the request opted in via ``SamplingParams.extra_args``.

    Reads the flag off the request's cached sampling params rather than the
    runner's ``model_intermediate_buffer``, which the engine overwrites with its
    own ``additional_information`` on the request's first scheduled step.
    """
    req_state = runner.requests.get(req_id)
    params = getattr(req_state, "sampling_params", None)
    extra = getattr(params, "extra_args", None)
    return bool(isinstance(extra, dict) and extra.get(RETURN_FLAG_KEY))


def apply_hidden_states_patches() -> None:
    """Install the hidden-state output channel patches (idempotent)."""
    global _applied
    if _applied:
        return

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    _patch_runner(GPUARModelRunner)
    _patch_npu_runner()
    _applied = True
    logger.info("vllm_omni hidden-states patches applied")


def _patch_npu_runner() -> None:
    """Patch the Ascend AR runner, which builds OmniModelRunnerOutput inline.

    NPUARModelRunner.sample_tokens clears ``execute_model_state`` and returns
    either an OmniModelRunnerOutput or an AsyncGPUModelRunnerOutput wrapper.
    We peek the state first, call the original, then attach the per-request
    hidden payloads to the inner model_runner_output.
    """
    try:
        from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner
    except ImportError:
        return  # non-Ascend deployment

    original_sample = NPUARModelRunner.sample_tokens

    def _sample_with_hidden_states(self, grammar_output):
        state = self.execute_model_state
        scheduler_output = state[0] if state is not None else None
        hidden_states = state[4] if state is not None else None

        output = original_sample(self, grammar_output)

        if scheduler_output is None or hidden_states is None:
            return output

        payloads = _collect_hidden_states_payloads(self, scheduler_output, hidden_states)
        if not payloads:
            return output

        # Sync path returns OmniModelRunnerOutput directly; async scheduling
        # wraps it in AsyncGPUModelRunnerOutput (private _model_runner_output).
        target = getattr(output, "_model_runner_output", None) or output
        _merge_into_multimodal_outputs(target, payloads)
        return output

    NPUARModelRunner.sample_tokens = _sample_with_hidden_states


def _patch_runner(runner_cls: type) -> None:
    """Attach hidden-state payloads to OmniModelRunnerOutput.multimodal_outputs."""
    original_build = runner_cls._build_omni_model_runner_output_from_snapshot

    def _build_with_hidden_states(self, **kwargs):
        output = original_build(self, **kwargs)
        if output is None:
            return output

        scheduler_output = kwargs["scheduler_output"]
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            return output

        hidden_payloads = _collect_hidden_states_payloads(self, scheduler_output, hidden_states)
        if not hidden_payloads:
            return output

        _merge_into_multimodal_outputs(output, hidden_payloads)
        return output

    runner_cls._build_omni_model_runner_output_from_snapshot = _build_with_hidden_states


def _merge_prefix_hidden_states(
    runner, scheduler_output, hidden_states, query_start_loc_cpu
) -> dict[str, torch.Tensor] | None:
    """Rebuild full-prompt hidden states from the omni prefix cache."""
    cache = getattr(runner, "omni_prefix_cache", None)
    if cache is None or not getattr(cache, "has_prefix_cached_new_req_ids", lambda: False)():
        return None
    if not getattr(runner, "_model_needs_full_prefix_hidden_states", lambda: True)():
        return None
    return cache.get_merged_hidden_states(
        query_start_loc=query_start_loc_cpu,
        input_batch=runner.input_batch,
        hidden_states=hidden_states,
        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
    )


def _collect_hidden_states_payloads(runner, scheduler_output, hidden_states) -> dict[str, dict]:
    """Slice per-request hidden states [S, D] (bf16, CPU) from the step's hidden tensor.

    Only requests that opted in AND are finishing their prefill in this step
    are captured; partial (chunked) prefills raise instead of emitting
    truncated tensors.
    """
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    snapshot_qsl = getattr(runner, "_snapshot_query_start_loc_cpu", None)
    if callable(snapshot_qsl):
        query_start_loc_cpu = snapshot_qsl()
    else:  # NPU AR runner keeps query_start_loc as a CpuGpuBuffer (vllm.v1.utils)
        qsl = runner.query_start_loc
        qsl_cpu = getattr(qsl, "cpu", None)
        if qsl_cpu is None:
            qsl_cpu = qsl
        query_start_loc_cpu = qsl_cpu() if callable(qsl_cpu) else qsl_cpu

    merged = _merge_prefix_hidden_states(runner, scheduler_output, hidden_states, query_start_loc_cpu)
    payloads: dict[str, dict] = {}
    for req_id, num_tokens in num_scheduled_tokens.items():
        if not request_wants_hidden_states(runner, req_id):
            continue
        request = runner.requests.get(req_id)
        if request is None or request.prompt_token_ids is None:
            continue

        num_prompt_tokens = len(request.prompt_token_ids)
        start_idx = request.num_computed_tokens
        num_tokens = int(num_tokens)

        num_remaining = num_prompt_tokens - start_idx - num_tokens
        if num_remaining > 0:
            raise RuntimeError(
                "return_hidden_states does not support chunked prefill "
                f"(req={req_id}: {num_remaining} prompt tokens left unscheduled). "
                "Launch the teacher engine with enable_chunked_prefill=False."
            )

        req_idx = runner.input_batch.req_id_to_index.get(req_id)
        if req_idx is None:
            continue

        if merged is not None:
            h = merged.get(req_id)
            if h is None or h.shape[0] != num_prompt_tokens:
                raise RuntimeError(
                    "vllm_omni hidden-states patch: merged hidden states for req "
                    f"{req_id} has shape {h.shape if h is not None else None}, "
                    f"expected ({num_prompt_tokens}, D)."
                )
        elif start_idx > 0:
            raise RuntimeError(
                f"return_hidden_states cannot reconstruct the full-prompt hidden states for req {req_id}: "
                f"{start_idx} tokens were already computed (prefix cache hit) but the merged hidden "
                "states are unavailable. The omni prefix cache must be enabled, and the model must "
                "not opt out via requires_full_prefix_cached_hidden_states=False."
            )
        else:
            offset = int(query_start_loc_cpu[req_idx])
            h = hidden_states[offset : offset + num_tokens]

        payloads[req_id] = {TEACHER_HIDDEN_STATES_KEY: h.detach().cpu()}

    return payloads


def _merge_into_multimodal_outputs(output, hidden_payloads: dict[str, dict]) -> None:
    """Merge hidden payloads into the per-request multimodal_outputs list."""
    req_id_to_index = output.req_id_to_index
    num_reqs = len(output.req_ids)

    mm: list = list(output.multimodal_outputs) if output.multimodal_outputs else [None] * num_reqs
    while len(mm) < num_reqs:
        mm.append(None)

    for req_id, payload in hidden_payloads.items():
        idx = req_id_to_index.get(req_id)
        if idx is None or idx >= num_reqs:
            continue
        entry = mm[idx]
        if entry is None:
            mm[idx] = payload
        elif isinstance(entry, dict):
            entry = dict(entry)
            entry.update(payload)
            mm[idx] = entry

    output.multimodal_outputs = mm


def extract_hidden_states(req_output) -> object:
    """Read the hidden states off a client-facing OmniRequestOutput.

    Looks at the first completion's ``multimodal_output`` payload (a
    MultimodalPayload or dict). Returns the tensor or None.
    """
    outputs = getattr(req_output, "outputs", None)
    if not outputs:
        return None
    mm = getattr(outputs[0], "multimodal_output", None)
    if mm is None:
        return None
    try:
        return mm.get(TEACHER_HIDDEN_STATES_KEY)
    except AttributeError:
        if isinstance(mm, dict):
            return mm.get(TEACHER_HIDDEN_STATES_KEY)
        return None
